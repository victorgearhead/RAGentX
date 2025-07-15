import os
import uuid
import re
import ast
import json
import subprocess
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import pickle
import os
from dotenv import load_dotenv
import torch
import nltk
import spacy
from nltk.tokenize import word_tokenize
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from neo4j import GraphDatabase
from transformers import (
    BertTokenizerFast,
    BertForQuestionAnswering,
    TrainingArguments,
    Trainer,
    default_data_collator
)
from peft import get_peft_model, LoraConfig, TaskType

nltk.download("punkt")
nlp = spacy.load("en_core_web_sm")
device = "cuda" if torch.cuda.is_available() else "cpu"
load_dotenv()
CORPUS_PATH = Path("corpus.json")
BM25_PATH = Path("bm25_corpus.pkl")
QDRANT_COLLECTION = "enterprise_docs"
PEFT_DIR = Path("peft_bert_qa")
QA_PATH = Path("qa_data.json")

class QADataset(torch.utils.data.Dataset):
    def __init__(self, features):
        self.features = features
    def __len__(self):
        return len(self.features)
    def __getitem__(self, idx):
        return self.features[idx]


def extract_pairs(output: str) -> List[Tuple[str,str]]:
    text = output.replace("```","")
    text = text.replace("'", '"')
    m = re.search(r"(\[.*?\])", text, re.DOTALL)
    if not m:
        print("[!] No JSON array found in model output.")
        return []
    try:
        data = ast.literal_eval(m.group(1))
    except:
        return []
    pairs = []
    for d in data:
        q = d.get('question') or d.get("question")
        a = d.get('answer') or d.get("answer")
        if not q or not a:
            keys = list(d.keys())
            if len(keys) >= 2:
                q, a = d[keys[0]], d[keys[1]]
        if q and a:
            pairs.append((str(q).strip(), str(a).strip()))
    return pairs


def generate_qa_for_chunk(text_chunk: str, model_name="llama3.1:latest", num_pairs=5) -> List[Tuple[str,str]]:
    prompt = f"Generate {num_pairs} question-answer pairs in valid JSON format as an array of objects, each with keys \"question\" and \"answer\", from the following text: {text_chunk[:2000]}"
    print(f"Generating {num_pairs} QA pairs for chunk of size {len(text_chunk)}")
    cmd = ["ollama","run", model_name, prompt]
    res = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8')
    print("LLM Output Successful")
    return extract_pairs(res.stdout)

def rerank_chunks_with_llm(query: str, chunks: List[str], model="llama3.1:latest") -> List[str]:
    prompt = f"""You are a helpful assistant. User asked: {query} Here are some document snippets:{chr(10).join(f'{i+1}. {chunk}' for i, chunk in enumerate(chunks))} Rank the top 2 most relevant chunks to the query and return as JSON: [{{"rank": 1, "text": "..."}}, {{"rank": 2, "text": "..."}}]"""

    res = subprocess.run([
        "ollama", "run", model, "-p", prompt, "--max-tokens", "512", "--temp", "0.4"
    ], capture_output=True, text=True, encoding="utf-8")

    try:
        data = json.loads(re.search(r'(\[.*?\])', res.stdout, re.DOTALL).group(1))
        return [item['text'] for item in data if 'text' in item]
    except Exception as e:
        print("[Rerank LLM] Failed to parse rerank result:", e)
        return chunks[:2]

def extract_facts_from_text(text: str) -> List[Dict[str, str]]:
    doc = nlp(text)
    triples = []
    for sent in doc.sents:
        subj, pred, obj = None, None, None
        for token in sent:
            if "subj" in token.dep_:
                subj = token.text
            elif "obj" in token.dep_:
                obj = token.text
            elif token.pos_ == "VERB":
                pred = token.lemma_
        if subj and pred and obj:
            triples.append({"subject": subj, "predicate": pred, "object": obj})
    return triples

def store_facts_to_neo4j(triples: List[Dict[str, str]], uri=os.getenv("NEO4J_URI"), user=os.getenv("NEO4J_USER"), password=os.getenv("NEO4J_PASS")):
    driver = GraphDatabase.driver(uri, auth=(user, password))
    with driver.session() as session:
        for t in triples:
            session.run("""
                MERGE (s:Entity {name: $subj})
                MERGE (o:Entity {name: $obj})
                MERGE (s)-[:REL {type: $pred}]->(o)
            """, subj=t['subject'], pred=t['predicate'], obj=t['object'])
    driver.close()

def query_kg_multi_hop(question: str, uri=os.getenv("NEO4J_URI"), user=os.getenv("NEO4J_USER"), password=os.getenv("NEO4J_PASS")) -> List[str]:
    driver = GraphDatabase.driver(uri, auth=(user, password))
    cypher = """
        MATCH (a:Entity)-[r1]->(b:Entity)-[r2]->(c:Entity)
        WHERE a.name CONTAINS $q OR b.name CONTAINS $q OR c.name CONTAINS $q
        RETURN a.name AS from, type(r1) AS via1, b.name AS mid, type(r2) AS via2, c.name AS to
        LIMIT 5
    """
    with driver.session() as session:
        results = session.run(cypher, q=question)
        paths = [f"{r['from']} -[{r['via1']}]-> {r['mid']} -[{r['via2']}]-> {r['to']}" for r in results]
    driver.close()
    return paths

def process_query_with_rerank_and_kg(question: str, candidates: List[str], answer: str):
    print("\n[Phase 1] Reranking retrieved candidates with LLM...")
    top_chunks = rerank_chunks_with_llm(question, candidates)

    print("\n[Phase 2] Extracting facts from final answer...")
    triples = extract_facts_from_text(answer)
    if triples:
        store_facts_to_neo4j(triples)
        print(f"[KG] Stored {len(triples)} new triples from LLM/QA answer.")

    print("\n[Phase 3] Multi-hop graph reasoning:")
    paths = query_kg_multi_hop(question)
    for p in paths:
        print(" -", p)

def cmd_ingest(args):
    from llama_index.core import SimpleDirectoryReader
    from llama_index.core.node_parser import SentenceSplitter
    docs = SimpleDirectoryReader(args.data_dir).load_data()
    splitter = SentenceSplitter(chunk_size=args.chunk_size, chunk_overlap=args.overlap)
    nodes = splitter.get_nodes_from_documents(docs)
    for node in nodes:
        node.metadata['id'] = str(uuid.uuid4())

    embed_model = SentenceTransformer(args.embedding_model)
    qclient = QdrantClient(url="http://localhost:6333")
    qclient.recreate_collection(
        collection_name=QDRANT_COLLECTION,
        vectors_config=VectorParams(size=embed_model.get_sentence_embedding_dimension(), distance=Distance.COSINE)
    )
    corpus = []
    tokenized = []
    qa_data = []

    for node in nodes:
        txt = node.text
        corpus.append(txt)
        tokenized.append(word_tokenize(txt.lower()))
        vec = embed_model.encode(txt).tolist()
        qclient.upsert(
            collection_name=QDRANT_COLLECTION,
            points=[PointStruct(id=node.metadata['id'], vector=vec, payload={'text': txt})]
        )
        for q, a in generate_qa_for_chunk(txt):
            qa_data.append({"context": txt, "question": q, "answer": a})

    CORPUS_PATH.write_text(json.dumps(corpus), encoding='utf-8')
    with open(BM25_PATH, 'wb') as f:
        pickle.dump(tokenized, f)
    QA_PATH.write_text(json.dumps(qa_data, indent=2), encoding='utf-8')

    print(f"Ingested {len(nodes)} chunks, and generated {len(qa_data)} QA pairs.")

    driver = GraphDatabase.driver(args.kg_uri, auth=(args.kg_user, args.kg_pass))
    def create_rel(tx, e1, rel, e2):
        tx.run("MERGE (a:Entity {name:$e1}) MERGE (b:Entity {name:$e2}) MERGE (a)-[r:"+rel+"]->(b)", e1=e1, e2=e2)
    with driver.session() as sess:
        print("Building knowledge graph...")
        for node in nodes:
            ents = [(ent.text, ent.label_) for ent in nlp(node.text).ents]
            for i in range(len(ents)-1):
                e1,l1 = ents[i]; e2,l2 = ents[i+1]
                rel = f"{l1}_TO_{l2}".upper()
                sess.write_transaction(create_rel, e1, rel, e2)
    print("Knowledge graph built.")

    args.qa_json = str(QA_PATH)
    cmd_train(args)

def cmd_train(args):
    with open(args.qa_json, 'r', encoding='utf-8') as f:
        qa_data = json.load(f)

    tokenizer = BertTokenizerFast.from_pretrained(args.language_model)
    model = BertForQuestionAnswering.from_pretrained(args.language_model)

    peft_conf = LoraConfig(
        task_type=None,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout
    )
    model = get_peft_model(model, peft_conf).to(device)

    features = []
    for ex in qa_data:
        inputs = tokenizer(
            ex["question"],
            ex["context"],
            truncation="only_second",
            max_length=args.max_seq_len,
            padding="max_length",
            return_offsets_mapping=True,
            return_tensors="pt"
        )
        offsets = inputs.pop("offset_mapping")[0]
        text = ex["context"]
        answer = ex["answer"]
        start_char = text.find(answer)
        end_char = start_char + len(answer)

        start_token, end_token = 0, 0
        for idx, (s, e) in enumerate(offsets.tolist()):
            if s <= start_char < e:
                start_token = idx
            if s < end_char <= e:
                end_token = idx

        inputs["start_positions"] = torch.tensor([start_token])
        inputs["end_positions"]   = torch.tensor([end_token])

        features.append({k: v.squeeze(0) for k, v in inputs.items()})

    class QADataset(torch.utils.data.Dataset):
        def __init__(self, features):
            self.features = features
        def __len__(self):
            return len(self.features)
        def __getitem__(self, idx):
            return self.features[idx]

    train_dataset = QADataset(features)

    training_args = TrainingArguments(
        output_dir=PEFT_DIR,                  
        evaluation_strategy="no",             
        learning_rate=args.lr,                
        per_device_train_batch_size=args.batch_size, 
        num_train_epochs=args.epochs,        
        weight_decay=0.01,
        save_total_limit=1,
        logging_steps=50,
        push_to_hub=False
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=default_data_collator,
        tokenizer=tokenizer
    )

    trainer.train()

    model.save_pretrained(PEFT_DIR)
    print(f"BERT fine‑tuned adapters saved at {PEFT_DIR}")


def cmd_query(args):
    corpus = json.loads(CORPUS_PATH.read_text(encoding='utf-8'))
    with open(BM25_PATH,'rb') as f: tokenized=pickle.load(f)
    bm25 = BM25Okapi(tokenized)
    embed_model = SentenceTransformer(args.embedding_model)
    qclient = QdrantClient(url="http://localhost:6333")
    driver = GraphDatabase.driver(args.kg_uri, auth=(args.kg_user, args.kg_pass))

    bert_tok=bert_model=None
    if PEFT_DIR.exists():
        bert_tok = BertTokenizerFast.from_pretrained(args.language_model)
        bert_model = BertForQuestionAnswering.from_pretrained(PEFT_DIR).to(device)

    bm25_res = sorted(bm25.get_top_n(word_tokenize(args.question.lower()), corpus, n=3))
    dq = embed_model.encode(args.question).tolist()
    hits = qclient.search(
    collection_name=QDRANT_COLLECTION,
    query_vector=dq,
    limit=3
    )
    
    dense_res = [hit.payload["text"] for hit in hits]
    candidates = list(dict.fromkeys(bm25_res + dense_res))

    if bert_model:
        best,score = None,-1e9
        for c in candidates:
            inp = bert_tok(args.question, c, return_tensors='pt', truncation=True).to(device)
            out = bert_model(**inp)
            sc = out.start_logits.max()+out.end_logits.max()
            if sc>score: score, best = sc, inp.input_ids[0][out.start_logits.argmax():out.end_logits.argmax()+1]
        ans = bert_tok.decode(best) if score>0 else None
        if ans: print("BERT Answer:", ans); return

    with driver.session() as sess:
        rec = sess.run(f"MATCH (a)-[r]->(b) WHERE toLower(a.name) CONTAINS '{args.question.lower()}' RETURN a.name,type(r),b.name LIMIT 1").single()
        if rec: print(f"KG Answer: {rec['a.name']} -[{rec['type(r)']}]-> {rec['b.name']}"); return

    print("Trying LLM fallback...")
    top_chunks = rerank_chunks_with_llm(args.question, candidates)
    fallback_prompt = f"Answer the following question using these most relevant facts:" + "".join(top_chunks[:2]) + f"Question: {args.question}"
    res = subprocess.run(["ollama", "run", args.language_model, fallback_prompt], capture_output=True, text=True, encoding="utf-8")
    llm_answer = res.stdout.strip()
    print("\nLLM Answer:", llm_answer)

    process_query_with_rerank_and_kg(args.question, candidates, llm_answer)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(prog='enterprise.py')
    sub = parser.add_subparsers()

    p1 = sub.add_parser('ingest'); p1.add_argument('--data-dir', default='data/')
    p1.add_argument('--chunk-size', type=int, default=512)
    p1.add_argument('--overlap', type=float, default=15)
    p1.add_argument('--kg-uri', default=os.getenv("NEO4J_URI"))
    p1.add_argument('--kg-user', default=os.getenv("NEO4J_USER"))
    p1.add_argument('--kg-pass', default=os.getenv("NEO4J_PASS"))
    p1.add_argument('--embedding-model', default=os.getenv("EMBED_MODEL"))
    p1.add_argument('--language-model', default=os.getenv("LAN_MODEL"))
    p1.add_argument('--batch-size', type=int, default=8)
    p1.add_argument('--epochs', type=int, default=3)
    p1.add_argument('--lr', type=float, default=3e-4)
    p1.add_argument('--lora-r', type=int, default=8)
    p1.add_argument('--lora-alpha', type=int, default=32)
    p1.add_argument('--lora-dropout', type=float, default=0.1)
    p1.add_argument('--max-seq-len', type=int, default=384)
    p1.set_defaults(func=cmd_ingest)

    p3 = sub.add_parser('query'); p3.add_argument('question')
    p3.add_argument('--kg-uri', default=os.getenv("NEO4J_URI"))
    p3.add_argument('--kg-user', default=os.getenv("NEO4J_USER"))
    p3.add_argument('--kg-pass', default=os.getenv("NEO4J_PASS"))
    p3.add_argument('--embedding-model', default=os.getenv("EMBED_MODEL"))
    p3.add_argument('--language-model', default=os.getenv("LAN_MODEL"))
    p3.set_defaults(func=cmd_query)

    args = parser.parse_args()
    if hasattr(args, 'func'):
        args.func(args)
    else:
        parser.print_help()
