import json
import subprocess
from typing import List
import pickle
import torch
from nltk.tokenize import word_tokenize
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from neo4j import GraphDatabase
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from lm_handler import rerank_chunks_with_llm
from kg_pipeline import extract_facts_from_text, store_facts_to_neo4j, query_kg_multi_hop
from langchain import LLMChain, PromptTemplate
from llm_utils import get_llm
from config import QDRANT_COLLECTION, CORPUS_PATH, BM25_PATH, PEFT_DIR, EMBED_MODEL, LM_MODEL, QDRANT_URL, NEO4J_PASS, NEO4J_USER, NEO4J_URI
device = "cuda" if torch.cuda.is_available() else "cpu"

def query_out(question: str, fallback_bool:bool) -> None:
    corpus = json.loads(CORPUS_PATH.read_text(encoding='utf-8'))
    with open(BM25_PATH,'rb') as f: tokenized=pickle.load(f)
    bm25 = BM25Okapi(tokenized)
    embed_model = SentenceTransformer(EMBED_MODEL)
    qclient = QdrantClient(url=QDRANT_URL)
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))

    tokenizer=model=None
    if PEFT_DIR.exists():
        tokenizer = AutoTokenizer.from_pretrained(LM_MODEL)
        model     = AutoModelForQuestionAnswering.from_pretrained(PEFT_DIR).to(device)

    bm25_res = sorted(bm25.get_top_n(word_tokenize(question.lower()), corpus, n=3))
    dq = embed_model.encode(question).tolist()
    hits = qclient.search(
    collection_name=QDRANT_COLLECTION,
    query_vector=dq,
    limit=3
    )
    
    dense_res = [hit.payload["text"] for hit in hits]
    candidates = list(dict.fromkeys(bm25_res + dense_res))
    print(f"BM42 Answer: {candidates[0]} from {len(candidates)} candidates\n\n")

    if model:
        best,score = None,-1e9
        for c in candidates:
            inp = tokenizer(question, c, return_tensors='pt', truncation=True).to(device)
            out = model(**inp)
            sc = out.start_logits.max()+out.end_logits.max()
            if sc>score: score, best = sc, inp.input_ids[0][out.start_logits.argmax():out.end_logits.argmax()+1]
        ans = tokenizer.decode(best) if score>0 else None
        if ans: print("Finetuned Model Answer:", ans, "\n\n")

    with driver.session() as sess:
        rec = sess.run(f"MATCH (a)-[r]->(b) WHERE toLower(a.name) CONTAINS '{question.lower()}' RETURN a.name,type(r),b.name LIMIT 1").single()
        if rec: print(f"KG Answer: {rec['a.name']} -[{rec['type(r)']}]-> {rec['b.name']}\n\n")

    if fallback_bool:

        print("Trying LLM fallback...")
        top_chunks = rerank_chunks_with_llm(question, candidates)
        fallback_prompt = f"Answer the following question using these most relevant facts:" + "".join(top_chunks[:2]) + f"Question: {question}"
        llm = get_llm()
        chain = LLMChain(llm=llm, prompt=None)
        res = chain.run(prompt=fallback_prompt)
        llm_answer = res.stdout.strip()
        

        top_chunks, paths = process_query_with_rerank_and_kg(question, candidates, llm_answer)
        print("Top Chunks after LLM rerank:", top_chunks, "\n\n")
        print("Knowledge Graph Paths:", paths, "\n\n")
        print("\nLLM Answer:", llm_answer, "\n\n")

        return [llm_answer, top_chunks[0]]
    
    return [ans, candidates[0]]

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

    return top_chunks, paths