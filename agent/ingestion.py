import uuid
import json
import pickle
from nltk.tokenize import word_tokenize
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from config import QDRANT_COLLECTION, CORPUS_PATH, BM25_PATH, QA_PATH, CHUNK_SIZE, OVERLAP, EMBED_MODEL, QDRANT_URL
from qa_instruction import generate_qa_for_chunk
from kg_pipeline import build_kg
from agent.lm_handler import finetune_lm

def ingest_in(data_dir, chunk_size: int = CHUNK_SIZE, overlap: int = OVERLAP):
    from llama_index.core import SimpleDirectoryReader
    from llama_index.core.node_parser import SentenceSplitter
    docs = SimpleDirectoryReader(data_dir).load_data()
    splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    nodes = splitter.get_nodes_from_documents(docs)
    for node in nodes:
        node.metadata['id'] = str(uuid.uuid4())

    embed_model = SentenceTransformer(EMBED_MODEL)
    qclient = QdrantClient(url=QDRANT_URL)
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
    build_kg(nodes)
    finetune_lm()
    
    return nodes