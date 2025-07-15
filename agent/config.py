import os
from dotenv import load_dotenv
from pathlib import Path
load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASS = os.getenv("NEO4J_PASS")

EMBED_MODEL = os.getenv("EMBED_MODEL")
MODEL_PROVIDER = os.getenv("MODEL_PROVIDER")
LLM_MODEL = os.getenv("LLM_MODEL")
MODEL_KEY = os.getenv("MODEL_KEY", None)
NUM_PAIRS = int(os.getenv("NUM_PAIRS", 5))
LM_MODEL = os.getenv("LM_MODEL")

SESSION_DIR = os.getenv("SESSION_DIR", "./.session_active")
CORPUS_PATH = Path(f"{SESSION_DIR}/corpus.json")
BM25_PATH = Path(f"{SESSION_DIR}/bm25_corpus.pkl")
QDRANT_COLLECTION = os.getenv("enterprise_docs")
PEFT_DIR = Path(f"{SESSION_DIR}/peft_lm_qa")
QA_PATH = Path(f"{SESSION_DIR}/qa_data.json")
CHUNK_SIZE = os.getenv("CHUNK_SIZE", 512)
OVERLAP = os.getenv("OVERLAP", 15)
LORA_R = int(os.getenv("LORA_R", 8))
LORA_ALPHA = int(os.getenv("LORA_ALPHA", 32))
LORA_DROPOUT = float(os.getenv("LORA_DROPOUT", 0.1))
MAX_SEQ_LEN = int(os.getenv("MAX_SEQ_LEN", 384))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", 8))
EPOCHS = int(os.getenv("EPOCHS", 3))
LR = float(os.getenv("LR", 3e-4))