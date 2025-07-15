import os
import requests
import streamlit as st
from agent.config import SESSION_DIR, EMBED_MODEL, MODEL_PROVIDER, LLM_MODEL, MODEL_KEY, NUM_PAIRS, QDRANT_URL, NEO4J_URI, NEO4J_USER, NEO4J_PASS, CHUNK_SIZE, OVERLAP, LORA_R, LORA_ALPHA, LORA_DROPOUT, MAX_SEQ_LEN, BATCH_SIZE, EPOCHS, LR, LM_MODEL, QDRANT_COLLECTION

def backend_url(path):
    return f"http://localhost:8000{path}"

if 'session_active' not in st.session_state:
    st.session_state.session_active = False

st.title("🚀 RAGentX")

st.sidebar.header("Session Management")
if not st.session_state.session_active:
    if st.sidebar.button("Start Session"):
        SESSION_DIR = st.sidebar.text_input("Path for Cache and Agent", SESSION_DIR)
        res = requests.post(backend_url('/session/start'))
        if res.ok:
            st.session_state.session_active = True
            st.sidebar.success("Session started")
        else:
            st.sidebar.error("Failed to start session")
else:
    if st.sidebar.button("End Session"):
        res = requests.post(backend_url('/session/end'))
        if res.ok:
            st.session_state.session_active = False
            st.sidebar.success("Session ended")
        else:
            st.sidebar.error("Failed to end session")

st.sidebar.markdown("---")

st.sidebar.header("Configuration")
EMBED_MODEL = st.sidebar.text_input("Embedding Model", EMBED_MODEL)
LM_MODEL = st.sidebar.text_input("Model to Finetune", LM_MODEL)
MODEL_PROVIDER = st.sidebar.selectbox("LLM Provider", ["ollama","openai","huggingface"], index=["ollama","openai","huggingface"].index(MODEL_PROVIDER))
LLM_MODEL = st.sidebar.text_input("LLM Model", LLM_MODEL)
MODEL_KEY = st.sidebar.text_input("Model Key (if needed)", MODEL_KEY or "")
NUM_PAIRS = st.sidebar.number_input("Number of QA Pairs", min_value=1, value=NUM_PAIRS)
st.sidebar.markdown("---")
CHUNK_SIZE = st.sidebar.number_input("Chunk Size", min_value=128, value=int(CHUNK_SIZE))
OVERLAP = st.sidebar.slider("Overlap %", min_value=0, max_value=50, value=int(OVERLAP))
NEO4J_URI = st.sidebar.text_input("NEO4J_URI", NEO4J_URI)
NEO4J_USER = st.sidebar.text_input("NEO4J_USER", NEO4J_USER)
NEO4J_PASS = st.sidebar.text_input("NEO4J_PASS", NEO4J_PASS)
QDRANT_URL = st.sidebar.text_input("QDRANT_URL", QDRANT_URL)
QDRANT_COLLECTION = st.sidebar.text_input("QDRANT Collection Name", QDRANT_COLLECTION)
st.sidebar.markdown("---")
LORA_R = st.sidebar.number_input("LORA_R", min_value=0, value=int(LORA_R))
LORA_ALPHA = st.sidebar.number_input("LORA_ALPHA", min_value=8, value=int(LORA_ALPHA))
LORA_DROPOUT = st.sidebar.number_input("LORA_DROPOUT", min_value=0, value=float(LORA_DROPOUT))
MAX_SEQ_LEN = st.sidebar.number_input("MAX_SEQ_LEN", value=int(MAX_SEQ_LEN))
BATCH_SIZE = st.sidebar.number_input("Chunk Size", min_value=8, value=int(BATCH_SIZE))
EPOCHS = st.sidebar.number_input("Chunk Size", min_value=0, value=int(EPOCHS))
LR = st.sidebar.number_input("Chunk Size", value=int(LR))


st.header("1. Ingest Documents")
if st.session_state.session_active:
    uploaded = st.file_uploader("Upload files (CSV/Excel/TXT)", type=["csv","xlsx","txt","pdf"], accept_multiple_files=True)
    if st.button("Run Ingest") and uploaded:
        files = [(f.name, f) for f in uploaded]
        multipart = [("files", (f.name, f.getvalue())) for f in uploaded]
        res = requests.post(backend_url('/ingest'), files=multipart)
        if res.ok:
            st.success("Ingestion complete")
        else:
            st.error("Ingestion failed")
else:
    st.info("Start a session to ingest files.")

st.markdown("---")

st.header("2. Ask a Question")
if st.session_state.session_active:
    question = st.text_input("Enter your query:")
    fallback = st.checkbox("Enable LLM Fallback")
    think = st.checkbox("Enable Chain-of-Thought (Think)")
    show_prints = st.checkbox("Show internal print logs", value=False)
    if st.button("Run Query") and question:
        payload = {"query": question, "fallback": fallback, "think": think}
        res_1, res_2 = requests.post(backend_url('/query'), json=payload)
        if res_1.ok:

            st.subheader("Answer")
            st.success(res_1)
            
            if show_prints:
                st.subheader("Internal Logs")
                st.text(res_2)
        else:
            st.error("Query failed")
else:
    st.info("Start a session to run queries.")
