from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from ingestion import ingest_in
from retrieval import query_out
from react_flow import react_agent
import os
import shutil
from config import SESSION_DIR

class SessionCache:
    def __init__(self):
        os.makedirs(SESSION_DIR, exist_ok=True)
        self.dir = SESSION_DIR
        print(f"Session cache cleaned up at {self.dir}")

    def cleanup(self):
        for fname in os.listdir(self.dir):
            path = os.path.join(self.dir, fname)
            if os.path.isfile(path):
                os.remove(path)
            else:
                shutil.rmtree(path, ignore_errors=True)

app = FastAPI()

session_cache: SessionCache = None

@app.post("/session/start")
def start_session():
    global session_cache
    if session_cache is None:
        session_cache = SessionCache()
        return {"status": "session started", "session_dir": session_cache.dir}
    return {"status": "session already active", "session_dir": session_cache.dir}

@app.post("/session/end")
def end_session():
    global session_cache
    if session_cache is None:
        raise HTTPException(400, "No active session")
    session_cache.cleanup()
    session_cache = None
    return {"status": "session ended, cache cleared"}

@app.post('/ingest')
async def ingest(files: list[UploadFile] = File(...)):
    if session_cache is None:
        raise HTTPException(400, "Start a session first")
    
    for file in files:
        dest = os.path.join(session_cache.dir, file.filename)
        with open(dest, "wb") as f:
            f.write(await file.read())
    ingest_in(data_dir=session_cache.dir)
    return {"status": "ingested"}

class QueryRequest(BaseModel):
    query: str
    fallback: bool = False
    think: bool = False

@app.post('/query')
def query(req: QueryRequest):
    if session_cache is None:
        raise HTTPException(400, "Start a session first")
    answer_1, answer_2 = react_agent(req.query, req.fallback, req.think,
        session_dir=session_cache.dir)
    return answer_1, answer_2