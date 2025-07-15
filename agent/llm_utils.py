from langchain.llms import Ollama, OpenAI, HuggingFaceHub
from config import MODEL_PROVIDER, LLM_MODEL, MODEL_KEY

def get_llm():
    provider = MODEL_PROVIDER.lower()
    if provider == "ollama":
        return Ollama(model=LLM_MODEL)
    if provider == "openai":
        return OpenAI(model_name=LLM_MODEL, openai_api_key=MODEL_KEY)
    if provider in ("huggingface", "hf"):
        return HuggingFaceHub(repo_id=LLM_MODEL, huggingfacehub_api_token=MODEL_KEY)
    raise ValueError(f"Edit llm_utils.py manually for more Model Providers: {MODEL_PROVIDER}")
