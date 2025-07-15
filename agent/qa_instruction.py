import re
import ast
import subprocess
from typing import List, Tuple
from config import LLM_MODEL, NUM_PAIRS
from langchain import LLMChain, PromptTemplate
from llm_utils import get_llm

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


def generate_qa_for_chunk(text_chunk: str, model_name=LLM_MODEL, num_pairs=NUM_PAIRS) -> List[Tuple[str,str]]:
    prompt = f"Generate {num_pairs} question-answer pairs in valid JSON format as an array of objects, each with keys \"question\" and \"answer\", from the following text: {text_chunk[:2000]}"
    print(f"Generating {num_pairs} QA pairs for chunk of size {len(text_chunk)}")
    llm = get_llm()
    chain = LLMChain(llm=llm, prompt=None)
    res = chain.run(prompt=prompt)
    print("LLM Output Successful")
    return extract_pairs(res.stdout)