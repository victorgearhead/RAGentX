import re
from typing import List
from llm_utils import get_llm
import re
from typing import List
from retrieval import query_out
from kg_pipeline import extract_facts_from_text, store_facts_to_neo4j
from langchain import LLMChain, PromptTemplate
from llm_utils import get_llm

def tool_llm(prompt: str) -> str:
    llm = get_llm()
    chain = LLMChain(llm=llm, prompt=None)
    res = chain.run(prompt=prompt)
    return res.stdout.strip()

def react_agent(question:str, fallback_bool:bool, think:bool) -> str:
    if think:
    
        tools = {
            "Retrieve": lambda q: query_out(q),
            "Distill": lambda q: tool_distill(q),
            "Critique": lambda q: tool_self_critique(q),
        }

        thoughts = []
        final_answer = ""

        for step in range(3):
            context = "\n".join([f"Thought {i+1}: {t}" for i, t in enumerate(thoughts)])
            planning_prompt = f"""
                You are an intelligent reasoning agent with access to tools: Retrieve, Critique, Distill.
                Your job is to answer complex enterprise questions using external tools.
                You are supposed to implement a multi-step reasoning process in this order:
                1. Retrieve relevant information from the knowledge base.
                2. Distill the retrieved information into a concise summary.
                3. Critique the summary against the original question and retrieved evidence.
                Question: {question}
                {context}

                What should you do next?
                Respond in this format:
                Thought: ...
                Action: {{ToolName}}: {{Input}}
                """
            response = tool_llm(planning_prompt)
            match = re.search(r"Action:\s*(\w+):\s*(.*)", response)
            if not match:
                break

            tool_name, argument = match.groups()
            tool = tools.get(tool_name)
            if not tool:
                break

            tool_output = tool(argument.strip())
            thoughts.append(f"Used {tool_name} with input '{argument.strip()}'. Result: {tool_output[0][:300]}...., {tool_output[1]}")

        if not final_answer:
            final_answer_1, final_answer_2 = query_out(question, fallback_bool)
            return final_answer_1, final_answer_2

        facts = extract_facts_from_text(final_answer)
        if facts:
            store_facts_to_neo4j(facts)

        return final_answer, thoughts
    
    final_answer_1, final_answer_2 = query_out(question, fallback_bool)
    return final_answer_1, final_answer_2

def tool_distill(chunks: List[str], llm):
    snippet = "\n---\n".join(chunks)
    prompt = f"Distill the following snippets into a concise, factual summary without hallucinations:\n{snippet}"
    llm = get_llm()
    chain = LLMChain(llm=llm, prompt=None)
    res = chain.run(prompt=prompt)
    return [res.stdout.strip(), None]

def tool_self_critique(answer: str, evidence: List[str], llm):
    evid = "\n".join(evidence)
    prompt = (
        f"Given the answer:\n{answer}\nand the evidence:\n{evid}\n"  
        "Critique the answer: is it fully supported by the evidence? "
        "If unsupported or hallucinated, point out inaccuracies."
    )
    llm = get_llm()
    chain = LLMChain(llm=llm, prompt=None)
    res = chain.run(prompt=prompt)
    return [res.stdout.strip(), None]