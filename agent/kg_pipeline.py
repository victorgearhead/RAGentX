import os
from typing import List, Dict
import os
import torch
import nltk
import spacy
from neo4j import GraphDatabase

nltk.download("punkt")
nlp = spacy.load("en_core_web_sm")
device = "cuda" if torch.cuda.is_available() else "cpu"
from config import NEO4J_PASS, NEO4J_USER, NEO4J_URI

def build_kg(nodes):
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))
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

def extract_facts_from_text(text: str):
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

def query_kg_multi_hop(question: str, uri=os.getenv("NEO4J_URI"), user=os.getenv("NEO4J_USER"), password=os.getenv("NEO4J_PASS")):
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