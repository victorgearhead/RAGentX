import re
import json
import subprocess
from typing import List
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForQuestionAnswering,
    TrainingArguments,
    Trainer,
    default_data_collator
)
from peft import get_peft_model, LoraConfig, TaskType
from config import PEFT_DIR, QA_PATH,  LLM_MODEL, LM_MODEL, LORA_R, LORA_ALPHA, LORA_DROPOUT, MAX_SEQ_LEN, BATCH_SIZE, EPOCHS, LR
from langchain import LLMChain, PromptTemplate
from llm_utils import get_llm

device = "cuda" if torch.cuda.is_available() else "cpu"

def finetune_lm():
    with open(QA_PATH, 'r', encoding='utf-8') as f:
        qa_data = json.load(f)

    tokenizer = AutoTokenizer.from_pretrained(LM_MODEL)
    model = AutoModelForQuestionAnswering.from_pretrained(LM_MODEL)

    peft_conf = LoraConfig(
        task_type=None,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT
    )
    model = get_peft_model(model, peft_conf).to(device)

    features = []
    for ex in qa_data:
        inputs = tokenizer(
            ex["question"],
            ex["context"],
            truncation="only_second",
            max_length=MAX_SEQ_LEN,
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
        learning_rate=LR,                
        per_device_train_batch_size=BATCH_SIZE, 
        num_train_epochs=EPOCHS,        
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
    print(f"Fine‑tuned adapters saved at {PEFT_DIR}")

def rerank_chunks_with_llm(query: str, chunks: List[str], model=LLM_MODEL) -> List[str]:
    prompt = f"""You are a helpful assistant. User asked: {query} Here are some document snippets:{chr(10).join(f'{i+1}. {chunk}' for i, chunk in enumerate(chunks))} Rank the top 2 most relevant chunks to the query and return as JSON: [{{"rank": 1, "text": "..."}}, {{"rank": 2, "text": "..."}}]"""
    llm = get_llm()
    chain = LLMChain(llm=llm, prompt=None)
    res = chain.run(prompt=prompt)

    try:
        data = json.loads(re.search(r'(\[.*?\])', res.stdout, re.DOTALL).group(1))
        return [item['text'] for item in data if 'text' in item]
    except Exception as e:
        print("[Rerank LLM] Failed to parse rerank result:", e)
        return chunks[:2]