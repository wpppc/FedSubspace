# data/dataset_tasks.py
import json
import os
from torch.utils.data import Dataset
import torch
import pandas as pd

class GenericGenDataset(Dataset):
    def __init__(self, path):
        self.data = []
        if path.endswith(".json"):
            try:
                self.data = json.load(open(path, "r", encoding="utf-8"))
            except json.JSONDecodeError:
                # Try reading as jsonl if json load fails
                with open(path, "r", encoding="utf-8") as f:
                    self.data = [json.loads(line) for line in f]
        elif path.endswith(".jsonl"):
            with open(path, "r", encoding="utf-8") as f:
                self.data = [json.loads(line) for line in f]
        elif path.endswith(".parquet"):
            df = pd.read_parquet(path)
            self.data = df.to_dict(orient="records")
        else:
            raise ValueError(f"Unsupported file format: {path}")

    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        ex = self.data[idx]
        # Support multiple key variations
        inp = ex.get("input", ex.get("query", ex.get("instruction", "")))
        out = ex.get("output", ex.get("response", ex.get("answer", "")))
        return {"input": inp, "output": out}

class GenCollator:
    def __init__(self, tokenizer, max_len=512, train_on_inputs=False):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.train_on_inputs = train_on_inputs

    def __call__(self, batch):
        if len(batch) > 0 and "input" not in batch[0]:
            print(f"DEBUG: batch[0] keys: {batch[0].keys()}")
            print(f"DEBUG: batch[0]: {batch[0]}")
        inputs = [b["input"] for b in batch]
        outputs = [b["output"] for b in batch]
        
        # Updated Prompt Template as per user request
        # Template: Alpaca Style
        # Append EOS token to ensure model learns to stop
        full_prompts = [f"### Instruction:\n{inp}\n\n### Response:\n{out}{self.tokenizer.eos_token}" for inp, out in zip(inputs, outputs)]
        
        enc = self.tokenizer(full_prompts, padding=True, truncation=True, max_length=self.max_len, return_tensors="pt")
        labels = enc["input_ids"].clone()
        if not self.train_on_inputs:
            # mask prompt portion
            # We need to encode the prompt part to know its length
            prompts_only = [f"### Instruction:\n{inp}\n\n### Response:\n" for inp in inputs]
            prompt_enc = self.tokenizer(prompts_only, padding=True, truncation=True, max_length=self.max_len, return_tensors="pt")
            
            mask = torch.ones_like(labels) * -100
            for i in range(len(inputs)):
                # Calculate length of the prompt part by finding the common prefix
                # This handles tokenizer weirdness with spaces at the boundary
                p_ids = prompt_enc["input_ids"][i]
                f_ids = enc["input_ids"][i]
                
                # Find the first index where they differ
                # We only care about the valid tokens (not padding)
                p_len_raw = (p_ids != self.tokenizer.pad_token_id).sum().item()
                f_len_raw = (f_ids != self.tokenizer.pad_token_id).sum().item()
                
                # Truncate to min length to compare
                min_len = min(p_len_raw, f_len_raw)
                
                # Find divergence point
                divergence_idx = min_len
                for idx in range(min_len):
                    if p_ids[idx] != f_ids[idx]:
                        divergence_idx = idx
                        break
                
                plen = divergence_idx
                
                # Safety check: if prompt takes up whole sequence
                if plen >= f_len_raw:
                    # If prompt takes up whole sequence, mask everything
                    mask[i, :] = -100
                else:
                    # Mask everything up to plen
                    # We want to train on the output, which starts after the prompt
                    mask[i, plen:f_len_raw] = f_ids[plen:f_len_raw]
                    
            labels = mask
        batch = {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"], "labels": labels}
        return batch
