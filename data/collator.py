# data/collator.py

import torch

class DataCollatorForSeq:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        max_len = max(len(x["input_ids"]) for x in batch)

        input_ids = []
        labels = []
        attn = []

        for x in batch:
            pad_len = max_len - len(x["input_ids"])
            input_ids.append(x["input_ids"] + [self.tokenizer.pad_token_id] * pad_len)
            labels.append(x["labels"] + [-100] * pad_len)
            attn.append(x["attention_mask"] + [0] * pad_len)

        return {
            "input_ids": torch.tensor(input_ids),
            "labels": torch.tensor(labels),
            "attention_mask": torch.tensor(attn),
        }

class DataCollatorForGLUE:
    def __init__(self, tokenizer, pad_to_multiple_of=8):
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id or 0
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, batch):
        max_len = max(len(x["input_ids"]) for x in batch)
        if self.pad_to_multiple_of:
            if max_len % self.pad_to_multiple_of != 0:
                max_len = ((max_len // self.pad_to_multiple_of) + 1) * self.pad_to_multiple_of

        input_ids = []
        attn = []
        labels = []
        for x in batch:
            pad_len = max_len - len(x["input_ids"])
            input_ids.append(x["input_ids"] + [self.pad_token_id] * pad_len)
            attn.append(x.get("attention_mask", [1]*len(x["input_ids"])) + [0] * pad_len)
            labels.append(x.get("label", -1))

        return {"input_ids": torch.tensor(input_ids, dtype=torch.long),
                "attention_mask": torch.tensor(attn, dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long)}