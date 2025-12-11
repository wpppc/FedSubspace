# data/dataset.py

from torch.utils.data import Dataset
import json

class InstructionDataset(Dataset):
    def __init__(self, path, tokenizer, cutoff_len=512, train_on_inputs=True):
        self.data = [json.loads(l) for l in open(path)]
        self.tokenizer = tokenizer
        self.cutoff_len = cutoff_len
        self.train_on_inputs = train_on_inputs

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        item = self.data[i]

        instruction = item["instruction"]
        context     = item.get("context", "")
        response    = item["response"]

        if context:
            prompt = f"Instruction: {instruction}\nContext: {context}\nAnswer:"
        else:
            prompt = f"Instruction: {instruction}\nAnswer:"

        target = response

        prompt_ids = self.tokenizer.encode(
            prompt,
            max_length=self.cutoff_len,
            truncation=True
        )
        target_ids = self.tokenizer.encode(
            target,
            max_length=self.cutoff_len,
            truncation=True
        )

        input_ids = prompt_ids + target_ids + [self.tokenizer.eos_token_id]
        labels = input_ids.copy()

        if not self.train_on_inputs:
            labels[:len(prompt_ids)] = [-100] * len(prompt_ids)

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": [1] * len(input_ids)
        }

class LocalJSONDataset(Dataset):
    """
    Load already-partitioned client json file where each example is a dict.
    The example dict shape depends on the task. This loader just provides raw fields;
    tokenization is done in collator or in model wrapper.
    """
    def __init__(self, path):
        with open(path, "r", encoding="utf-8") as f:
            self.data = json.load(f)
    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]


class GlueClientDataset(Dataset):
    """
    Lightweight wrapper for a list of glue-style examples (dicts) and a tokenizer.
    Will produce input_ids, attention_mask, labels based on glue task specifics
    """
    def __init__(self, examples, tokenizer, task="sst2", cutoff_len=128, train_on_inputs=False):
        self.examples = examples
        self.tokenizer = tokenizer
        self.task = task
        self.cutoff_len = cutoff_len
        self.train_on_inputs = train_on_inputs

    def __len__(self): return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        # mapping for common GLUE fields -- try generic patterns
        if "sentence" in ex:
            text = ex["sentence"]
        elif "sentence1" in ex and "sentence2" in ex:
            text = ex["sentence1"] + " " + ex["sentence2"]
        elif "question" in ex:
            text = ex.get("question", "") + " " + ex.get("sentence", "")
        else:
            # fallback: attempt to stringify
            text = " ".join([str(v) for v in ex.values() if isinstance(v, str)])[:self.cutoff_len]

        label = ex.get("label", -1)
        enc = self.tokenizer(text, truncation=True, max_length=self.cutoff_len, padding=False)
        return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"], "label": label}