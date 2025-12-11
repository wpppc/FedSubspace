# data/partition_tasks.py
import os, json, random
import numpy as np
from collections import defaultdict
from datasets import load_dataset

def save_json(out_path, data):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)

def partition_list(items, num_clients, strategy="dirichlet", alpha=0.5, seed=42):
    random.seed(seed); np.random.seed(seed)
    N = len(items)
    if strategy == "dirichlet":
        # simple label-agnostic dirichlet on indices
        proportions = np.random.dirichlet([alpha]*num_clients)
        counts = (proportions * N).astype(int)
        while counts.sum() < N:
            counts[np.random.randint(0, num_clients)] += 1
        idxs = np.arange(N)
        np.random.shuffle(idxs)
        splits = np.split(idxs, np.cumsum(counts)[:-1])
        return [ [items[int(i)] for i in s] for s in splits ]
    else:
        # shard-based
        num_shards = num_clients * 2
        shards = np.array_split(np.arange(N), num_shards)
        random.shuffle(shards)
        shards_per_client = len(shards)//num_clients
        parts=[]
        ptr=0
        for i in range(num_clients):
            sel = []
            for j in range(shards_per_client):
                if ptr < len(shards):
                    sel.extend(shards[ptr].tolist())
                    ptr += 1
            parts.append([items[idx] for idx in sel])
        return parts

def partition_task(task_cfg, out_root, num_clients, strategy="dirichlet", alpha=0.5, seed=42):
    """
    task_cfg structure:
    {
      "name": "metamathqa",
      "hf_dataset": "namespace/dataset" or local path,
      "hf_config": optional,
      "train_split": "train",
      "format_fn": function(item)->{"input":..., "output":..., ...},
      "eval_sets": { "gsm8k": ("gsm8k","test"), ... }
    }
    """
    path = task_cfg["hf_dataset"]
    config = task_cfg.get("hf_config", None)
    
    # Smart loading for local files
    if os.path.exists(path):
        if os.path.isdir(path):
            # Check for specific file types in directory
            files = os.listdir(path)
            json_files = [f for f in files if f.endswith(".json") and f != "dataset_infos.json"]
            jsonl_files = [f for f in files if f.endswith(".jsonl")]
            parquet_files = [f for f in files if f.endswith(".parquet")]

            if json_files:
                ds = load_dataset("json", data_dir=path, split=task_cfg.get("train_split","train"))
            elif jsonl_files:
                ds = load_dataset("json", data_dir=path, split=task_cfg.get("train_split","train"))
            elif parquet_files:
                ds = load_dataset("parquet", data_dir=path, split=task_cfg.get("train_split","train"))
            else:
                # Fallback to default loading (e.g. if it's a HF repo clone with script)
                ds = load_dataset(path, config, split=task_cfg.get("train_split","train"))
        else:
             # It's a file
            ext = path.split(".")[-1]
            ds = load_dataset(ext, data_files=path, split=task_cfg.get("train_split","train"))
    else:
        # Remote HF dataset
        ds = load_dataset(path, config, split=task_cfg.get("train_split","train"))

    train_list = [task_cfg["format_fn"](item) for item in ds]
    parts = partition_list(train_list, num_clients, strategy=strategy, alpha=alpha, seed=seed)

    task_dir = os.path.join(out_root, task_cfg["name"])
    os.makedirs(task_dir, exist_ok=True)
    for cid, part in enumerate(parts):
        save_json(os.path.join(task_dir, f"local_training_{cid}.json"), part)

    # save eval sets
    for eval_name, val in task_cfg.get("eval_sets", {}).items():
        if len(val) == 3:
            ds_name, split, config_name = val
        else:
            ds_name, split = val
            config_name = None

        # Similar smart loading for eval sets
        if os.path.exists(ds_name):
             if os.path.isdir(ds_name):
                files = os.listdir(ds_name)
                json_files = [f for f in files if f.endswith(".json") and f != "dataset_infos.json"]
                jsonl_files = [f for f in files if f.endswith(".jsonl")]
                parquet_files = [f for f in files if f.endswith(".parquet")]
                
                if json_files:
                    eds = load_dataset("json", data_dir=ds_name, split=split)
                elif jsonl_files:
                    eds = load_dataset("json", data_dir=ds_name, split=split)
                elif parquet_files:
                    eds = load_dataset("parquet", data_dir=ds_name, split=split)
                else:
                    py_files = [f for f in files if f.endswith(".py")]
                    if py_files:
                        eds = load_dataset(os.path.join(ds_name, py_files[0]), config_name, split=split, trust_remote_code=True)
                    else:
                        eds = load_dataset(ds_name, config_name, split=split)
             else:
                ext = ds_name.split(".")[-1]
                eds = load_dataset(ext, data_files=ds_name, split=split)
        else:
            eds = load_dataset(ds_name, config_name, split=split)
            
        if task_cfg.get("format_eval", True):
            eval_list = [task_cfg["format_fn"](item) for item in eds]
        else:
            # Save raw items
            eval_list = [item for item in eds]
            
        save_json(os.path.join(task_dir, f"global_eval_{eval_name}.json"), eval_list)

    print(f"[Partitioned {task_cfg['name']}] saved to {task_dir}")
    return task_dir

# Example usage (call from CLI or script)
if __name__ == "__main__":
    # small example for metamathqa (you'll replace with real task_cfg dict)
    pass
