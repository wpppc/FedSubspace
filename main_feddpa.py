import os
# [防死锁] 必须在 torch 之前
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys
import yaml
import torch
import json
import csv
import time
import gc
import re
import math
from tqdm import tqdm
from collections import Counter

from transformers import AutoTokenizer

# Custom modules
from data.dataset_tasks import GenericGenDataset, GenCollator
from models.llama_wrapper_dpa import FedDPAModelWrapper
from federated.client_dpa import FedDPAClient

# --- [关键修复] 禁用 SDPA 防止 RecursionError ---
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_math_sdp(True)

CLIENT_TASK_MAPPING = {
    0: "Sentiment Analysis", 1: "Natural Language Inference", 2: "Text Classification",
    3: "Commonsense Reasoning", 4: "Paraphrase Detection", 5: "Struct to Text",
    6: "Reading Comprehension", 7: "Coreference Resolution"
}

def compute_rouge1(prediction, reference):
    pred_tokens = re.findall(r'\w+', prediction.lower())
    ref_tokens = re.findall(r'\w+', reference.lower())
    if not pred_tokens or not ref_tokens: return 0.0
    pred_counts = Counter(pred_tokens)
    ref_counts = Counter(ref_tokens)
    overlap = sum((pred_counts & ref_counts).values())
    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)
    if precision + recall == 0: return 0.0
    return 2 * (precision * recall) / (precision + recall)

def update_summary_csv(output_dir, round_num, client_scores):
    summary_file = os.path.join(output_dir, "summary_table.csv")
    columns = ["Method", "Round"] + sorted(list(set(CLIENT_TASK_MAPPING.values()))) + ["Average"]
    file_exists = os.path.exists(summary_file)
    row_data = {"Method": "FedDPA", "Round": round_num, "Average": 0.0}
    
    total = 0
    for cid, score in client_scores.items():
        task = CLIENT_TASK_MAPPING.get(cid, f"Client {cid}")
        row_data[task] = f"{score * 100:.2f}"
        total += score
    if len(client_scores) > 0:
        row_data["Average"] = f"{(total / len(client_scores)) * 100:.2f}"
    
    with open(summary_file, mode='a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        if not file_exists: writer.writeheader()
        writer.writerow(row_data)

def get_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        num_params = param.numel()
        if num_params == 0 and hasattr(param, "ds_numel"): num_params = param.ds_numel
        if param.__class__.__name__ == "Params4bit": num_params = num_params * 2
        all_param += num_params
        if param.requires_grad: trainable_params += num_params
    return trainable_params, all_param

def main():
    # 1. Config
    cfg_path = "configs/fedsubspace_flan.yaml"
    cfg = yaml.safe_load(open(cfg_path, "r"))
    cfg["output_dir"] = "outputs/fed_dpa" # Distinct output dir
    os.makedirs(cfg["output_dir"], exist_ok=True)
    
    # [关键对齐] 
    # Batch Size = 1 (to save memory for Dual Adapters)
    # Accumulation = 16 (Total Batch = 16, Same as FedLESS)
    batch_size = 1
    grad_accum = 16
    lr = 2e-4 # Same as FedLESS
    
    print(f">> [Config] Batch: {batch_size}, Accum: {grad_accum}, LR: {lr}")

    # 2. Model & Tokenizer
    torch_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["path"])
    tokenizer.pad_token_id = tokenizer.eos_token_id or 0
    
    # Shared Model (Wrapper)
    shared_model = FedDPAModelWrapper(
        cfg["model"]["path"],
        lora_r=cfg["lora"]["r"],
        target_modules=cfg["lora"]["target_modules"],
        torch_dtype=torch_dtype
    )
    
    # Calculate Params
    trainable_params, all_param = get_trainable_parameters(shared_model)
    subspace_params = 0 # FedDPA doesn't use subspace
    full_lora_params = trainable_params # Approx
    
    # 3. Server State (Global Adapter Only)
    server_global_state = shared_model.get_global_state_dict()
    
    # 4. Data
    client_dataloaders = []
    for cid in range(cfg["data"]["num_clients"]):
        data_path = os.path.join(cfg["data"]["root"], f"client_{cid}.json")
        if not os.path.exists(data_path):
            client_dataloaders.append(None); continue
        ds = GenericGenDataset(data_path)
        collator = GenCollator(tokenizer, max_len=cfg["data"]["cutoff_len"], train_on_inputs=False)
        dl = DataLoader(ds, batch_size=batch_size, shuffle=True, collate_fn=lambda b: collator(b), num_workers=0)
        client_dataloaders.append((dl, collator))

    # 5. Resume Logic
    start_round = 0
    checkpoint_pattern = re.compile(r"global_round(\d+).pt")
    checkpoints = [int(m.group(1)) for f in os.listdir(cfg["output_dir"]) if (m := checkpoint_pattern.match(f))]
    
    if checkpoints:
        last_round = max(checkpoints)
        ckpt_path = os.path.join(cfg["output_dir"], f"global_round{last_round}.pt")
        print(f">> [Resume] Loading checkpoint: {ckpt_path}")
        server_global_state = torch.load(ckpt_path, map_location="cpu")
        start_round = last_round + 1

    # 6. Training Loop
    rounds = cfg["federated"]["rounds"]
    # [关键对齐] Warmup Rounds = 1 (Same as FedLESS)
    warmup_rounds = 1 
    
    # CSV Init
    csv_path = os.path.join(cfg["output_dir"], "experiment_results.csv")
    if not os.path.exists(csv_path):
        with open(csv_path, 'w') as f:
            csv.writer(f).writerow(["Timestamp", "BaseModel", "Dataset", "Round", "Metric", "Value", "SubspaceParams", "FullLoRAParams", "TrainableParams", "AllParams", "TrainableRatio"])

    for r in range(start_round, rounds):
        print(f"\n=== Round {r} (FedDPA) ===")
        
        # LR Schedule
        if r < warmup_rounds:
            current_lr = lr * (r + 1) / warmup_rounds
        else:
            progress = (r - warmup_rounds) / (rounds - warmup_rounds)
            current_lr = 1e-6 + 0.5 * (lr - 1e-6) * (1 + math.cos(progress * math.pi))
        
        print(f"Current LR: {current_lr:.2e}")
        
        updates = []
        sizes = []
        
        # Train Clients
        for cid, dl_info in enumerate(client_dataloaders):
            if not dl_info: continue
            dl, collator = dl_info
            
            client = FedDPAClient(
                client_id=cid, model=shared_model, tokenizer=tokenizer, 
                dataloader=dl, output_dir=cfg["output_dir"],
                local_epochs=cfg["train"]["local_epochs"], lr=current_lr,
                dtype=torch_dtype, gradient_accumulation_steps=grad_accum
            )
            
            # Load State (Global + Local)
            client.load_state(server_global_state)
            
            # Train
            client.train()
            
            # Collect Global Update
            updates.append(client.get_update_for_server())
            sizes.append(len(dl.dataset))
            
            # Cleanup
            client.model = None; del client; gc.collect(); torch.cuda.empty_cache()
        
        # Aggregate
        print("Aggregating...")
        total_samples = sum(sizes)
        agg_state = {k: torch.zeros_like(v) for k, v in updates[0].items()}
        for update, size in zip(updates, sizes):
            weight = size / total_samples
            for k in agg_state:
                agg_state[k] += update[k] * weight
        
        server_global_state = agg_state
        torch.save(server_global_state, os.path.join(cfg["output_dir"], f"global_round{r}.pt"))
        
        # Evaluation
        if r % cfg["eval"].get("eval_every", 1) == 0:
            print("Evaluating...")
            client_scores = {}
            
            for cid in range(cfg["data"]["num_clients"]):
                test_path = os.path.join(cfg["data"]["root"], f"test_{cid}.json")
                if not os.path.exists(test_path): continue
                
                # Load correct state for eval (Global + Persistent Local)
                shared_model.load_global_state_dict(server_global_state)
                local_path = os.path.join(cfg["output_dir"], f"client_{cid}_local_state.pt")
                if os.path.exists(local_path):
                    shared_model.load_local_state_dict(torch.load(local_path))
                
                shared_model.eval()
                
                # Inference
                with open(test_path, "r") as f: data = json.load(f)[:100] # Limit 100
                
                scores = []
                for ex in tqdm(data, desc=f"Client {cid}", leave=False):
                    prompt = f"### Instruction:\n{ex['input']}\n\n### Response:\n"
                    inputs = tokenizer(prompt, return_tensors="pt").to(shared_model.base_model.device)
                    with torch.no_grad():
                        out = shared_model.base_model.generate(**inputs, max_new_tokens=50, pad_token_id=tokenizer.eos_token_id)
                    pred = tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).split("###")[0].strip()
                    scores.append(compute_rouge1(pred, ex["output"]))
                
                avg = sum(scores)/len(scores) if scores else 0
                client_scores[cid] = avg
                t_name = CLIENT_TASK_MAPPING.get(cid, f"Client {cid}")
                print(f"  {t_name}: {avg:.4f}")
                
                with open(csv_path, 'a') as f:
                    csv.writer(f).writerow([
                        time.strftime("%Y-%m-%d %H:%M:%S"), 
                        os.path.basename(cfg["model"]["path"]), 
                        f"FedDPA ({t_name})", 
                        r, 
                        "ROUGE-1", 
                        avg,
                        subspace_params,
                        full_lora_params,
                        trainable_params,
                        all_param,
                        f"{100 * trainable_params / all_param:.4f}%"
                    ])
            
            update_summary_csv(cfg["output_dir"], r, client_scores)

if __name__ == "__main__":
    main()