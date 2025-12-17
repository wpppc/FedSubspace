import os
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
import copy
from tqdm import tqdm
from collections import Counter
from transformers import AutoTokenizer

from data.dataset_tasks import GenericGenDataset, GenCollator
from models.llama_wrapper_fedtt import FedTTModelWrapper
from federated.client_fedtt import FedTTClient

# Disable SDPA for stability
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

def update_summary_csv(output_dir, method_name, round_num, client_scores):
    summary_file = os.path.join(output_dir, "summary_table.csv")
    task_columns = sorted(list(set(CLIENT_TASK_MAPPING.values())))
    columns = ["Method", "Round"] + task_columns + ["Average"]
    file_exists = os.path.exists(summary_file)
    
    row_data = {"Method": method_name, "Round": round_num, "Average": 0.0}
    
    total_score = 0
    count = 0
    for cid, score in client_scores.items():
        task_name = CLIENT_TASK_MAPPING.get(cid, f"Client {cid}")
        row_data[task_name] = f"{score * 100:.2f}"
        total_score += score
        count += 1
        
    if count > 0:
        row_data["Average"] = f"{(total_score / count) * 100:.2f}"
    
    with open(summary_file, mode='a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        if not file_exists: writer.writeheader()
        writer.writerow(row_data)
    print(f">> [Summary] Updated {summary_file}")

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

def server_aggregate(updates, sizes):
    total = sum(sizes)
    agg_state = copy.deepcopy(updates[0])
    for k in agg_state.keys():
        agg_state[k] = torch.zeros_like(agg_state[k], dtype=torch.float32)
        
    for update, size in zip(updates, sizes):
        w = size / total
        for k in update.keys():
            agg_state[k] += update[k].to(torch.float32) * w
    return agg_state

def main():
    cfg_path = "configs/fedsubspace_flan.yaml"
    cfg = yaml.safe_load(open(cfg_path, "r"))
    cfg["output_dir"] = "outputs/fed_tt_acl2025"
    os.makedirs(cfg["output_dir"], exist_ok=True)
    
    # Align Hyperparams
    batch_size = 1
    grad_accum = 16 # Effective Batch 16
    lr = 2e-4
    
    print(f">> [FedTT] ACL 2025 Config | Batch: {batch_size} | Accum: {grad_accum} | LR: {lr}")

    # Model
    torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model_wrapper = FedTTModelWrapper(
        cfg["model"]["path"],
        bottleneck_dim=64,
        tt_rank=8, # Paper suggested 5 or 8. 8 is stronger baseline.
        target_modules=cfg["lora"]["target_modules"],
        torch_dtype=torch_dtype
    )
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["path"])
    tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Calculate Params
    trainable_params, all_param = get_trainable_parameters(model_wrapper)
    subspace_params = 0
    full_lora_params = trainable_params # Approx

    # Server State
    global_state = model_wrapper.get_trainable_state_dict()
    
    # Data
    client_dataloaders = []
    for cid in range(cfg["data"]["num_clients"]):
        data_path = os.path.join(cfg["data"]["root"], f"client_{cid}.json")
        if not os.path.exists(data_path):
            client_dataloaders.append(None); continue
        ds = GenericGenDataset(data_path)
        col = GenCollator(tokenizer, max_len=cfg["data"]["cutoff_len"], train_on_inputs=False)
        dl = DataLoader(ds, batch_size=batch_size, shuffle=True, collate_fn=lambda b: col(b), num_workers=0)
        client_dataloaders.append((dl, col))

    # Resume
    start_r = 0
    cp_pat = re.compile(r"global_round(\d+).pt")
    cps = [int(m.group(1)) for f in os.listdir(cfg["output_dir"]) if (m := cp_pat.match(f))]
    if cps:
        start_r = max(cps) + 1
        global_state = torch.load(os.path.join(cfg["output_dir"], f"global_round{max(cps)}.pt"), map_location="cpu")
        print(f">> Resuming from Round {start_r}")

    # CSV
    csv_p = os.path.join(cfg["output_dir"], "experiment_results.csv")
    if not os.path.exists(csv_p):
        with open(csv_p, 'w') as f: 
            csv.writer(f).writerow(["Timestamp", "BaseModel", "Dataset", "Round", "Metric", "Value", "SubspaceParams", "FullLoRAParams", "TrainableParams", "AllParams", "TrainableRatio"])

    rounds = cfg["federated"]["rounds"]
    # [关键对齐] Warmup Rounds = 1 (Same as FedLESS)
    warmup = 1

    for r in range(start_r, rounds):
        print(f"\n=== FedTT Round {r} ===")
        
        # LR Schedule
        if r < warmup: cur_lr = lr * (r+1)/warmup
        else: cur_lr = 1e-6 + 0.5*(lr-1e-6)*(1+math.cos((r-warmup)/(rounds-warmup)*math.pi))
        
        updates = []
        sizes = []
        
        # Train
        model_wrapper.load_trainable_state_dict(global_state)
        for cid, dl_info in enumerate(client_dataloaders):
            if not dl_info: continue
            
            client = FedTTClient(
                cid, model_wrapper, dl_info[0], cfg["output_dir"],
                local_epochs=cfg["train"]["local_epochs"], lr=cur_lr,
                dtype=torch_dtype, gradient_accumulation_steps=grad_accum
            )
            client.train()
            updates.append(client.get_update_for_server())
            sizes.append(len(dl_info[0].dataset))
            del client; torch.cuda.empty_cache()
            
        # Aggregate
        global_state = server_aggregate(updates, sizes)
        torch.save(global_state, os.path.join(cfg["output_dir"], f"global_round{r}.pt"))
        
        # Eval
        if cfg["eval"]["enabled"] and (r % cfg["eval"].get("eval_every", 1) == 0):
            print("Evaluating...")
            model_wrapper.load_trainable_state_dict(global_state)
            model_wrapper.eval()
            
            for cid in range(cfg["data"]["num_clients"]):
                t_path = os.path.join(cfg["data"]["root"], f"test_{cid}.json")
                if not os.path.exists(t_path): continue
                with open(t_path) as f: data = json.load(f)[:100]
                
                scores = []
                for ex in tqdm(data, desc=f"Eval C{cid}", leave=False):
                    prompt = f"### Instruction:\n{ex['input']}\n\n### Response:\n"
                    inps = tokenizer(prompt, return_tensors="pt").to(model_wrapper.base_model.device)
                    with torch.no_grad():
                        out = model_wrapper.base_model.generate(**inps, max_new_tokens=50, pad_token_id=tokenizer.eos_token_id)
                    pred = tokenizer.decode(out[0][inps.input_ids.shape[1]:], skip_special_tokens=True).split("###")[0].strip()
                    scores.append(compute_rouge1(pred, ex["output"]))
                
                avg = sum(scores)/len(scores)
                t_name = CLIENT_TASK_MAPPING.get(cid, f"C{cid}")
                print(f" {t_name}: {avg:.4f}")
                client_scores[cid] = avg

                with open(csv_p, 'a') as f: 
                    csv.writer(f).writerow([
                        time.strftime("%Y-%m-%d %H:%M:%S"),
                        os.path.basename(cfg["model"]["path"]),
                        f"FedTT ({t_name})",
                        r,
                        "ROUGE-1",
                        avg,
                        subspace_params,
                        full_lora_params,
                        trainable_params,
                        all_param,
                        f"{100 * trainable_params / all_param:.4f}%"
                    ])
            
            update_summary_csv(cfg["output_dir"], "FedTT", r, client_scores)

if __name__ == "__main__":
    main()