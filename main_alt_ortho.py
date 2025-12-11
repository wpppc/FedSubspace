
# ----------------------------------------------------------------
# 1. 确保最先导入系统库，并设置环境变量 (防止 Tokenizer 并行死锁)
# ----------------------------------------------------------------
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys
import time
import re
import csv
import json
import gc
import copy
import yaml
from collections import Counter

# ----------------------------------------------------------------
# 2. 延迟导入重型库 (PyTorch/Transformers)
# ----------------------------------------------------------------
print(">> [Init] Importing PyTorch & Transformers...", flush=True)
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

# 你的自定义模块
from data.dataset_tasks import GenericGenDataset, GenCollator
from models.lora_utils import extract_lora_shapes
from models.llama_wrapper_alt import FedDualSubspaceModelWrapper
from federated.client_alt_ortho import FedAltClientOrtho
from federated.server import FedSubspaceServer
from subspace.projection import RandomSubspaceProjection
from subspace.utils import unflatten_lora_params

print(">> [Init] Libraries loaded.", flush=True)

# ============================================================
#                 Helper Functions (Pure Python)
# ============================================================

# Task Mapping for Logging
CLIENT_TASK_MAPPING = {
    0: "Sentiment Analysis",
    1: "Natural Language Inference",
    2: "Text Classification",
    3: "Commonsense Reasoning",
    4: "Paraphrase Detection",
    5: "Struct to Text",
    6: "Reading Comprehension",
    7: "Coreference Resolution"
}

def update_summary_csv(output_dir, method_name, round_num, client_scores):
    """
    Updates a summary CSV with columns: Method, Round, [Task Names...], Average
    """
    summary_file = os.path.join(output_dir, "summary_table.csv")
    
    # Define columns
    task_columns = sorted(list(set(CLIENT_TASK_MAPPING.values())))
    columns = ["Method", "Round"] + task_columns + ["Average"]
    
    # Check if file exists to write header
    file_exists = os.path.exists(summary_file)
    
    row_data = {
        "Method": method_name,
        "Round": round_num,
        "Average": 0.0
    }
    
    # Fill scores
    total_score = 0
    count = 0
    for cid, score in client_scores.items():
        task_name = CLIENT_TASK_MAPPING.get(cid, f"Client {cid}")
        row_data[task_name] = f"{score * 100:.2f}" # Convert to percentage
        total_score += score
        count += 1
        
    if count > 0:
        row_data["Average"] = f"{(total_score / count) * 100:.2f}"
    
    # Write to CSV
    with open(summary_file, mode='a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row_data)
    
    print(f">> [Summary] Updated {summary_file}")

def get_trainable_parameters(model):
    """
    Returns (trainable_params, all_param)
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        num_params = param.numel()
        # if using DS Zero 3 and the weights are initialized empty
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel

        # Due to the design of 4bit linear layers from bitsandbytes
        if param.__class__.__name__ == "Params4bit":
            num_params = num_params * 2

        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params
    
    return trainable_params, all_param

def compute_rouge1(prediction, reference):
    """
    Robust ROUGE-1 F1 score (Local implementation, no network required).
    """
    # Simple tokenizer
    def get_tokens(text):
        return re.findall(r'\w+', text.lower())

    pred_tokens = get_tokens(prediction)
    ref_tokens = get_tokens(reference)

    if not pred_tokens or not ref_tokens:
        return 0.0

    # Use Counter to handle word frequency correctly
    pred_counts = Counter(pred_tokens)
    ref_counts = Counter(ref_tokens)
    
    # Calculate overlap count
    overlap = sum((pred_counts & ref_counts).values())

    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)

    if precision + recall == 0:
        return 0.0
    
    f1 = 2 * (precision * recall) / (precision + recall)
    return f1

# ============================================================
#                     Main Execution
# ============================================================

def main():
    # Hardcoded config path for simplicity
    cfg_path = "configs/fedsubspace_flan.yaml"
    
    print(f">> [Main] Loading config from {cfg_path}...", flush=True)
    cfg = yaml.safe_load(open(cfg_path,"r"))
    
    # Change output dir for Ortho experiment
    cfg["output_dir"] = "outputs/fed_alt_ortho"
    os.makedirs(cfg["output_dir"], exist_ok=True)

    print(f">> [Main] Loading base model: {cfg['model']['path']}", flush=True)
    
    # Auto-select best dtype (bfloat16 is much more stable for LLaMA-2)
    torch_dtype = torch.float16
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        print(">> [Main] bfloat16 supported. Using bfloat16 for stability.", flush=True)
        torch_dtype = torch.bfloat16
    else:
        print(">> [Main] bfloat16 NOT supported. Using float16 (risk of NaN/Overflow).", flush=True)

    # Explicit device map
    base = AutoModelForCausalLM.from_pretrained(
        cfg["model"]["path"], 
        torch_dtype=torch_dtype, 
        device_map={"": "cuda"}
    )
    
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["path"])
    tokenizer.pad_token_id = tokenizer.eos_token_id or 0

    # Freeze Base
    for p in base.parameters(): p.requires_grad=False

    print(">> [Main] Initializing Dual Subspace Model...", flush=True)
    lora_shapes = extract_lora_shapes(base, target_modules=cfg["lora"]["target_modules"], r=cfg["lora"]["r"])
    
    # Use Dual Subspace Wrapper
    shared_model = FedDualSubspaceModelWrapper(base, lora_shapes, d_s=cfg["subspace"]["dim"], seed=cfg["subspace"]["seed"],
                                          target_modules=cfg["lora"]["target_modules"])

    # Setup Server (Standard Subspace Server is fine, it just aggregates vectors)
    server = FedSubspaceServer(cfg["subspace"]["dim"])
    
    # Initialize server theta from model's random initialization (Global Adapter)
    # Note: FedDualSubspaceModelWrapper initializes both adapters with same seed, so they start identical.
    server.global_theta = shared_model.adapter_global.theta_s.detach().cpu().clone()
    
    # Also need to track Global Gates?
    # Server logic in `server.py` might need update if we want to aggregate gates properly.
    # But `FedSubspaceServer` only aggregates `theta`.
    # We can handle gate aggregation manually in the main loop like in `main_alt.py`.
    # So we initialize a `global_state` dict.
    server.global_state = {
        'theta': server.global_theta,
        'gates': {} # Will be populated after first round
    }

    # Calculate params for logging
    trainable_params, all_param = get_trainable_parameters(shared_model)
    subspace_params = cfg["subspace"]["dim"]
    full_lora_params = sum(A.numel() + B.numel() for A, B in lora_shapes.values())

    print(f"Full LoRA Params (Theoretical): {full_lora_params:,}")
    print(f"Subspace Params (Configured): {subspace_params:,}")
    print(f"Trainable Params (Actual): {trainable_params:,}")
    print(f"All Params: {all_param:,}")
    print(f"Trainable Ratio: {100 * trainable_params / all_param:.4f}%")

    print(f">> [Main] Loading Datasets...", flush=True)
    client_dataloaders = []
    
    for cid in range(cfg["data"]["num_clients"]):
        data_path = os.path.join(cfg["data"]["root"], f"client_{cid}.json")
        if not os.path.exists(data_path):
            client_dataloaders.append(None)
            continue
            
        ds = GenericGenDataset(data_path)
        collator = GenCollator(tokenizer, max_len=cfg["data"]["cutoff_len"], train_on_inputs=cfg["data"]["train_on_inputs"])
        
        # Critical fix: num_workers=0
        dl = DataLoader(ds, batch_size=cfg["train"]["batch_size"], shuffle=True, collate_fn=lambda b: collator(b), num_workers=0)
        client_dataloaders.append((dl, collator))
    
    print(f">> [Main] Start Training Loop ({cfg['federated']['rounds']} rounds)...", flush=True)

    # CSV Init
    csv_file = os.path.join(cfg["output_dir"], "experiment_results.csv")
    if not os.path.exists(csv_file):
        with open(csv_file, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Timestamp", "BaseModel", "Dataset", "Round", "Metric", "Value", "SubspaceParams", "FullLoRAParams", "TrainableParams", "AllParams", "TrainableRatio"])

    # Load Eval Data
    global_eval_path = os.path.join(cfg["data"]["root"], "global_eval.json")
    global_eval_data = []
    if os.path.exists(global_eval_path):
        with open(global_eval_path, "r") as f:
            global_eval_data = json.load(f)
        if cfg["eval"]["max_samples"]: 
            global_eval_data = global_eval_data[:cfg["eval"]["max_samples"]]

    import math
    initial_lr = float(cfg["train"]["lr"])
    min_lr = 1e-6  # Modified: Lower minimum LR to prevent overfitting
    warmup_rounds = 5 # Modified: Remove warmup to start decay immediately

    # Loop
    for r in range(cfg["federated"]["rounds"]):
        print(f"\n=== Round {r} (FedALT+Ortho) ===", flush=True)
        
        # LR Scheduler: Cosine Decay from start
        if r < warmup_rounds:
            current_lr = initial_lr
        else:
            # Cosine Decay
            progress = (r - warmup_rounds) / (cfg["federated"]["rounds"] - warmup_rounds)
            current_lr = min_lr + 0.5 * (initial_lr - min_lr) * (1 + math.cos(progress * math.pi))
        
        print(f"Current LR: {current_lr:.2e}")

        thetas_to_aggregate = []
        sizes = []
        
        # Train
        # We need to store client updates to calculate RoW later
        # But we can't wait until everyone finishes to train, because we need to load the correct global state.
        # Wait, RoW means: Client k uses Global = Average(Others).
        # This Global state is calculated from the PREVIOUS round's updates.
        
        # So, at the start of Round r, we should have a list of updates from Round r-1.
        # Let's assume server.global_state stores the AGGREGATED global state (Average of All).
        # We need to store individual client updates from the previous round to compute RoW.
        
        # Initialize previous_updates if it's the first round
        if not hasattr(server, 'previous_updates'):
            server.previous_updates = [None] * cfg["data"]["num_clients"]
            # Initialize with zeros or the initial global state
            # For simplicity, if it's Round 0, everyone uses the initial global state (which is random or zero).
            # RoW is only meaningful after Round 0.
        
        for cid, dl_info in enumerate(client_dataloaders):
            if dl_info is None: continue
            dl, collator = dl_info
            
            # Calculate RoW Global State for this client
            if r == 0 or any(u is None for u in server.previous_updates):
                # Round 0: Use the unified global state (Average of All, or Initial)
                client_global_state = server.global_state
            else:
                # Round > 0: Calculate RoW = (Sum_All - Self) / (K-1)
                # 1. Reconstruct Sum_All from server.global_state (which is Average)
                # Sum_All = Average * K
                # But wait, server.global_state might be weighted average.
                # Let's do it properly: Sum individual updates.
                
                # Sum all theta
                sum_theta = torch.zeros_like(server.previous_updates[0]['theta'])
                for u in server.previous_updates:
                    if u is not None:
                        sum_theta += u['theta']
                
                # Subtract self
                self_theta = server.previous_updates[cid]['theta']
                row_theta = (sum_theta - self_theta) / (cfg["data"]["num_clients"] - 1)
                
                # Gates: We can also do RoW for gates if we want, or just use Global Average.
                # For simplicity and stability, let's use Global Average for Gates, 
                # because Gates are structural parameters, not knowledge parameters.
                # Or we can do RoW for gates too. Let's stick to Global Average for Gates for now to avoid complexity.
                client_global_state = {
                    'theta': row_theta,
                    'gates': server.global_state['gates']
                }

            # Instantiate client on the fly
            # Use FedAltClientOrtho with lambda_ortho=0.1
            client = FedAltClientOrtho(
                client_id=cid, 
                model=shared_model, 
                tokenizer=tokenizer, 
                dataloader=dl,
                output_dir=cfg["output_dir"], 
                local_epochs=cfg["train"]["local_epochs"],
                lr=current_lr,
                device="cuda", 
                data_collator=collator, 
                dtype=torch_dtype,
                lambda_ortho=0.1 # Orthogonality Strength
            )
            
            # Client loads Global (Frozen) and Local (Trainable)
            client.load_vectors(client_global_state)
            client.train()
            
            # Client sends Local Vector for aggregation
            update = client.get_update_for_server()
            thetas_to_aggregate.append(update)
            sizes.append(len(client.dataloader.dataset))

            # Explicit cleanup
            client.model = None
            del client
            gc.collect()
            torch.cuda.empty_cache()
            
            print(f"Client {cid} done.", flush=True)
        
        # Store updates for next round's RoW calculation
        # We need to map updates back to client IDs. 
        # Since we iterate cid in order, thetas_to_aggregate is ordered by cid (skipping None).
        # But wait, if some clients are skipped (dl_info is None), the index will shift.
        # Let's be careful.
        
        # Update server.previous_updates
        current_update_idx = 0
        for cid, dl_info in enumerate(client_dataloaders):
            if dl_info is not None:
                server.previous_updates[cid] = thetas_to_aggregate[current_update_idx]
                current_update_idx += 1
        
        print(f" Aggregating...", flush=True)
        
        # Manual Aggregation for Theta AND Gates
        # 1. Aggregate Theta
        total_samples = sum(sizes)
        agg_theta = torch.zeros_like(thetas_to_aggregate[0]['theta'])
        
        # 2. Aggregate Gates
        # Initialize agg_gates with zeros based on first client's structure
        agg_gates = {}
        first_gates = thetas_to_aggregate[0]['gates']
        for k in first_gates.keys():
            agg_gates[k] = torch.zeros_like(first_gates[k])
            
        for update, size in zip(thetas_to_aggregate, sizes):
            weight = size / total_samples
            
            # Theta
            agg_theta += update['theta'] * weight
            
            # Gates
            for k in agg_gates.keys():
                if k in update['gates']:
                    agg_gates[k] += update['gates'][k] * weight
        
        # Update Server State
        server.global_state = {
            'theta': agg_theta,
            'gates': agg_gates
        }
        
        # Save Checkpoint
        torch.save(server.global_state, os.path.join(cfg["output_dir"], f"theta_global_round{r}.pt"))
        
        # Eval
        if cfg["eval"]["enabled"] and (r % cfg["eval"].get("eval_every", 1) == 0):
            print(f" Evaluating...", flush=True)
            
            # 1. Load Global Theta
            shared_model.adapter_global.theta_s.data.copy_(server.global_state['theta'].to(base.device))
            
            # 2. Load Global Gates
            if 'gates' in server.global_state:
                for name, val in server.global_state['gates'].items():
                    if name.endswith(".gate_g"):
                        module_path = name.rsplit('.', 1)[0]
                        module = shared_model.get_submodule(module_path)
                        module.gate_g.data.copy_(val.to(base.device))
            
            # 3. Disable Local Branch for Global Eval
            backup_gates_l = {}
            for name, module in shared_model.named_modules():
                if hasattr(module, "gate_l"):
                    backup_gates_l[name] = module.gate_l.data.clone()
                    module.gate_l.data.fill_(0.0)
            
            # Global Eval Loop
            scores = []
            for ex in tqdm(global_eval_data, desc="Global Eval", leave=False):
                raw_input = ex["input"]
                prompt = f"### Instruction:\n{raw_input}\n\n### Response:\n"
                
                inputs = tokenizer(prompt, return_tensors="pt").to(base.device)
                with torch.no_grad():
                    outputs = base.generate(**inputs, max_new_tokens=50, pad_token_id=tokenizer.eos_token_id)
                
                input_len = inputs["input_ids"].shape[1]
                pred = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
                if "###" in pred:
                    pred = pred.split("###")[0]
                pred = pred.strip()
                scores.append(compute_rouge1(pred, ex["output"]))
            
            # Restore Local Gates
            for name, val in backup_gates_l.items():
                module = shared_model.get_submodule(name)
                module.gate_l.data.copy_(val)
            
            avg_rouge = sum(scores)/len(scores) if scores else 0
            print(f" >> Round {r} Global ROUGE: {avg_rouge:.4f}", flush=True)
            
            with open(csv_file, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    time.strftime("%Y-%m-%d %H:%M:%S"),
                    os.path.basename(cfg["model"]["path"]),
                    "FedALT+Ortho",
                    r,
                    "ROUGE-1",
                    avg_rouge,
                    subspace_params,
                    full_lora_params,
                    trainable_params,
                    all_param,
                    f"{100 * trainable_params / all_param:.4f}%"
                ])

            # 4. Task-Specific Evaluation (Conditional)
            if avg_rouge >= 0.4:
                print(f" Running Task-Specific Evaluation (Global ROUGE {avg_rouge:.4f} >= 0.4)...", flush=True)
                client_scores = {}
                
                for cid in range(cfg["data"]["num_clients"]):
                    test_path = os.path.join(cfg["data"]["root"], f"test_{cid}.json")
                    if not os.path.exists(test_path): continue
                    
                    with open(test_path, "r") as f:
                        test_data = json.load(f)
                    if cfg["eval"]["max_samples"]: test_data = test_data[:cfg["eval"]["max_samples"]]
                    
                    # Load Local State for this client
                    local_state_path = os.path.join(cfg["output_dir"], f"client_{cid}_local_state.pt")
                    if not os.path.exists(local_state_path):
                        print(f"Warning: Local state for client {cid} not found. Skipping.")
                        continue
                    
                    local_state = torch.load(local_state_path, map_location=base.device)
                    
                    # Load Local Theta
                    shared_model.adapter_local.theta_s.data.copy_(local_state['theta'])
                    
                    # Load Local Gates
                    if 'gates' in local_state:
                        for name, val in local_state['gates'].items():
                            if name.endswith(".gate_l"):
                                module_path = name.rsplit('.', 1)[0]
                                module = shared_model.get_submodule(module_path)
                                module.gate_l.data.copy_(val.to(base.device))
                    
                    c_scores = []
                    for ex in tqdm(test_data, desc=f"Client {cid} Eval", leave=False):
                        raw_input = ex["input"]
                        prompt = f"### Instruction:\n{raw_input}\n\n### Response:\n"
                        
                        inputs = tokenizer(prompt, return_tensors="pt").to(base.device)
                        with torch.no_grad():
                            outputs = base.generate(**inputs, max_new_tokens=50, pad_token_id=tokenizer.eos_token_id)
                        input_len = inputs["input_ids"].shape[1]
                        pred = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
                        if "###" in pred:
                            pred = pred.split("###")[0]
                        pred = pred.strip()
                        c_scores.append(compute_rouge1(pred, ex["output"]))
                    
                    avg_c_score = sum(c_scores) / len(c_scores) if c_scores else 0
                    client_scores[cid] = avg_c_score
                    
                    task_name = CLIENT_TASK_MAPPING.get(cid, f"Client {cid}")
                    print(f"  {task_name}: {avg_c_score:.4f}")

                    # Log individual client score to main CSV
                    with open(csv_file, mode='a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([
                            time.strftime("%Y-%m-%d %H:%M:%S"),
                            os.path.basename(cfg["model"]["path"]),
                            f"FedALT+Ortho ({task_name})",
                            r,
                            "ROUGE-1",
                            avg_c_score,
                            subspace_params,
                            full_lora_params,
                            trainable_params,
                            all_param,
                            f"{100 * trainable_params / all_param:.4f}%"
                        ])
                
                # Update Summary Table CSV
                update_summary_csv(cfg["output_dir"], "FedALT+Ortho", r, client_scores)
            else:
                print(f" Skipping Task-Specific Evaluation (Global ROUGE {avg_rouge:.4f} < 0.4)", flush=True)

    print("\n>> Done.", flush=True)

if __name__ == "__main__":
    main()
