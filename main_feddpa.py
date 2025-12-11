
# ----------------------------------------------------------------
# 1. Init
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
import math
from collections import Counter

print(">> [Init] Importing PyTorch & Transformers...", flush=True)
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

# Custom Modules
from data.dataset_tasks import GenericGenDataset, GenCollator
from models.llama_wrapper_dpa import FedDPAModelWrapper
from federated.client_dpa import FedDPAClient
from models.lora_utils import extract_lora_shapes

print(">> [Init] Libraries loaded.", flush=True)

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

# Task Mapping
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
    summary_file = os.path.join(output_dir, "summary_table.csv")
    task_columns = sorted(list(set(CLIENT_TASK_MAPPING.values())))
    columns = ["Method", "Round"] + task_columns + ["Average"]
    file_exists = os.path.exists(summary_file)
    
    row_data = {
        "Method": method_name,
        "Round": round_num,
        "Average": 0.0
    }
    
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
        if not file_exists:
            writer.writeheader()
        writer.writerow(row_data)
    print(f">> [Summary] Updated {summary_file}")

def main():
    # 1. Load Config
    config_path = "configs/fedsubspace_flan.yaml"
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    
    # Override output dir for FedDPA
    cfg["output_dir"] = "outputs/fed_dpa"
    os.makedirs(cfg["output_dir"], exist_ok=True)
    
    print(f">> [Main] Output Dir: {cfg['output_dir']}")
    
    # 2. Load Base Model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Auto-select best dtype
    torch_dtype = torch.float16
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        print(">> [Main] bfloat16 supported. Using bfloat16 for stability.", flush=True)
        torch_dtype = torch.bfloat16
    
    print(f">> [Main] Loading Base Model: {cfg['model']['path']}")
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["path"], use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    # Force model to load on the correct device without splitting across GPUs
    # device_map="auto" allows splitting across multiple GPUs
    base_model = AutoModelForCausalLM.from_pretrained(
        cfg["model"]["path"],
        torch_dtype=torch_dtype,
        device_map="auto" # Enable multi-GPU splitting
    )
    
    # Enable Gradient Checkpointing to save memory
    base_model.gradient_checkpointing_enable()
    base_model.enable_input_require_grads()
    
    # Freeze Base Model
    for param in base_model.parameters():
        param.requires_grad = False
    
    # Override batch size to 1 to prevent OOM
    cfg["train"]["batch_size"] = 1
    print(f">> [Main] Batch Size set to {cfg['train']['batch_size']} to prevent OOM")
    
    # 3. Initialize FedDPA Wrapper
    print(">> [Main] Initializing FedDPAModelWrapper...")
    shared_model = FedDPAModelWrapper(
        base_model,
        lora_r=cfg["lora"]["r"],
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=cfg["lora"]["target_modules"]
    )

    # Calculate params for logging
    trainable_params, all_param = get_trainable_parameters(shared_model)
    lora_shapes = extract_lora_shapes(base_model, target_modules=cfg["lora"]["target_modules"], r=cfg["lora"]["r"])
    full_lora_params = sum(A.numel() + B.numel() for A, B in lora_shapes.values())
    subspace_params = full_lora_params # FedDPA operates in the full LoRA space
    
    print(f"Full LoRA Params (Theoretical): {full_lora_params:,}")
    print(f"Subspace Params (Equivalent): {subspace_params:,}")
    print(f"Trainable Params (Actual): {trainable_params:,}")
    print(f"All Params: {all_param:,}")
    print(f"Trainable Ratio: {100 * trainable_params / all_param:.4f}%")
    
    # 4. Initialize Server State (Global Adapter)
    # We store the state dict of the global adapter
    server_global_state = shared_model.get_global_state_dict()
    
    # 5. Load Datasets
    print(f">> [Main] Loading Datasets...", flush=True)
    client_dataloaders = []
    for cid in range(cfg["data"]["num_clients"]):
        data_path = os.path.join(cfg["data"]["root"], f"client_{cid}.json")
        if not os.path.exists(data_path):
            client_dataloaders.append(None)
            continue
        ds = GenericGenDataset(data_path)
        collator = GenCollator(tokenizer, max_len=cfg["data"]["cutoff_len"], train_on_inputs=cfg["data"]["train_on_inputs"])
        dl = DataLoader(ds, batch_size=cfg["train"]["batch_size"], shuffle=True, collate_fn=lambda b: collator(b), num_workers=0)
        client_dataloaders.append((dl, collator))
        
    # CSV Init
    csv_file = os.path.join(cfg["output_dir"], "experiment_results.csv")
    if not os.path.exists(csv_file):
        with open(csv_file, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Timestamp", "BaseModel", "Dataset", "Round", "Metric", "Value", "SubspaceParams", "FullLoRAParams", "TrainableParams", "AllParams", "TrainableRatio"])

    # 6. Training Loop
    initial_lr = float(cfg["train"]["lr"])
    min_lr = 1e-6
    warmup_rounds = 0
    
    for r in range(cfg["federated"]["rounds"]):
        print(f"\n=== Round {r} (FedDPA) ===", flush=True)
        
        # LR Scheduler
        if r < warmup_rounds:
            current_lr = initial_lr
        else:
            progress = (r - warmup_rounds) / (cfg["federated"]["rounds"] - warmup_rounds)
            current_lr = min_lr + 0.5 * (initial_lr - min_lr) * (1 + math.cos(progress * math.pi))
        print(f"Current LR: {current_lr:.2e}")
        
        global_updates = []
        sizes = []
        
        # Train Clients
        for cid, dl_info in enumerate(client_dataloaders):
            if dl_info is None: continue
            dl, collator = dl_info
            
            client = FedDPAClient(
                client_id=cid,
                model=shared_model,
                tokenizer=tokenizer,
                dataloader=dl,
                output_dir=cfg["output_dir"],
                local_epochs=cfg["train"]["local_epochs"],
                lr=current_lr,
                device=base_model.device, # Use model's device
                data_collator=collator,
                dtype=torch_dtype
            )
            
            # Load Global & Local
            client.load_state(server_global_state)
            
            # Train
            client.train()
            
            # Get Update (Global only)
            global_updates.append(client.get_update_for_server())
            sizes.append(len(dl.dataset))
            
            # Cleanup
            client.model = None
            del client
            gc.collect()
            torch.cuda.empty_cache()
            print(f"Client {cid} done.", flush=True)
            
        # Aggregate Global Updates
        print(f" Aggregating...", flush=True)
        total_samples = sum(sizes)
        
        # Initialize aggregated state with zeros
        agg_state = {}
        first_update = global_updates[0]
        for k in first_update.keys():
            agg_state[k] = torch.zeros_like(first_update[k])
            
        for update, size in zip(global_updates, sizes):
            weight = size / total_samples
            for k in agg_state.keys():
                agg_state[k] += update[k] * weight
                
        server_global_state = agg_state
        
        # Save Checkpoint
        torch.save(server_global_state, os.path.join(cfg["output_dir"], f"global_round{r}.pt"))
        
        # Evaluation (Personalized)
        # We need to evaluate each client using their PERSONALIZED model (Global + Local)
        # Since we just trained them, the local state on disk is up to date.
        # We need to reload the model for each client to evaluate.
        
        if r % 1 == 0: # Eval every round
            print(f" Evaluating Personalized Models...", flush=True)
            client_scores = {}
            
            for cid in range(cfg["data"]["num_clients"]):
                test_path = os.path.join(cfg["data"]["root"], f"test_{cid}.json")
                if not os.path.exists(test_path): continue
                
                # Load Test Data
                with open(test_path, "r") as f:
                    test_data = json.load(f)
                if cfg["eval"]["max_samples"]: 
                    test_data = test_data[:cfg["eval"]["max_samples"]]
                
                # Setup Client Model for Eval
                # Load Global
                shared_model.load_state_dict(server_global_state, strict=False)
                # Load Local
                local_state_path = os.path.join(cfg["output_dir"], f"client_{cid}_local_state.pt")
                if os.path.exists(local_state_path):
                    local_state = torch.load(local_state_path, map_location=device)
                    shared_model.load_state_dict(local_state, strict=False)
                else:
                    # If no local state (shouldn't happen after training), init from global
                    for module in shared_model.base_model.modules():
                        if hasattr(module, "adapter_global") and hasattr(module, "adapter_local"):
                            module.adapter_local.lora_A.data.copy_(module.adapter_global.lora_A.data)
                            module.adapter_local.lora_B.data.copy_(module.adapter_global.lora_B.data)
                
                shared_model.eval()
                
                c_scores = []
                for ex in tqdm(test_data, desc=f"Client {cid} Eval", leave=False):
                    raw_input = ex["input"]
                    prompt = f"### Instruction:\n{raw_input}\n\n### Response:\n"
                    inputs = tokenizer(prompt, return_tensors="pt").to(device)
                    with torch.no_grad():
                        outputs = base_model.generate(**inputs, max_new_tokens=50, pad_token_id=tokenizer.eos_token_id)
                    input_len = inputs["input_ids"].shape[1]
                    pred = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
                    if "###" in pred: pred = pred.split("###")[0]
                    pred = pred.strip()
                    c_scores.append(compute_rouge1(pred, ex["output"]))
                
                avg_score = sum(c_scores)/len(c_scores) if c_scores else 0
                client_scores[cid] = avg_score
                task_name = CLIENT_TASK_MAPPING.get(cid, f"Client {cid}")
                print(f"  {task_name}: {avg_score:.4f}")

                # Log individual client score to main CSV
                with open(csv_file, mode='a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        time.strftime("%Y-%m-%d %H:%M:%S"),
                        "llama-2-7b",
                        f"FedDPA ({task_name})",
                        r,
                        "ROUGE-1",
                        avg_score,
                        subspace_params,
                        full_lora_params,
                        trainable_params,
                        all_param,
                        f"{100 * trainable_params / all_param:.4f}%"
                    ])
            
            update_summary_csv(cfg["output_dir"], "FedDPA", r, client_scores)

if __name__ == "__main__":
    main()
