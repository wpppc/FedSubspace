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
import yaml  # <--- [修复] 之前遗漏了这一行
from collections import Counter

# ----------------------------------------------------------------
# 2. 延迟导入重型库 (PyTorch/Transformers)
# ----------------------------------------------------------------
print(">> [Init] Importing PyTorch & Transformers...", flush=True)
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType, set_peft_model_state_dict, get_peft_model_state_dict
from tqdm import tqdm

# 你的自定义模块
from data.dataset_tasks import GenericGenDataset, GenCollator

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
#                 FedIT Classes
# ============================================================

class FedITClient:
    def __init__(self, client_id, model, dataloader, local_epochs=1, lr=3e-4, device="cuda", dtype=torch.float16):
        self.client_id = client_id
        self.model = model
        self.dataloader = dataloader
        self.local_epochs = local_epochs
        self.lr = lr
        self.device = device
        self.dtype = dtype

    def train(self):
        self.model.train()
        optimizer = AdamW(self.model.parameters(), lr=self.lr)
        
        # Setup scaler for fp16 (disable for bfloat16)
        scaler = torch.amp.GradScaler('cuda', enabled=(self.dtype == torch.float16))
        
        for epoch in range(self.local_epochs):
            # Added tqdm for progress visibility
            with tqdm(self.dataloader, desc=f"Client {self.client_id} Epoch {epoch+1}/{self.local_epochs}", leave=False) as pbar:
                for batch in pbar:
                    input_ids = batch["input_ids"].to(self.device)
                    attention_mask = batch["attention_mask"].to(self.device)
                    labels = batch["labels"].to(self.device)
                    
                    optimizer.zero_grad()
                    
                    with torch.amp.autocast('cuda', dtype=self.dtype):
                        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                        loss = outputs.loss
                    
                    if torch.isnan(loss):
                        print(f"\n[Warning] NaN loss detected at Client {self.client_id}, Epoch {epoch+1}! Skipping step.")
                        optimizer.zero_grad()
                        del outputs, loss
                        continue

                    scaler.scale(loss).backward()
                    
                    # Gradient Clipping & Monitoring
                    # Unscale before clipping
                    scaler.unscale_(optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    
                    # Manual check for NaN/Inf gradients when using bfloat16 (scaler disabled)
                    if not scaler.is_enabled():
                        if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                            print(f"\n[Warning] NaN/Inf gradient detected at Client {self.client_id}! Skipping step.")
                            optimizer.zero_grad()
                            del outputs, loss
                            continue

                    scaler.step(optimizer)
                    scaler.update()
                    
                    # Update progress bar with loss and gradient norm
                    current_loss = loss.item()
                    
                    pbar.set_postfix(loss=f"{current_loss:.4f}", grad=f"{grad_norm.item():.2f}")
                    
                    del outputs, loss

    def get_parameters(self):
        state_dict = get_peft_model_state_dict(self.model)
        return {k: v.cpu() for k, v in state_dict.items()}

class FedITServer:
    def __init__(self):
        self.global_state_dict = None

    def aggregate(self, updates, sizes):
        total_samples = sum(sizes)
        aggregated_dict = copy.deepcopy(updates[0])
        
        for key in aggregated_dict.keys():
            aggregated_dict[key] = torch.zeros_like(aggregated_dict[key])
            
        for update, size in zip(updates, sizes):
            weight = size / total_samples
            for key in update.keys():
                aggregated_dict[key] += update[key] * weight
                
        self.global_state_dict = aggregated_dict
        return aggregated_dict

# ============================================================
#                     Main Execution
# ============================================================

def main():
    # Hardcoded config path for simplicity
    cfg_path = "configs/fedsubspace_flan.yaml"
    
    print(f">> [Main] Loading config from {cfg_path}...", flush=True)
    cfg = yaml.safe_load(open(cfg_path,"r"))
    
    cfg["output_dir"] = "outputs/fedit_flan"
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
    base_model = AutoModelForCausalLM.from_pretrained(
        cfg["model"]["path"], 
        torch_dtype=torch_dtype, 
        device_map={"": "cuda"}
    )
    
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["path"])
    tokenizer.pad_token_id = tokenizer.eos_token_id or 0

    print(">> [Main] Initializing LoRA...", flush=True)
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=cfg["lora"]["r"],
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=cfg["lora"]["target_modules"]
    )
    
    model = get_peft_model(base_model, peft_config)
    model.print_trainable_parameters()
    
    # Calculate params for logging
    trainable_params, all_param = get_trainable_parameters(model)
    subspace_params = 0 # FedIT doesn't use subspace
    full_lora_params = trainable_params # FedIT is just LoRA

    server = FedITServer()
    init_params = get_peft_model_state_dict(model)
    server.global_state_dict = {k: v.cpu() for k, v in init_params.items()}

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
    min_lr = 1e-5
    warmup_rounds = 5

    # Loop
    for r in range(cfg["federated"]["rounds"]):
        print(f"\n=== Round {r} ===", flush=True)
        
        # LR Scheduler: Constant for first 5 rounds, then Cosine Decay
        if r < warmup_rounds:
            current_lr = initial_lr
        else:
            # Cosine Decay
            progress = (r - warmup_rounds) / (cfg["federated"]["rounds"] - warmup_rounds)
            current_lr = min_lr + 0.5 * (initial_lr - min_lr) * (1 + math.cos(progress * math.pi))
        
        # FedIT (Full LoRA) might need lower LR than Subspace methods
        # Heuristic: Reduce LR for FedIT by factor of 2 or 3 compared to Subspace
        # current_lr = current_lr * 0.5 
        
        print(f"Current LR: {current_lr:.2e}")

        client_updates = []
        client_sizes = []
        
        # Train
        for cid, dl_info in enumerate(client_dataloaders):
            if dl_info is None: continue
            dl, collator = dl_info
            
            # Sync
            set_peft_model_state_dict(model, server.global_state_dict)
            
            # Re-initialize optimizer state implicitly by creating new client/optimizer each round
            # This is standard FedAvg (stateless clients)
            client = FedITClient(
                client_id=cid, 
                model=model, 
                dataloader=dl,
                local_epochs=cfg["train"]["local_epochs"],
                lr=current_lr,
                device="cuda",
                dtype=torch_dtype
            )
            
            # Train
            client.train()
            
            client_updates.append(client.get_parameters())
            client_sizes.append(len(dl.dataset))
            
            del client
            gc.collect()
            torch.cuda.empty_cache()
            
            # Simple progress indicator
            print(f"Client {cid} done.", flush=True)
        
        print(f" Aggregating...", flush=True)
        new_state_dict = server.aggregate(client_updates, client_sizes)
        set_peft_model_state_dict(model, new_state_dict)
        
        # Eval
        if cfg["eval"]["enabled"] and (r % cfg["eval"].get("eval_every", 1) == 0):
            print(f" Evaluating...", flush=True)
            model.eval()
            
            # 1. Global Eval
            scores = []
            for ex in tqdm(global_eval_data, desc="Global Eval", leave=False):
                # Use the new prompt template for evaluation too!
                raw_input = ex["input"]
                prompt = f"### Instruction:\n{raw_input}\n\n### Response:\n"
                
                inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
                with torch.no_grad():
                    outputs = model.generate(**inputs, max_new_tokens=50, pad_token_id=tokenizer.eos_token_id)
                
                input_len = inputs["input_ids"].shape[1]
                pred = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
                if "###" in pred:
                    pred = pred.split("###")[0]
                pred = pred.strip()
                scores.append(compute_rouge1(pred, ex["output"]))
            
            avg_rouge = sum(scores)/len(scores) if scores else 0
            print(f" >> Round {r} Global ROUGE: {avg_rouge:.4f}", flush=True)
            
            with open(csv_file, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    time.strftime("%Y-%m-%d %H:%M:%S"),
                    os.path.basename(cfg["model"]["path"]),
                    "FedIT",
                    r,
                    "ROUGE-1",
                    avg_rouge,
                    subspace_params,
                    full_lora_params,
                    trainable_params,
                    all_param,
                    f"{100 * trainable_params / all_param:.4f}%"
                ])

            # 2. Task-Specific Evaluation (Conditional)
            if avg_rouge >= 0.4:
                print(f" Running Task-Specific Evaluation (Global ROUGE {avg_rouge:.4f} >= 0.4)...", flush=True)
                client_scores = {}
                
                for cid in range(cfg["data"]["num_clients"]):
                    test_path = os.path.join(cfg["data"]["root"], f"test_{cid}.json")
                    if not os.path.exists(test_path): continue
                    
                    with open(test_path, "r") as f:
                        test_data = json.load(f)
                    if cfg["eval"]["max_samples"]: test_data = test_data[:cfg["eval"]["max_samples"]]
                    
                    c_scores = []
                    for ex in tqdm(test_data, desc=f"Client {cid} Eval", leave=False):
                        raw_input = ex["input"]
                        prompt = f"### Instruction:\n{raw_input}\n\n### Response:\n"
                        
                        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
                        with torch.no_grad():
                            outputs = model.generate(**inputs, max_new_tokens=50, pad_token_id=tokenizer.eos_token_id)
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
                            f"FedIT ({task_name})",
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
                update_summary_csv(cfg["output_dir"], "FedIT", r, client_scores)
            else:
                print(f" Skipping Task-Specific Evaluation (Global ROUGE {avg_rouge:.4f} < 0.4)", flush=True)

        # Save Checkpoint
        ckpt_path = os.path.join(cfg["output_dir"], f"global_round{r}.pt")
        torch.save(server.global_state_dict, ckpt_path)
        print(f" >> Saved checkpoint: {ckpt_path}", flush=True)

    print("\n>> Done.", flush=True)

if __name__ == "__main__":
    main()