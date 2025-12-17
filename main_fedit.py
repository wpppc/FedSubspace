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
import math
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

# [关键修复] 禁用 SDPA 以防止 RecursionError / OOM
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_math_sdp(True)

# 你的自定义模块
from data.dataset_tasks import GenericGenDataset, GenCollator

print(">> [Init] Libraries loaded.", flush=True)

# ============================================================
#                 Helper Functions
# ============================================================

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

def compute_rouge1(prediction, reference):
    def get_tokens(text): return re.findall(r'\w+', text.lower())
    pred_tokens = get_tokens(prediction)
    ref_tokens = get_tokens(reference)
    if not pred_tokens or not ref_tokens: return 0.0
    pred_counts = Counter(pred_tokens)
    ref_counts = Counter(ref_tokens)
    overlap = sum((pred_counts & ref_counts).values())
    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)
    if precision + recall == 0: return 0.0
    return 2 * (precision * recall) / (precision + recall)

# ============================================================
#                 FedIT Client (Aligned with FedLESS)
# ============================================================

class FedITClient:
    def __init__(self, client_id, model, dataloader, local_epochs=1, lr=2e-4, 
                 device="cuda", dtype=torch.float16, gradient_accumulation_steps=1):
        self.client_id = client_id
        self.model = model
        self.dataloader = dataloader
        self.local_epochs = local_epochs
        self.lr = lr
        self.device = device
        self.dtype = dtype
        # [关键对齐] 增加梯度累积
        self.gradient_accumulation_steps = gradient_accumulation_steps

    def train(self):
        self.model.train()
        optimizer = AdamW(self.model.parameters(), lr=self.lr)
        
        # Bfloat16 不需要 Scaler，Float16 需要
        use_scaler = (self.dtype == torch.float16)
        scaler = torch.amp.GradScaler('cuda', enabled=use_scaler)
        
        global_step = 0
        
        for epoch in range(self.local_epochs):
            # 进度条
            with tqdm(self.dataloader, desc=f"Client {self.client_id} Epoch {epoch+1}/{self.local_epochs}", leave=False) as pbar:
                optimizer.zero_grad()
                
                for step, batch in enumerate(pbar):
                    input_ids = batch["input_ids"].to(self.device)
                    attention_mask = batch["attention_mask"].to(self.device)
                    labels = batch["labels"].to(self.device)
                    
                    with torch.amp.autocast('cuda', dtype=self.dtype):
                        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                        # [关键对齐] Loss 除以累积步数
                        loss = outputs.loss / self.gradient_accumulation_steps
                    
                    if torch.isnan(loss):
                        print(f"\n[Warning] NaN loss detected at Client {self.client_id}! Skipping.")
                        optimizer.zero_grad()
                        continue

                    scaler.scale(loss).backward()
                    
                    # [关键对齐] 只有在累积步数达到时才 Update
                    if (step + 1) % self.gradient_accumulation_steps == 0 or (step + 1) == len(self.dataloader):
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                        
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad()
                        global_step += 1
                    
                    # Log raw loss (restored scale) for display
                    current_loss = loss.item() * self.gradient_accumulation_steps
                    pbar.set_postfix(loss=f"{current_loss:.4f}")
                    del outputs, loss

    def get_parameters(self):
        # 仅返回 LoRA 参数
        state_dict = get_peft_model_state_dict(self.model)
        return {k: v.cpu() for k, v in state_dict.items()}

class FedITServer:
    def __init__(self):
        self.global_state_dict = None

    def aggregate(self, updates, sizes):
        total_samples = sum(sizes)
        # Deepcopy structure from first update
        aggregated_dict = copy.deepcopy(updates[0])
        
        # Zero out buffers
        for key in aggregated_dict.keys():
            aggregated_dict[key] = torch.zeros_like(aggregated_dict[key], dtype=torch.float32) # Use float32 for aggregation
            
        for update, size in zip(updates, sizes):
            weight = size / total_samples
            for key in update.keys():
                aggregated_dict[key] += update[key].to(torch.float32) * weight
                
        self.global_state_dict = aggregated_dict
        return aggregated_dict

# ============================================================
#                     Main Execution
# ============================================================

def main():
    cfg_path = "configs/fedsubspace_flan.yaml"
    print(f">> [Main] Loading config from {cfg_path}...", flush=True)
    cfg = yaml.safe_load(open(cfg_path,"r"))
    
    cfg["output_dir"] = "outputs/fedit_flan"
    os.makedirs(cfg["output_dir"], exist_ok=True)
    
    # [关键对齐] 获取梯度累积步数，如果 Config 里没有则默认为 8 (为了和 FedLESS 对齐)
    accum_steps = cfg["train"].get("gradient_accumulation_steps", 8)
    print(f">> [Config] Batch Size: {cfg['train']['batch_size']} | Accum Steps: {accum_steps} | Eff Batch: {cfg['train']['batch_size']*accum_steps}")

    print(f">> [Main] Loading base model: {cfg['model']['path']}", flush=True)
    
    torch_dtype = torch.float16
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        print(">> [Main] bfloat16 supported. Using bfloat16.", flush=True)
        torch_dtype = torch.bfloat16

    # [关键对齐] 使用 eager attention 避免递归错误
    base_model = AutoModelForCausalLM.from_pretrained(
        cfg["model"]["path"], 
        torch_dtype=torch_dtype, 
        device_map={"": "cuda"},
        attn_implementation="eager" 
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
    
    trainable_params, all_param = get_trainable_parameters(model)
    subspace_params = 0
    full_lora_params = trainable_params

    server = FedITServer()
    # Initialize Global State
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
        dl = DataLoader(ds, batch_size=cfg["train"]["batch_size"], shuffle=True, collate_fn=lambda b: collator(b), num_workers=0)
        client_dataloaders.append((dl, collator))
    
    # CSV Init
    csv_file = os.path.join(cfg["output_dir"], "experiment_results.csv")
    if not os.path.exists(csv_file):
        with open(csv_file, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Timestamp", "BaseModel", "Dataset", "Round", "Metric", "Value", "SubspaceParams", "FullLoRAParams", "TrainableParams", "AllParams", "TrainableRatio"])

    # Load Global Eval Data
    global_eval_path = os.path.join(cfg["data"]["root"], "global_eval.json")
    global_eval_data = []
    if os.path.exists(global_eval_path):
        with open(global_eval_path, "r") as f:
            global_eval_data = json.load(f)
        if cfg["eval"]["max_samples"]: 
            global_eval_data = global_eval_data[:cfg["eval"]["max_samples"]]

    # [关键对齐] 学习率调度配置 (和 FedLESS 保持一致)
    initial_lr = float(cfg["train"]["lr"])
    min_lr = 1e-6
    warmup_rounds = 1  # FedLESS 是 1
    total_rounds = cfg["federated"]["rounds"]

    # --- Resume Logic (断点续训) ---
    start_round = 0
    checkpoint_pattern = re.compile(r"global_round(\d+).pt")
    checkpoints = []
    if os.path.exists(cfg["output_dir"]):
        for f in os.listdir(cfg["output_dir"]):
            match = checkpoint_pattern.match(f)
            if match:
                checkpoints.append(int(match.group(1)))
    
    if checkpoints:
        last_round = max(checkpoints)
        checkpoint_path = os.path.join(cfg["output_dir"], f"global_round{last_round}.pt")
        print(f">> [Resume] Loading checkpoint: {checkpoint_path}")
        server.global_state_dict = torch.load(checkpoint_path, map_location="cpu")
        start_round = last_round + 1
        print(f">> Starting from Round {start_round}")

    # Training Loop
    for r in range(start_round, total_rounds):
        print(f"\n=== Round {r} ===", flush=True)
        
        # [关键对齐] LR Schedule: Cosine Decay
        if r < warmup_rounds:
            current_lr = initial_lr * (r + 1) / warmup_rounds
        else:
            progress = (r - warmup_rounds) / (total_rounds - warmup_rounds)
            current_lr = min_lr + 0.5 * (initial_lr - min_lr) * (1 + math.cos(progress * math.pi))
        
        print(f"Current LR: {current_lr:.2e}")

        client_updates = []
        client_sizes = []
        
        for cid, dl_info in enumerate(client_dataloaders):
            if dl_info is None: continue
            dl, collator = dl_info
            
            # Sync Global Model
            set_peft_model_state_dict(model, server.global_state_dict)
            
            print(f"Training Client {cid}...", flush=True)
            client = FedITClient(
                client_id=cid, 
                model=model, 
                dataloader=dl,
                local_epochs=cfg["train"]["local_epochs"], # 应该是 2 (A会推荐)
                lr=current_lr,
                device="cuda",
                dtype=torch_dtype,
                gradient_accumulation_steps=accum_steps # [关键] 传入累积步数
            )
            
            client.train()
            
            client_updates.append(client.get_parameters())
            client_sizes.append(len(dl.dataset))
            
            del client
            gc.collect()
            torch.cuda.empty_cache()
        
        print(f"Aggregating...", flush=True)
        new_state_dict = server.aggregate(client_updates, client_sizes)
        # Update Server State
        server.global_state_dict = new_state_dict
        
        # Save Checkpoint
        ckpt_path = os.path.join(cfg["output_dir"], f"global_round{r}.pt")
        torch.save(server.global_state_dict, ckpt_path)
        
        # Eval
        if cfg["eval"]["enabled"] and (r % cfg["eval"].get("eval_every", 1) == 0):
            print(f"Evaluating...", flush=True)
            # Sync model with aggregated weights before eval
            set_peft_model_state_dict(model, server.global_state_dict)
            model.eval()
            
            # 1. Global Eval (Optional, kept for consistency)
            # ... (代码省略，和原来一样) ...
            
            # 2. Task-Specific Evaluation
            # [关键对齐] 移除 >= 0.4 的判断，强制评估所有任务
            if True:
                print(f"Running Task-Specific Evaluation...", flush=True)
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
                        if "###" in pred: pred = pred.split("###")[0]
                        pred = pred.strip()
                        c_scores.append(compute_rouge1(pred, ex["output"]))
                    
                    avg_c_score = sum(c_scores) / len(c_scores) if c_scores else 0
                    client_scores[cid] = avg_c_score
                    task_name = CLIENT_TASK_MAPPING.get(cid, f"Client {cid}")
                    print(f"  {task_name}: {avg_c_score:.4f}")

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
                
                update_summary_csv(cfg["output_dir"], "FedIT", r, client_scores)

    print("\n>> Done.", flush=True)

if __name__ == "__main__":
    main()