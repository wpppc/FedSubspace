import os
import yaml
import torch
import json
import re
import csv
import time
import gc
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
try:
    import evaluate
except ImportError:
    evaluate = None

from data.dataset_tasks import GenericGenDataset, GenCollator
from models.lora_utils import extract_lora_shapes
from models.llama_wrapper import FedSubspaceModelWrapper
from federated.client import FedSubspaceClient
from federated.server import FedSubspaceServer
from subspace.projection import RandomSubspaceProjection
from subspace.utils import unflatten_lora_params

# ============================================================
#                Helper Functions
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

# Initialize ROUGE metric
rouge_metric = None
# Skip evaluate.load by default to avoid network hangs
print(">> [Init] Using local ROUGE-1 implementation (Counter-based) to avoid network hangs.", flush=True)

def compute_rouge1(prediction, reference):
    """
    Standard ROUGE-1 F1 score implementation using HuggingFace evaluate.
    """
    if rouge_metric:
        results = rouge_metric.compute(predictions=[prediction], references=[reference])
        return results['rouge1']
    else:
        # Fallback to simple implementation (less accurate but works without deps)
        # Improved to use Counter for bag-of-words instead of set
        from collections import Counter
        def get_unigrams(text):
            return re.findall(r'\w+', text.lower())
        
        pred_tokens = get_unigrams(prediction)
        ref_tokens = get_unigrams(reference)
        
        if not pred_tokens or not ref_tokens:
            return 0.0
            
        pred_counts = Counter(pred_tokens)
        ref_counts = Counter(ref_tokens)
        
        overlap = sum((pred_counts & ref_counts).values())
        precision = overlap / len(pred_tokens)
        recall = overlap / len(ref_tokens)
        
        if precision + recall == 0:
            return 0.0
        
        return 2 * (precision * recall) / (precision + recall)

def build_meta_from_shapes(lora_shapes):
    meta = []
    for name, (A_shape, B_shape) in lora_shapes.items():
        meta.append((f"{name}.A", list(A_shape), A_shape.numel()))
        meta.append((f"{name}.B", list(B_shape), B_shape.numel()))
    return meta

def decode_adapter(theta_s, lora_shapes, seed, device="cpu"):
    D = sum(A.numel() + B.numel() for A, B in lora_shapes.values())
    P = RandomSubspaceProjection(D, len(theta_s), seed=seed, device=device)
    theta_D = P.project(theta_s.to(device)).cpu()
    meta = build_meta_from_shapes(lora_shapes)
    return unflatten_lora_params(theta_D, meta)

class AdapterInjector:
    def __init__(self, base_model, adapter_state):
        self.base_model = base_model
        self.adapter_state = adapter_state
        self.applied = False

    def __enter__(self):
        self.apply_adapter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.remove_adapter()

    def apply_adapter(self):
        if self.applied: return
        params = {}
        for k, v in self.adapter_state.items():
            name = k[:-2]
            type_ = k[-1]
            if name not in params: params[name] = {}
            params[name][type_] = v
        
        modules_dict = dict(self.base_model.named_modules())
        for name, mats in params.items():
            if "A" in mats and "B" in mats and name in modules_dict:
                module = modules_dict[name]
                if isinstance(module, torch.nn.Linear):
                    A = mats["A"].to(module.weight.device, dtype=module.weight.dtype)
                    B = mats["B"].to(module.weight.device, dtype=module.weight.dtype)
                    delta = B @ A
                    module.weight.data += delta
        self.applied = True

    def remove_adapter(self):
        if not self.applied: return
        params = {}
        for k, v in self.adapter_state.items():
            name = k[:-2]; type_ = k[-1]
            if name not in params: params[name] = {}
            params[name][type_] = v
        
        modules_dict = dict(self.base_model.named_modules())
        for name, mats in params.items():
            if "A" in mats and "B" in mats and name in modules_dict:
                module = modules_dict[name]
                if isinstance(module, torch.nn.Linear):
                    A = mats["A"].to(module.weight.device, dtype=module.weight.dtype)
                    B = mats["B"].to(module.weight.device, dtype=module.weight.dtype)
                    delta = B @ A
                    module.weight.data -= delta
        self.applied = False

# ============================================================
#                     Main Experiment
# ============================================================

def main(cfg_path="configs/fedsubspace_flan.yaml"):
    cfg = yaml.safe_load(open(cfg_path,"r"))
    print(f">> [DEBUG] Config Loaded: Batch={cfg['train']['batch_size']}, Accum={cfg['train'].get('gradient_accumulation_steps', 'Default')}, LR={cfg['train']['lr']}")
    os.makedirs(cfg["output_dir"], exist_ok=True)

    # Load Model
    print(f"Loading base model: {cfg['model']['path']}")
    
    # Auto-select best dtype (bfloat16 is much more stable for LLaMA-2)
    torch_dtype = torch.float16
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        print(">> [Main] bfloat16 supported. Using bfloat16 for stability.", flush=True)
        torch_dtype = torch.bfloat16
    else:
        print(">> [Main] bfloat16 NOT supported. Using float16 (risk of NaN/Overflow).", flush=True)

    base = AutoModelForCausalLM.from_pretrained(cfg["model"]["path"], torch_dtype=torch_dtype, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["path"])
    tokenizer.pad_token_id = tokenizer.eos_token_id or 0

    # Freeze Base
    for p in base.parameters(): p.requires_grad=False

    # Setup Subspace
    lora_shapes = extract_lora_shapes(base, target_modules=cfg["lora"]["target_modules"], r=cfg["lora"]["r"])
    shared_model = FedSubspaceModelWrapper(base, lora_shapes, d_s=cfg["subspace"]["dim"], seed=cfg["subspace"]["seed"],
                                          target_modules=cfg["lora"]["target_modules"])

    # Setup Server
    server = FedSubspaceServer(cfg["subspace"]["dim"])
    # Initialize server theta from model's random initialization
    server.global_theta = shared_model.adapter.theta_s.detach().cpu().clone()

    # --- Parameter Calculation ---
    full_lora_params = sum(A.numel() + B.numel() for A, B in lora_shapes.values())
    subspace_params = cfg["subspace"]["dim"]
    
    trainable_params, all_param = get_trainable_parameters(shared_model)
    
    print(f"Full LoRA Params (Theoretical): {full_lora_params:,}")
    print(f"Subspace Params (Configured): {subspace_params:,}")
    print(f"Trainable Params (Actual): {trainable_params:,}")
    print(f"All Params: {all_param:,}")
    print(f"Trainable Ratio: {100 * trainable_params / all_param:.4f}%")

    # --- CSV Logging Setup ---
    csv_file = os.path.join(cfg["output_dir"], "experiment_results.csv")
    if not os.path.exists(csv_file):
        with open(csv_file, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Timestamp", "BaseModel", "Dataset", "Round", "Metric", "Value", "SubspaceParams", "FullLoRAParams", "TrainableParams", "AllParams", "TrainableRatio"])

    # Setup Clients DataLoaders (Keep data in memory, but create Client objects on demand)
    client_dataloaders = []
    for cid in range(cfg["data"]["num_clients"]):
        data_path = os.path.join(cfg["data"]["root"], f"client_{cid}.json")
        if not os.path.exists(data_path):
            print(f"Warning: Client data {data_path} not found. Skipping client {cid}.")
            client_dataloaders.append(None)
            continue
            
        ds = GenericGenDataset(data_path)
        collator = GenCollator(tokenizer, max_len=cfg["data"]["cutoff_len"], train_on_inputs=cfg["data"]["train_on_inputs"])
        dl = DataLoader(ds, batch_size=cfg["train"]["batch_size"], shuffle=True, collate_fn=lambda b: collator(b))
        client_dataloaders.append((dl, collator))
    
    print(f"Initialized {len([x for x in client_dataloaders if x is not None])} client dataloaders.")

    # Federated Training Loop
    rounds = cfg["federated"]["rounds"]
    
    # Load Global Eval Data
    global_eval_path = os.path.join(cfg["data"]["root"], "global_eval.json")
    global_eval_data = []
    if os.path.exists(global_eval_path):
        with open(global_eval_path, "r") as f:
            global_eval_data = json.load(f)
        # Limit samples
        if cfg["eval"]["max_samples"] and cfg["eval"]["max_samples"] > 0:
            global_eval_data = global_eval_data[:cfg["eval"]["max_samples"]]
        print(f"Loaded {len(global_eval_data)} global eval samples.")

    import math
    initial_lr = float(cfg["train"]["lr"])
    min_lr = 1e-6
    warmup_rounds = 1

    # Initialize global gates (for aggregation)
    global_gates = [p.detach().clone() for p in shared_model.get_gate_params()]

    # --- Resume Logic ---
    start_round = 0
    checkpoint_pattern = re.compile(r"checkpoint_round(\d+).pt")
    checkpoints = []
    if os.path.exists(cfg["output_dir"]):
        for f in os.listdir(cfg["output_dir"]):
            match = checkpoint_pattern.match(f)
            if match:
                checkpoints.append(int(match.group(1)))
    
    if checkpoints:
        last_round = max(checkpoints)
        checkpoint_path = os.path.join(cfg["output_dir"], f"checkpoint_round{last_round}.pt")
        print(f">> Resuming from checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        
        # Load Theta
        server.global_theta = checkpoint['theta']
        
        # Load Gates
        if 'gates' in checkpoint:
            global_gates = checkpoint['gates']
            print(f">> Loaded global gates from checkpoint.")
        
        start_round = last_round + 1
        print(f">> Starting from Round {start_round}")

    for r in range(start_round, rounds):
        print(f"\n--- Round {r} ---")
        
        # LR Scheduler: Cosine Decay with Warmup
        if r < warmup_rounds:
            # Linear Warmup
            current_lr = initial_lr * (r + 1) / warmup_rounds
        else:
            # Cosine Decay
            progress = (r - warmup_rounds) / (rounds - warmup_rounds)
            current_lr = min_lr + 0.5 * (initial_lr - min_lr) * (1 + math.cos(progress * math.pi))
        
        print(f"Current LR: {current_lr:.2e}")

        thetas = []
        sizes = []
        
        # Train
        for cid, dl_info in enumerate(client_dataloaders):
            if dl_info is None: continue
            dl, collator = dl_info
            
            print(f"Training Client {cid}...")
            
            # Instantiate client on the fly to ensure full cleanup after training
            client = FedSubspaceClient(client_id=cid, model=shared_model, tokenizer=tokenizer, dataloader=dl,
                                   output_dir=cfg["output_dir"], local_epochs=cfg["train"]["local_epochs"],
                                   lr=current_lr, device="cuda", data_collator=collator, dtype=torch_dtype,
                                   batch_size=cfg["train"]["batch_size"],
                                   gradient_accumulation_steps=cfg["train"].get("gradient_accumulation_steps", 1))
            
            # Load Global Theta
            client.load_theta(server.global_theta)
            
            # Load Global Gates (Fix for Shared State Bug)
            model_gates = shared_model.get_gate_params()
            for p_model, p_global in zip(model_gates, global_gates):
                p_model.data.copy_(p_global)
            
            client.train()
            
            # Collect Updates
            theta_update = client.get_theta()
            # Format gates as dict for server aggregation
            gate_update = {f"gate_{i}": p.detach().clone() for i, p in enumerate(shared_model.get_gate_params())}
            
            thetas.append({'theta': theta_update, 'gates': gate_update})
            sizes.append(len(client.dataloader.dataset))
            
            # Destroy client and force cleanup
            client.model = None
            del client
            gc.collect()
            torch.cuda.empty_cache()
        
        # Aggregate
        aggregated = server.aggregate(thetas, sizes)
        
        # Update Global State
        server.global_theta = aggregated['theta']
        new_theta = server.global_theta
        
        # Update Global Gates
        aggregated_gates = aggregated['gates']
        # Sort by index to ensure order
        sorted_keys = sorted(aggregated_gates.keys(), key=lambda x: int(x.split('_')[1]))
        global_gates = [aggregated_gates[k] for k in sorted_keys]
        
        torch.save(new_theta, os.path.join(cfg["output_dir"], f"theta_round{r}.pt"))
        
        # Save Checkpoint (Theta + Gates) for Resume
        checkpoint = {
            'theta': new_theta,
            'gates': global_gates
        }
        torch.save(checkpoint, os.path.join(cfg["output_dir"], f"checkpoint_round{r}.pt"))

        # Evaluate
        if cfg["eval"]["enabled"] and (r % cfg["eval"].get("eval_every", 1) == 0):
            print(f"Evaluating Round {r}...")
            
            # Load Global Theta into Shared Model
            shared_model.adapter.theta_s.data.copy_(new_theta.to(base.device))
            
            # Load Global Gates into Shared Model
            model_gates = shared_model.get_gate_params()
            for p_model, p_global in zip(model_gates, global_gates):
                p_model.data.copy_(p_global)
            
            # 1. Global Eval (Skipped as per request)
            # scores = []
            # print("Running Global Evaluation...")
            # ... (Global Eval Code Removed) ...
            # avg_rouge = sum(scores) / len(scores) if scores else 0
            # print(f"Round {r} Global ROUGE-1: {avg_rouge:.4f}")
            
            # --- Personalized Evaluation (Local Eval for FedSubspace) ---
            # Always run task-specific evaluation
            if True:
                print(f"Running Task-Specific Evaluation for Round {r}...")
                client_scores = {}
                
                for cid in range(cfg["data"]["num_clients"]):
                    test_path = os.path.join(cfg["data"]["root"], f"test_{cid}.json")
                    if not os.path.exists(test_path): continue
                    
                    with open(test_path, "r") as f:
                        test_data = json.load(f)
                    if cfg["eval"]["max_samples"]: test_data = test_data[:cfg["eval"]["max_samples"]]
                    
                    c_scores = []
                    for i, ex in enumerate(tqdm(test_data, desc=f"Client {cid} Eval")):
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
                        
                        # [DEBUG] Print first few predictions for each client
                        # if i < 2:
                        #     print(f"\n[DEBUG Client {cid}] Prompt: {prompt[:50]}...")
                        #     print(f"[DEBUG Client {cid}] Ref: {ex['output']}")
                        #     print(f"[DEBUG Client {cid}] Pred: {pred}")
                        #     print("-" * 20)

                        c_scores.append(compute_rouge1(pred, ex["output"]))
                    
                    avg_c_score = sum(c_scores) / len(c_scores) if c_scores else 0
                    client_scores[cid] = avg_c_score
                    
                    task_name = CLIENT_TASK_MAPPING.get(cid, f"Client {cid}")
                    print(f"  {task_name}: {avg_c_score:.4f}")

                    # Log individual client score
                    with open(csv_file, mode='a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([
                            time.strftime("%Y-%m-%d %H:%M:%S"),
                            os.path.basename(cfg["model"]["path"]),
                            f"Flan ({task_name})",
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
                update_summary_csv(cfg["output_dir"], "FedLESS", r, client_scores)
            
            # injector.remove_adapter()
            # else:
            #     print(f"Skipping Task-Specific Evaluation (Global ROUGE {avg_rouge:.4f} < 0.4)")
    
    print("Experiment Completed.")

if __name__ == "__main__":
    main()
