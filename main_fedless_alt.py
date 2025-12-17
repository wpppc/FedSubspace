
import os
import yaml
import torch
import json
import csv
import time
import gc
import re
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
try:
    import evaluate
except ImportError:
    evaluate = None

from data.dataset_tasks import GenericGenDataset, GenCollator
from models.lora_utils import extract_lora_shapes
from models.llama_wrapper_alt import FedDualSubspaceModelWrapper
from federated.client_alt import FedAltClient
from federated.server import FedSubspaceServer
from subspace.projection import RandomSubspaceProjection
from subspace.utils import unflatten_lora_params

# Reuse helper functions from main_flan.py (simplified here)
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

def main(cfg_path="configs/fedsubspace_flan.yaml"):
    # Load config but override output dir for ALT experiment
    cfg = yaml.safe_load(open(cfg_path,"r"))
    cfg["output_dir"] = "outputs/fedless+alt"
    os.makedirs(cfg["output_dir"], exist_ok=True)

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

    for p in base.parameters(): p.requires_grad=False

    # Setup Dual Subspace Model
    lora_shapes = extract_lora_shapes(base, target_modules=cfg["lora"]["target_modules"], r=cfg["lora"]["r"])
    shared_model = FedDualSubspaceModelWrapper(base, lora_shapes, d_s=cfg["subspace"]["dim"], seed=cfg["subspace"]["seed"],
                                          target_modules=cfg["lora"]["target_modules"])

    # Setup Server
    server = FedSubspaceServer(cfg["subspace"]["dim"])
    # Initialize global theta from model's random init
    # Also initialize global gates structure
    initial_gates = {}
    for name, module in shared_model.named_modules():
        if hasattr(module, "gate_g"):
            initial_gates[f"{name}.gate_g"] = module.gate_g.detach().cpu()
            initial_gates[f"{name}.gate_l"] = module.gate_l.detach().cpu()
            
    server.global_state = {
        'theta': shared_model.adapter_global.theta_s.detach().cpu().clone(),
        'gates': initial_gates
    }

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

    # Setup Clients DataLoaders
    client_dataloaders = []
    for cid in range(cfg["data"]["num_clients"]):
        data_path = os.path.join(cfg["data"]["root"], f"client_{cid}.json")
        if not os.path.exists(data_path): continue
            
        ds = GenericGenDataset(data_path)
        collator = GenCollator(tokenizer, max_len=cfg["data"]["cutoff_len"], train_on_inputs=cfg["data"]["train_on_inputs"])
        dl = DataLoader(ds, batch_size=cfg["train"]["batch_size"], shuffle=True, collate_fn=lambda b: collator(b))
        client_dataloaders.append((dl, collator))
    
    print(f"Initialized {len(client_dataloaders)} client dataloaders for FedLESS+ALT.")

    rounds = cfg["federated"]["rounds"]
    
    # Global Eval Data
    global_eval_path = os.path.join(cfg["data"]["root"], "global_eval.json")
    global_eval_data = []
    if os.path.exists(global_eval_path):
        with open(global_eval_path, "r") as f:
            global_eval_data = json.load(f)
        if cfg["eval"]["max_samples"]: global_eval_data = global_eval_data[:cfg["eval"]["max_samples"]]

    import math
    initial_lr = float(cfg["train"]["lr"])
    min_lr = 1e-6
    warmup_rounds = 1

    for r in range(rounds):
        print(f"\n--- Round {r} (FedLESS+ALT) ---")
        
        # LR Scheduler: Cosine Decay with Warmup
        if r < warmup_rounds:
            # Linear Warmup
            current_lr = initial_lr * (r + 1) / warmup_rounds
        else:
            # Cosine Decay
            progress = (r - warmup_rounds) / (rounds - warmup_rounds)
            current_lr = min_lr + 0.5 * (initial_lr - min_lr) * (1 + math.cos(progress * math.pi))
        
        print(f"Current LR: {current_lr:.2e}")

        thetas_to_aggregate = []
        sizes = []
        
        # Initialize previous_updates if it's the first round
        if not hasattr(server, 'previous_updates'):
            server.previous_updates = [None] * cfg["data"]["num_clients"]
        
        for cid, dl_info in enumerate(client_dataloaders):
            dl, collator = dl_info
            print(f"Training Client {cid}...")
            
            # Calculate RoW Global State for this client
            if r == 0:
                # Round 0: Use the unified global state
                client_global_state = server.global_state
            else:
                # Round > 0: 
                # We need: Client Global = (Server History) + (Average of Others' Residuals)
                
                # 1. Calculate Sum of Residuals from previous round
                sum_residuals = torch.zeros_like(server.previous_updates[0]['theta'])
                for u in server.previous_updates:
                    if u is not None:
                        sum_residuals += u['theta']
                
                # 2. Calculate RoW Residual (Exclude self)
                self_residual = server.previous_updates[cid]['theta']
                row_residual_avg = (sum_residuals - self_residual) / (cfg["data"]["num_clients"] - 1)
                
                # 3. Add to Server History
                row_theta = server.global_state['theta'] + row_residual_avg
                
                client_global_state = {
                    'theta': row_theta,
                    'gates': server.global_state['gates']
                }

            # Instantiate client on the fly
            client = FedAltClient(
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
                gradient_accumulation_steps=cfg["train"].get("gradient_accumulation_steps", 1)
            )
            
            # Client loads Global (Frozen) and Local (Trainable)
            # Pass the full global state (theta + gates)
            client.load_vectors(client_global_state)
            client.train()
            
            # Client sends Local Vector for aggregation
            update = client.get_update_for_server()
            thetas_to_aggregate.append(update)
            sizes.append(len(client.dataloader.dataset))

            # Explicit cleanup between clients
            client.model = None # Break reference
            del client
            gc.collect()
            torch.cuda.empty_cache()
        
        # Update server.previous_updates
        current_update_idx = 0
        for cid, dl_info in enumerate(client_dataloaders):
            if dl_info is not None:
                server.previous_updates[cid] = thetas_to_aggregate[current_update_idx]
                current_update_idx += 1

        # Server aggregates Local Vectors (Residuals)
        # We use update_state=False because we want to manually accumulate
        avg_update = server.aggregate(thetas_to_aggregate, sizes, update_state=False)
        
        # Accumulate Residuals into Global History
        # Global(t) = Global(t-1) + Avg(Residuals) * server_lr
        server_lr = 0.5
        new_global_theta = server.global_state['theta'] + (avg_update['theta'] * server_lr)
        
        # Update Server State
        server.global_state = {
            'theta': new_global_theta,
            'gates': avg_update['gates'] # Gates are averaged (replacement), not accumulated
        }
        
        new_global_state = server.global_state
        torch.save(new_global_state, os.path.join(cfg["output_dir"], f"theta_global_round{r}.pt"))
        
        # Evaluate (Using the new Global Vector as the "Global Model")
        if cfg["eval"]["enabled"] and (r % cfg["eval"].get("eval_every", 1) == 0):
            print(f"Evaluating Round {r}...")
            
            # 1. Load Global Theta
            shared_model.adapter_global.theta_s.data.copy_(new_global_state['theta'].to(base.device))
            
            # 2. Load Global Gates (Aggregated)
            if 'gates' in new_global_state:
                for name, val in new_global_state['gates'].items():
                    # Set gate_g to aggregated value
                    if name.endswith(".gate_g"):
                        # Find module
                        module_path = name.rsplit('.', 1)[0]
                        module = shared_model.get_submodule(module_path)
                        module.gate_g.data.copy_(val.to(base.device))
            
            # 3. Disable Local Branch for Evaluation
            # We set gate_l to 0.0 (tanh(0)=0) to completely shut off local branch
            # Backup first
            backup_gates_l = {}
            for name, module in shared_model.named_modules():
                if hasattr(module, "gate_l"):
                    backup_gates_l[name] = module.gate_l.data.clone()
                    module.gate_l.data.fill_(0.0)
            
            scores = []
            print("Running Global Evaluation...")
            for ex in tqdm(global_eval_data, desc="Global Eval"):
                raw_input = ex["input"]
                ref = ex["output"]
                prompt = f"### Instruction:\n{raw_input}\n\n### Response:\n"
                inputs = tokenizer(prompt, return_tensors="pt").to(base.device)
                
                with torch.no_grad():
                    outputs = base.generate(**inputs, max_new_tokens=50, pad_token_id=tokenizer.eos_token_id)
                
                input_len = inputs["input_ids"].shape[1]
                pred = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
                if "###" in pred:
                    pred = pred.split("###")[0]
                pred = pred.strip()
                scores.append(compute_rouge1(pred, ref))
            
            # 4. Restore Local Gates (though shared_model is reset next iteration anyway, good practice)
            for name, val in backup_gates_l.items():
                module = shared_model.get_submodule(name)
                module.gate_l.data.copy_(val)
            
            avg_rouge = sum(scores) / len(scores) if scores else 0
            print(f"Round {r} Global ROUGE-1: {avg_rouge:.4f}")
            
            # Save result
            with open(csv_file, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    time.strftime("%Y-%m-%d %H:%M:%S"),
                    os.path.basename(cfg["model"]["path"]),
                    "Flan (FedLESS+ALT)",
                    r,
                    "ROUGE-1",
                    avg_rouge,
                    subspace_params,
                    full_lora_params,
                    trainable_params,
                    all_param,
                    f"{100 * trainable_params / all_param:.4f}%"
                ])
            
            # --- Personalized Evaluation (Local Eval) ---
            if avg_rouge >= 0.4:
                print(f"Running Personalized Evaluation for Round {r}...")
                client_scores = {}
                local_scores = []
                
                # We need to iterate over clients again to evaluate on their specific test sets
                # Note: In a real scenario, clients would do this locally. Here we simulate it.
                for cid in range(cfg["data"]["num_clients"]):
                    test_path = os.path.join(cfg["data"]["root"], f"test_{cid}.json")
                    if not os.path.exists(test_path):
                        continue
                    
                    # Load test data
                    with open(test_path, "r") as f:
                        test_data = json.load(f)
                    if cfg["eval"]["max_samples"]: test_data = test_data[:cfg["eval"]["max_samples"]]
                    
                    # Load Client's Local State (Theta + Gates)
                    local_state_path = os.path.join(cfg["output_dir"], f"client_{cid}_local_state.pt")
                    if not os.path.exists(local_state_path):
                        print(f"Warning: Local state for client {cid} not found. Skipping.")
                        continue
                    
                    local_state = torch.load(local_state_path, map_location=base.device)
                    
                    # 1. Load Global State (Already loaded in shared_model from Global Eval step)
                    # shared_model.adapter_global.theta_s is already set to new_global_state['theta']
                    # Global gates are also set.
                    
                    # 2. Load Local State
                    shared_model.adapter_local.theta_s.data.copy_(local_state['theta'])
                    if 'gates' in local_state:
                        for name, val in local_state['gates'].items():
                            # Set gate_l (and gate_g if saved locally, but we use global gate_g)
                            if name.endswith(".gate_l"):
                                module_path = name.rsplit('.', 1)[0]
                                module = shared_model.get_submodule(module_path)
                                module.gate_l.data.copy_(val.to(base.device))
                    
                    # 3. Evaluate
                    c_scores = []
                    for ex in tqdm(test_data, desc=f"Client {cid} Eval"):
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
                    local_scores.append(avg_c_score)
                    print(f"  Client {cid} Local ROUGE-1: {avg_c_score:.4f}")

                    # Log individual client score
                    with open(csv_file, mode='a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([
                            time.strftime("%Y-%m-%d %H:%M:%S"),
                            os.path.basename(cfg["model"]["path"]),
                            f"Flan (FedLESS+ALT-Client {cid})",
                            r,
                            "ROUGE-1",
                            avg_c_score,
                            subspace_params,
                            full_lora_params,
                            trainable_params,
                            all_param,
                            f"{100 * trainable_params / all_param:.4f}%"
                        ])
                
                avg_personalized_rouge = sum(local_scores) / len(local_scores) if local_scores else 0
                print(f"Round {r} Personalized ROUGE-1: {avg_personalized_rouge:.4f}")
                
                # Save Personalized Result
                with open(csv_file, mode='a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        time.strftime("%Y-%m-%d %H:%M:%S"),
                        os.path.basename(cfg["model"]["path"]),
                        "Flan (FedLESS+ALT-Personalized)",
                        r,
                        "ROUGE-1",
                        avg_personalized_rouge,
                        subspace_params,
                        full_lora_params,
                        trainable_params,
                        all_param,
                        f"{100 * trainable_params / all_param:.4f}%"
                    ])
                
                # Update Summary Table CSV
                update_summary_csv(cfg["output_dir"], "FedLESS+ALT", r, client_scores)
            else:
                print(f"Skipping Personalized Evaluation (Global ROUGE {avg_rouge:.4f} < 0.4)")

    print("FedLESS+ALT Experiment Completed.")

if __name__ == "__main__":
    main()
