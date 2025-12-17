
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
from models.llama_wrapper_dual_mix import FedDualMixModelWrapper
from federated.client_dual_mix import FedDualMixClient
from federated.server import FedSubspaceServer
from subspace.projection import RandomSubspaceProjection
from subspace.utils import unflatten_lora_params

# ============================================================
#                Helper Functions
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

def get_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        num_params = param.numel()
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel
        if param.__class__.__name__ == "Params4bit":
            num_params = num_params * 2
        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params
    return trainable_params, all_param

def compute_rouge1(prediction, reference):
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

# ============================================================
#                     Main Experiment
# ============================================================

def main():
    # Hardcoded config path
    cfg_path = "configs/fedsubspace_flan.yaml"
    cfg = yaml.safe_load(open(cfg_path,"r"))
    
    # Override output dir for this experiment
    cfg["output_dir"] = "outputs/fedless+dual+reg"
    os.makedirs(cfg["output_dir"], exist_ok=True)
    
    print(f">> [Main] Loading config from {cfg_path}...", flush=True)
    print(f">> [Main] Output Dir: {cfg['output_dir']}", flush=True)

    # Load Model
    print(f"Loading base model: {cfg['model']['path']}")
    torch_dtype = torch.float16
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        print(">> [Main] bfloat16 supported. Using bfloat16.", flush=True)
        torch_dtype = torch.bfloat16
    else:
        print(">> [Main] bfloat16 NOT supported. Using float16.", flush=True)

    base = AutoModelForCausalLM.from_pretrained(cfg["model"]["path"], torch_dtype=torch_dtype, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["path"])
    tokenizer.pad_token_id = tokenizer.eos_token_id or 0

    # Freeze Base
    for p in base.parameters(): p.requires_grad=False

    # Setup Dual Mix Model
    print(">> [Main] Initializing Dual Mix Model...", flush=True)
    lora_shapes = extract_lora_shapes(base, target_modules=cfg["lora"]["target_modules"], r=cfg["lora"]["r"])
    shared_model = FedDualMixModelWrapper(base, lora_shapes, d_s=cfg["subspace"]["dim"], seed=cfg["subspace"]["seed"],
                                          target_modules=cfg["lora"]["target_modules"])

    # Setup Server
    server = FedSubspaceServer(cfg["subspace"]["dim"])
    # Initialize server theta from model's global adapter (which is 0 initialized)
    server.global_theta = shared_model.adapter_global.theta_s.detach().cpu().clone()

    # --- Parameter Calculation ---
    trainable_params, all_param = get_trainable_parameters(shared_model)
    print(f"Trainable Params: {trainable_params:,}")
    print(f"All Params: {all_param:,}")
    print(f"Trainable Ratio: {100 * trainable_params / all_param:.4f}%")

    # --- CSV Logging Setup ---
    csv_file = os.path.join(cfg["output_dir"], "experiment_results.csv")
    if not os.path.exists(csv_file):
        with open(csv_file, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Timestamp", "BaseModel", "Dataset", "Round", "Metric", "Value", "TrainableParams", "AllParams", "TrainableRatio"])

    # Setup Clients DataLoaders
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
        if cfg["eval"]["max_samples"] and cfg["eval"]["max_samples"] > 0:
            global_eval_data = global_eval_data[:cfg["eval"]["max_samples"]]
        print(f"Loaded {len(global_eval_data)} global eval samples.")

    import math
    initial_lr = float(cfg["train"]["lr"])
    min_lr = 1e-6
    warmup_rounds = 1

    # --- Resume Logic ---
    start_round = 0
    checkpoint_pattern = re.compile(r"theta_global_round(\d+).pt")
    checkpoints = []
    if os.path.exists(cfg["output_dir"]):
        for f in os.listdir(cfg["output_dir"]):
            match = checkpoint_pattern.match(f)
            if match:
                checkpoints.append(int(match.group(1)))
    
    if checkpoints:
        last_round = max(checkpoints)
        checkpoint_path = os.path.join(cfg["output_dir"], f"theta_global_round{last_round}.pt")
        print(f">> Resuming from checkpoint: {checkpoint_path}")
        server.global_theta = torch.load(checkpoint_path, map_location="cpu")
        start_round = last_round + 1
        print(f">> Starting from Round {start_round}")

    for r in range(start_round, rounds):
        print(f"\n--- Round {r} ---")
        
        if r < warmup_rounds:
            current_lr = initial_lr * (r + 1) / warmup_rounds
        else:
            progress = (r - warmup_rounds) / (rounds - warmup_rounds)
            current_lr = min_lr + 0.5 * (initial_lr - min_lr) * (1 + math.cos(progress * math.pi))
        
        print(f"Current LR: {current_lr:.2e}")

        thetas_to_aggregate = []
        sizes = []
        
        # Train
        for cid, dl_info in enumerate(client_dataloaders):
            if dl_info is None: continue
            dl, collator = dl_info
            
            print(f"Training Client {cid}...")
            
            # Instantiate client
            # lambda_reg = 0.005 for FedLESS+dual+reg (Dual Stream, Learnable Alpha, L2 Reg)
            client = FedDualMixClient(client_id=cid, model=shared_model, tokenizer=tokenizer, dataloader=dl,
                                   output_dir=cfg["output_dir"], local_epochs=cfg["train"]["local_epochs"],
                                   lr=current_lr, device="cuda", data_collator=collator, dtype=torch_dtype,
                                   batch_size=cfg["train"]["batch_size"],
                                   gradient_accumulation_steps=cfg["train"].get("gradient_accumulation_steps", 1),
                                   lambda_reg=0.005)
            
            # 1. Load Global Theta
            client.load_theta_global(server.global_theta)
            
            # 2. Load Local Theta and Alphas (if exist)
            local_state_path = os.path.join(cfg["output_dir"], f"client_{cid}_local_state.pt")
            if os.path.exists(local_state_path):
                local_state = torch.load(local_state_path, map_location="cpu")
                client.load_theta_local(local_state['theta_local'])
                client.load_alphas(local_state['alphas'])
            else:
                # Initialize local theta to 0
                client.model.adapter_local.theta_s.data.zero_()
                for m in client.model.modules():
                    if hasattr(m, 'gate_g'):
                        m.gate_g.data.fill_(0.1) # Initialize to 0.1
                    if hasattr(m, 'gate_l'):
                        m.gate_l.data.fill_(0.1) # Initialize to 0.1

            # 3. Train
            client.train()
            
            # 4. Save Local State
            local_state = {
                'theta_local': client.get_theta_local(),
                'alphas': client.get_alphas()
            }
            torch.save(local_state, local_state_path)
            
            # 5. Collect Global Update
            thetas_to_aggregate.append(client.get_theta_global())
            sizes.append(len(client.dataloader.dataset))
            
            # Cleanup
            client.model = None
            del client
            gc.collect()
            torch.cuda.empty_cache()
        
        # Aggregate Global Theta
        new_global_theta = server.aggregate(thetas_to_aggregate, sizes)
        torch.save(new_global_theta, os.path.join(cfg["output_dir"], f"theta_global_round{r}.pt"))
        
        # Evaluate
        if cfg["eval"]["enabled"] and (r % cfg["eval"].get("eval_every", 1) == 0):
            print(f"Evaluating Round {r}...")
            
            # 1. Global Eval (Generalization)
            # Load Global Theta
            shared_model.adapter_global.theta_s.data.copy_(new_global_theta.to(base.device))
            # Disable Local (theta_l = 0) and Force Global (gate_g = 10, gate_l = 0)
            # Backup first
            # --- Global Evaluation (Skipped) ---
            # Backup first
            # backup_theta_l = shared_model.adapter_local.theta_s.detach().clone()
            # ... (Global Eval Code Removed) ...
            # Restore Backup (though we will overwrite in Personalized Eval loop)
            # shared_model.adapter_local.theta_s.data.copy_(backup_theta_l)
            # ...

            # 2. Personalized Eval
            # Always run personalized evaluation
            if True:
                print(f"Running Personalized Evaluation for Round {r}...")
                client_scores = {}
                
                for cid in range(cfg["data"]["num_clients"]):
                    test_path = os.path.join(cfg["data"]["root"], f"test_{cid}.json")
                    if not os.path.exists(test_path): continue
                    
                    with open(test_path, "r") as f:
                        test_data = json.load(f)
                    if cfg["eval"]["max_samples"]: test_data = test_data[:cfg["eval"]["max_samples"]]
                    
                    # Load Client State
                    local_state_path = os.path.join(cfg["output_dir"], f"client_{cid}_local_state.pt")
                    if not os.path.exists(local_state_path):
                        print(f"Warning: Local state for client {cid} not found. Skipping.")
                        continue
                    
                    local_state = torch.load(local_state_path, map_location=base.device)
                    
                    # Set Global (Already set)
                    # Set Local
                    shared_model.adapter_local.theta_s.data.copy_(local_state['theta_local'])
                    # Set Gates
                    for n, m in shared_model.named_modules():
                        if hasattr(m, 'gate_g') and f"{n}.gate_g" in local_state['alphas']:
                            m.gate_g.data.copy_(local_state['alphas'][f"{n}.gate_g"].to(base.device))
                        if hasattr(m, 'gate_l') and f"{n}.gate_l" in local_state['alphas']:
                            m.gate_l.data.copy_(local_state['alphas'][f"{n}.gate_l"].to(base.device))
                    
                    c_scores = []
                    for ex in tqdm(test_data, desc=f"Client {cid} Eval"):
                        raw_input = ex["input"]
                        prompt = f"### Instruction:\n{raw_input}\n\n### Response:\n"
                        inputs = tokenizer(prompt, return_tensors="pt").to(base.device)
                        with torch.no_grad():
                            outputs = base.generate(**inputs, max_new_tokens=50, pad_token_id=tokenizer.eos_token_id)
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
                        writer.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), os.path.basename(cfg["model"]["path"]), f"Flan ({task_name})", r, "ROUGE-1", avg_c_score, trainable_params, all_param, f"{100 * trainable_params / all_param:.4f}%"])
                
                update_summary_csv(cfg["output_dir"], "FedLESS+dual+reg", r, client_scores)

    print("Experiment Completed.")

if __name__ == "__main__":
    main()
