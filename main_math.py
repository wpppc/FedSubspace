# main_math.py
import os, yaml, torch, json, csv, time, gc, random
from copy import deepcopy
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.partition_tasks import partition_task
from data.format_fns import format_metamathqa
from data.dataset_tasks import GenericGenDataset, GenCollator

from models.lora_utils import extract_lora_shapes
from models.llama_wrapper import FedSubspaceModelWrapper, SubspaceLoRALinear
from federated.client import FedSubspaceClient
from federated.server import FedSubspaceServer

from subspace.projection import RandomSubspaceProjection
from subspace.utils import unflatten_lora_params

from evaluation.math_eval import eval_gsm8k_with_adapter

# ============================================================
#                Parameter Counting Helper
# ============================================================

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

# ============================================================
#                Decode & Inject Helpers
# ============================================================

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
            name = k[:-2]; type_ = k[-1]
            if name not in params: params[name] = {}
            params[name][type_] = v
        
        modules_dict = dict(self.base_model.named_modules())
        for name, mats in params.items():
            if "A" in mats and "B" in mats and name in modules_dict:
                module = modules_dict[name]
                
                # Handle SubspaceLoRALinear
                if isinstance(module, SubspaceLoRALinear):
                    module = module.original_module
                
                if isinstance(module, torch.nn.Linear):
                    A = mats["A"].to(module.weight.device, dtype=module.weight.dtype)
                    B = mats["B"].to(module.weight.device, dtype=module.weight.dtype)
                    module.weight.data += B @ A
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
                
                # Handle SubspaceLoRALinear
                if isinstance(module, SubspaceLoRALinear):
                    module = module.original_module
                
                if isinstance(module, torch.nn.Linear):
                    A = mats["A"].to(module.weight.device, dtype=module.weight.dtype)
                    B = mats["B"].to(module.weight.device, dtype=module.weight.dtype)
                    module.weight.data -= B @ A
        self.applied = False

def main(cfg_path="configs/fedsubspace_multi_domain.yaml"):
    cfg = yaml.safe_load(open(cfg_path,"r"))
    os.makedirs(cfg["output_dir"], exist_ok=True)

    # Load Model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"Loading base model: {cfg['model']['path']}")
    # device_map="auto" can cause issues with Trainer moving models. 
    # Since we use single GPU per script (CUDA_VISIBLE_DEVICES), load directly to cuda.
    base = AutoModelForCausalLM.from_pretrained(cfg["model"]["path"], torch_dtype=torch.float16)
    base.to("cuda")
    # Enable gradient checkpointing to save memory
    base.gradient_checkpointing_enable()
    base.enable_input_require_grads()
    
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["path"])
    tokenizer.pad_token_id = tokenizer.eos_token_id or 0
    for p in base.parameters(): p.requires_grad=False

    lora_shapes = extract_lora_shapes(base, target_modules=cfg["lora"]["target_modules"], r=cfg["lora"]["r"])
    shared_model = FedSubspaceModelWrapper(base, lora_shapes, d_s=cfg["subspace"]["dim"], seed=cfg["subspace"]["seed"],
                                          target_modules=cfg["lora"]["target_modules"])

    # --- Parameter Calculation ---
    full_lora_params = sum(A.numel() + B.numel() for A, B in lora_shapes.values())
    subspace_params = cfg["subspace"]["dim"]
    compression_ratio = full_lora_params / subspace_params
    
    trainable_params, all_param = get_trainable_parameters(shared_model)
    
    print(f"Full LoRA Params (Theoretical): {full_lora_params:,}")
    print(f"Subspace Params (Configured): {subspace_params:,}")
    print(f"Trainable Params (Actual): {trainable_params:,}")
    print(f"All Params: {all_param:,}")
    print(f"Trainable Ratio: {100 * trainable_params / all_param:.4f}%")
    print(f"Compression Ratio: {compression_ratio:.2f}x")

    # --- CSV Logging Setup ---
    csv_file = os.path.join(cfg["output_dir"], "experiment_results.csv")
    if not os.path.exists(csv_file):
        with open(csv_file, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Timestamp", "BaseModel", "PartitionStrategy", "Domain", "Round", "Task", "Metric", "Value", "SubspaceParams", "FullLoRAParams", "TrainableParams", "AllParams", "TrainableRatio"])

    # Math Task Config
    task_cfg = {
        "name":"metamathqa",
        "hf_dataset": cfg["datasets"]["metamathqa"]["hf_name"],
        "train_split": cfg["datasets"]["metamathqa"].get("train_split","train"),
        "format_fn": format_metamathqa,
        "format_eval": False,
        "eval_sets": {
            "gsm8k": (cfg["datasets"]["gsm8k"]["hf_name"], cfg["datasets"]["gsm8k"].get("split","test"), "main"),
            "math": (cfg["datasets"]["math"]["hf_name"], cfg["datasets"]["math"].get("split","test"))
        }
    }

    # Partition
    out_root = os.path.join(cfg["data"]["root"], "metamathqa")
    eval_file_check_1 = os.path.join(out_root, "global_eval_gsm8k.json")
    eval_file_check_2 = os.path.join(out_root, "global_eval_math.json")
    
    if not os.path.exists(out_root) or not any(n.startswith("local_training_") for n in os.listdir(out_root)) or not os.path.exists(eval_file_check_1) or not os.path.exists(eval_file_check_2):
        print("Partitioning MetaMathQA...")
        partition_task(task_cfg, cfg["data"]["root"], cfg["data"]["num_clients"],
                       strategy=cfg["data"]["partition_strategy"],
                       alpha=cfg["data"]["partition_alpha"], seed=cfg["subspace"]["seed"])

    # Server
    server = FedSubspaceServer(cfg["subspace"]["dim"])
    # Initialize server with the model's random initialization to avoid starting at zero (saddle point)
    server.global_theta = shared_model.adapter.theta_s.detach().cpu().clone()
    
    # Clients
    clients = []
    task_dir = os.path.join(cfg["data"]["root"], "metamathqa")
    for cid in range(cfg["data"]["num_clients"]):
        path = os.path.join(task_dir, f"local_training_{cid}.json")
        ds = GenericGenDataset(path)
        collator = GenCollator(tokenizer, max_len=cfg["data"]["cutoff_len"], train_on_inputs=cfg["data"]["train_on_inputs"])
        dl = DataLoader(ds, batch_size=cfg["train"]["batch_size"], shuffle=True, collate_fn=lambda b: collator(b))
        client = FedSubspaceClient(client_id=cid, model=shared_model, tokenizer=tokenizer, dataloader=dl,
                                   output_dir=cfg["output_dir"], local_epochs=cfg["train"]["local_epochs"],
                                   max_steps=cfg["train"].get("max_steps", -1),
                                   lr=float(cfg["train"]["lr"]), device="cuda", data_collator=collator,
                                   gradient_accumulation_steps=cfg["train"].get("gradient_accumulation_steps", 1),
                                   lr_scheduler_type=cfg["train"].get("lr_scheduler_type", "linear"),
                                   warmup_ratio=cfg["train"].get("warmup_ratio", 0.0))
        clients.append(client)

    # Training Loop
    rounds = cfg["federated"]["rounds"]
    num_selected = max(1, int(cfg["data"]["num_clients"] * cfg["federated"]["client_fraction"]))

    for r in range(rounds):
        print(f"--- Math Round {r} ---")
        
        # Randomly select clients, but seeded for reproducibility across experiments
        random.seed(cfg["subspace"]["seed"] + r)
        selected_cids = random.sample(range(cfg["data"]["num_clients"]), num_selected)
        selected_cids.sort()
        print(f"Selected Clients: {selected_cids}")

        thetas=[]; sizes=[]
        for cid in selected_cids:
            c = clients[cid]
            c.load_theta(server.global_theta)
            c.train()
            thetas.append(c.get_theta()); sizes.append(len(c.dataloader.dataset))
            
            # Explicit cleanup between clients
            gc.collect()
            torch.cuda.empty_cache()
        
        new_theta = server.aggregate(thetas, sizes)
        torch.save(new_theta, os.path.join(cfg["output_dir"], f"math_theta_round{r}.pt"))

        # Eval
        if cfg["eval"]["enabled"] and (r % cfg["eval"].get("eval_every",1) == 0):
            print(f"Decoding adapter for Math round {r}...")
            adapter_state = decode_adapter(new_theta, lora_shapes, seed=cfg["subspace"]["seed"], device=base.device)
            
            with AdapterInjector(base, adapter_state):
                for eval_set in ["gsm8k", "math"]:
                    eval_file = os.path.join(task_dir, f"global_eval_{eval_set}.json")
                    if os.path.exists(eval_file):
                        with open(eval_file, 'r') as f: raw_examples = json.load(f)
                        
                        # Normalize to input/output for eval function
                        examples = []
                        for ex in raw_examples:
                            if "question" in ex and "answer" in ex: # GSM8K
                                examples.append({"input": ex["question"], "output": ex["answer"]})
                            elif "problem" in ex and "solution" in ex: # MATH
                                examples.append({"input": ex["problem"], "output": ex["solution"]})
                            elif "query" in ex and "response" in ex: # MetaMathQA style
                                examples.append({"input": ex["query"], "output": ex["response"]})
                            else:
                                # Fallback or skip
                                pass
                                
                        if examples:
                            # Limit examples for quick verification
                            max_samples = cfg["eval"].get("max_samples", None)
                            if max_samples and max_samples > 0 and len(examples) > max_samples:
                                examples = examples[:max_samples]
                                
                            print(f"Evaluating on {len(examples)} examples...")
                            acc, res_str = eval_gsm8k_with_adapter(adapter_state, base, tokenizer, examples)
                            print(f"[Math] {eval_set} Result: {res_str}")
                            
                            # Log to CSV
                            with open(csv_file, mode='a', newline='') as f:
                                writer = csv.writer(f)
                                writer.writerow([
                                    time.strftime("%Y-%m-%d %H:%M:%S"),
                                    os.path.basename(cfg["model"]["path"]),
                                    cfg["data"]["partition_strategy"],
                                    "Math",
                                    r,
                                    eval_set,
                                    "Accuracy",
                                    acc,
                                    subspace_params,
                                    full_lora_params,
                                    trainable_params,
                                    all_param,
                                    f"{100 * trainable_params / all_param:.4f}%"
                                ])

if __name__=="__main__":
    main()
