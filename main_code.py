# main_code.py
import os, yaml, torch, json, csv, time
from copy import deepcopy
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.partition_tasks import partition_task
from data.format_fns import format_codefeedback
from data.dataset_tasks import GenericGenDataset, GenCollator

from models.lora_utils import extract_lora_shapes
from models.llama_wrapper import FedSubspaceModelWrapper
from federated.client import FedSubspaceClient
from federated.server import FedSubspaceServer

from subspace.projection import RandomSubspaceProjection
from subspace.utils import unflatten_lora_params

from evaluation.code_eval import eval_humaneval_with_adapter

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
    base = AutoModelForCausalLM.from_pretrained(cfg["model"]["path"], torch_dtype=torch.float16)
    base.to("cuda")
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
    print(f"Full LoRA Params: {full_lora_params:,}")
    print(f"Subspace Params (Communication): {subspace_params:,}")
    print(f"Compression Ratio: {compression_ratio:.2f}x")

    # --- CSV Logging Setup ---
    csv_file = os.path.join(cfg["output_dir"], "experiment_results.csv")
    if not os.path.exists(csv_file):
        with open(csv_file, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Timestamp", "Domain", "Round", "Task", "Metric", "Value", "SubspaceParams", "FullLoRAParams"])

    # Code Task Config
    task_cfg = {
        "name":"codefeedback",
        "hf_dataset": cfg["datasets"]["codefeedback"]["hf_name"],
        "train_split": cfg["datasets"]["codefeedback"].get("train_split","train"),
        "format_fn": format_codefeedback,
        "format_eval": False,
        "eval_sets": {
            "humaneval": (cfg["datasets"]["humaneval"]["hf_name"], cfg["datasets"]["humaneval"].get("split","test")),
            "mbpp": (cfg["datasets"]["mbpp"]["hf_name"], cfg["datasets"]["mbpp"].get("split","test"))
        }
    }

    # Partition
    out_root = os.path.join(cfg["data"]["root"], "codefeedback")
    eval_check_1 = os.path.join(out_root, "global_eval_humaneval.json")
    eval_check_2 = os.path.join(out_root, "global_eval_mbpp.json")
    if not os.path.exists(out_root) or not any(n.startswith("local_training_") for n in os.listdir(out_root)) or not os.path.exists(eval_check_1) or not os.path.exists(eval_check_2):
        print("Partitioning CodeFeedback...")
        partition_task(task_cfg, cfg["data"]["root"], cfg["data"]["num_clients"],
                       strategy=cfg["data"]["partition_strategy"],
                       alpha=cfg["data"]["partition_alpha"], seed=cfg["subspace"]["seed"])

    # Server
    server = FedSubspaceServer(cfg["subspace"]["dim"])
    
    # Clients
    clients = []
    task_dir = os.path.join(cfg["data"]["root"], "codefeedback")
    for cid in range(cfg["data"]["num_clients"]):
        path = os.path.join(task_dir, f"local_training_{cid}.json")
        ds = GenericGenDataset(path)
        collator = GenCollator(tokenizer, max_len=cfg["data"]["cutoff_len"], train_on_inputs=cfg["data"]["train_on_inputs"])
        dl = DataLoader(ds, batch_size=cfg["train"]["batch_size"], shuffle=True, collate_fn=lambda b: collator(b))
        client = FedSubspaceClient(client_id=cid, model=shared_model, tokenizer=tokenizer, dataloader=dl,
                                   output_dir=cfg["output_dir"], local_epochs=cfg["train"]["local_epochs"],
                                   lr=float(cfg["train"]["lr"]), device="cuda", data_collator=collator)
        clients.append(client)

    # Training Loop
    rounds = cfg["federated"]["rounds"]
    for r in range(rounds):
        print(f"--- Code Round {r} ---")
        thetas=[]; sizes=[]
        for cid in range(len(clients)):
            c = clients[cid]
            c.load_theta(server.global_theta)
            c.train()
            thetas.append(c.get_theta()); sizes.append(len(c.dataloader.dataset))
        
        new_theta = server.aggregate(thetas, sizes)
        torch.save(new_theta, os.path.join(cfg["output_dir"], f"code_theta_round{r}.pt"))

        # Eval
        if cfg["eval"]["enabled"] and (r % cfg["eval"].get("eval_every",1) == 0):
            print(f"Decoding adapter for Code round {r}...")
            adapter_state = decode_adapter(new_theta, lora_shapes, seed=cfg["subspace"]["seed"], device=base.device)
            
            with AdapterInjector(base, adapter_state):
                for eval_set in ["humaneval", "mbpp"]:
                    eval_file = os.path.join(task_dir, f"global_eval_{eval_set}.json")
                    if os.path.exists(eval_file):
                        with open(eval_file, 'r') as f: examples = json.load(f)
                        
                        prompts = []
                        for ex in examples:
                            if "prompt" in ex: # HumanEval
                                prompts.append(ex["prompt"])
                            elif "text" in ex: # MBPP
                                prompts.append(ex["text"])
                            elif "input" in ex:
                                prompts.append(ex["input"])
                            else:
                                prompts.append("") # Fallback
                        
                        prompts = prompts[:10]
                        res = eval_humaneval_with_adapter(adapter_state, base, tokenizer, prompts)
                        print(f"[Code] {eval_set} Generated {len(res)} samples.")
                        
                        # Log to CSV
                        with open(csv_file, mode='a', newline='') as f:
                            writer = csv.writer(f)
                            writer.writerow([
                                time.strftime("%Y-%m-%d %H:%M:%S"),
                                "Code",
                                r,
                                eval_set,
                                "GeneratedSamples",
                                len(res),
                                subspace_params,
                                full_lora_params
                            ])

if __name__=="__main__":
    main()
