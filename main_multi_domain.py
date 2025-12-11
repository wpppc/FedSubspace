# main_multi_domain.py
import os, yaml, torch, json
from copy import deepcopy
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.partition_tasks import partition_task
from data.format_fns import format_metamathqa, format_codefeedback, format_commonsense
from data.dataset_tasks import GenericGenDataset, GenCollator

from models.lora_utils import extract_lora_shapes
from models.llama_wrapper import FedSubspaceModelWrapper
from federated.client import FedSubspaceClient
from federated.server import FedSubspaceServer

from subspace.projection import RandomSubspaceProjection
from subspace.utils import unflatten_lora_params

# Import evaluation functions
from evaluation.math_eval import eval_gsm8k_with_adapter
from evaluation.code_eval import eval_humaneval_with_adapter
from evaluation.commonsense_eval import eval_boolq

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
    # compute D
    D = sum(A.numel() + B.numel() for A, B in lora_shapes.values())
    P = RandomSubspaceProjection(D, len(theta_s), seed=seed, device=device)
    theta_D = P.project(theta_s.to(device)).cpu()
    meta = build_meta_from_shapes(lora_shapes)
    return unflatten_lora_params(theta_D, meta)

class AdapterInjector:
    """
    Context manager to temporarily inject LoRA weights (W + BA) into the base model.
    """
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
        # Group by module
        params = {}
        for k, v in self.adapter_state.items():
            # k is like "model.layers.0.self_attn.q_proj.A"
            name = k[:-2] # remove .A or .B
            type_ = k[-1] # A or B
            if name not in params: params[name] = {}
            params[name][type_] = v
        
        # Apply
        modules_dict = dict(self.base_model.named_modules())
        for name, mats in params.items():
            if "A" in mats and "B" in mats and name in modules_dict:
                module = modules_dict[name]
                if isinstance(module, torch.nn.Linear):
                    A = mats["A"].to(module.weight.device, dtype=module.weight.dtype)
                    B = mats["B"].to(module.weight.device, dtype=module.weight.dtype)
                    # W_new = W + B @ A
                    delta = B @ A
                    module.weight.data += delta
        self.applied = True

    def remove_adapter(self):
        if not self.applied: return
        # Group by module
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
#                     Task Configs
# ============================================================

def build_task_cfgs(cfg):
    # return dict of task_cfgs for partition_tasks.partition_task
    root = cfg["data"]["root"]
    task_cfgs = {}

    # math
    task_cfgs["metamathqa"] = {
        "name":"metamathqa",
        "hf_dataset": cfg["datasets"]["metamathqa"]["hf_name"],
        "train_split": cfg["datasets"]["metamathqa"].get("train_split","train"),
        "format_fn": format_metamathqa,
        "eval_sets": {
            "gsm8k": (cfg["datasets"]["gsm8k"]["hf_name"], cfg["datasets"]["gsm8k"].get("split","test")),
            "math": (cfg["datasets"]["math"]["hf_name"], cfg["datasets"]["math"].get("split","test"))
        }
    }

    # code
    task_cfgs["codefeedback"] = {
        "name":"codefeedback",
        "hf_dataset": cfg["datasets"]["codefeedback"]["hf_name"],
        "train_split": cfg["datasets"]["codefeedback"].get("train_split","train"),
        "format_fn": format_codefeedback,
        "eval_sets": {
            "humaneval": (cfg["datasets"]["humaneval"]["hf_name"], cfg["datasets"]["humaneval"].get("split","test")),
            "mbpp": (cfg["datasets"]["mbpp"]["hf_name"], cfg["datasets"]["mbpp"].get("split","test"))
        }
    }

    # commonsense
    task_cfgs["commonsense170k"] = {
        "name":"commonsense170k",
        "hf_dataset": cfg["datasets"]["commonsense170k"]["hf_name"],
        "train_split": cfg["datasets"]["commonsense170k"].get("train_split","train"),
        "format_fn": format_commonsense,
        "eval_sets": {
            "boolq": (cfg["datasets"]["boolq"]["hf_name"], cfg["datasets"]["boolq"].get("split","validation")),
            # Add others as needed in config
        }
    }
    return task_cfgs

def main(cfg_path="configs/fedsubspace_multi_domain.yaml"):
    cfg = yaml.safe_load(open(cfg_path,"r"))
    os.makedirs(cfg["output_dir"], exist_ok=True)

    # load model & tokenizer
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"Loading base model: {cfg['model']['path']}")
    base = AutoModelForCausalLM.from_pretrained(cfg["model"]["path"], torch_dtype=torch.float16, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["path"])
    tokenizer.pad_token_id = tokenizer.eos_token_id or 0

    # freeze base
    for p in base.parameters(): p.requires_grad=False

    lora_shapes = extract_lora_shapes(base, target_modules=cfg["lora"]["target_modules"], r=cfg["lora"]["r"])
    shared_model = FedSubspaceModelWrapper(base, lora_shapes, d_s=cfg["subspace"]["dim"], seed=cfg["subspace"]["seed"],
                                          target_modules=cfg["lora"]["target_modules"])

    # partition each training dataset into num_clients splits (placed under data/root/<taskname>/)
    task_cfgs = build_task_cfgs(cfg)
    for taskname, tcfg in task_cfgs.items():
        out_root = os.path.join(cfg["data"]["root"], taskname)
        if not os.path.exists(out_root) or not any(n.startswith("local_training_") for n in os.listdir(out_root)):
            print(f"Partitioning {taskname}...")
            partition_task(tcfg, cfg["data"]["root"], cfg["data"]["num_clients"],
                           strategy=cfg["data"]["partition_strategy"],
                           alpha=cfg["data"]["partition_alpha"], seed=cfg["subspace"]["seed"])

    # For simplicity, we will simulate federated training per-domain sequentially:
    server = FedSubspaceServer(cfg["subspace"]["dim"])

    # Define domains and their eval sets mapping
    domains = [
        ("metamathqa", ["gsm8k", "math"]),
        ("codefeedback", ["humaneval", "mbpp"]),
        ("commonsense170k", ["boolq", "piqa", "siqa", "hellaswag", "winogrande", "arc_e", "arc_c", "obqa"])
    ]

    for domain, eval_datasets in domains:
        print(f"\n=== Running domain: {domain} ===")
        task_dir = os.path.join(cfg["data"]["root"], domain)
        
        # build client dataloaders
        clients = []
        for cid in range(cfg["data"]["num_clients"]):
            path = os.path.join(task_dir, f"local_training_{cid}.json")
            ds = GenericGenDataset(path)
            collator = GenCollator(tokenizer, max_len=cfg["data"]["cutoff_len"], train_on_inputs=cfg["data"]["train_on_inputs"])
            dl = DataLoader(ds, batch_size=cfg["train"]["batch_size"], shuffle=True, collate_fn=lambda b: collator(b))
            client = FedSubspaceClient(client_id=cid, model=shared_model, tokenizer=tokenizer, dataloader=dl,
                                       output_dir=cfg["output_dir"], local_epochs=cfg["train"]["local_epochs"],
                                       lr=cfg["train"]["lr"], device="cuda")
            clients.append(client)

        # federated rounds per domain
        rounds = cfg["federated"]["rounds"]
        for r in range(rounds):
            print(f"--- Domain {domain} Round {r} ---")
            thetas=[]; sizes=[]
            
            # Client Selection could be added here
            
            for cid in range(len(clients)):
                c = clients[cid]
                c.load_theta(server.global_theta)
                c.train()
                thetas.append(c.get_theta()); sizes.append(len(c.dataloader.dataset))
            
            new_theta = server.aggregate(thetas, sizes)
            torch.save(new_theta, os.path.join(cfg["output_dir"], f"{domain}_theta_round{r}.pt"))

            # ============================================================
            #            Decode -> Inject -> Evaluate
            # ============================================================
            if cfg["eval"]["enabled"] and (r % cfg["eval"].get("eval_every",1) == 0):
                print(f"Decoding adapter for {domain} round {r}...")
                adapter_state = decode_adapter(new_theta, lora_shapes, seed=cfg["subspace"]["seed"], device=base.device)
                
                # Inject adapter into base model temporarily
                with AdapterInjector(base, adapter_state):
                    print(f"Evaluating on {eval_datasets}...")
                    
                    for eval_set in eval_datasets:
                        eval_file = os.path.join(task_dir, f"global_eval_{eval_set}.json")
                        if not os.path.exists(eval_file):
                            print(f"Warning: Eval file {eval_file} not found. Skipping.")
                            continue
                            
                        try:
                            with open(eval_file, 'r') as f:
                                examples = json.load(f)
                        except Exception as e:
                            print(f"Error loading {eval_file}: {e}")
                            continue

                        # Dispatch to appropriate evaluator
                        if eval_set in ["gsm8k", "math"]:
                            # Use math evaluator
                            res = eval_gsm8k_with_adapter(adapter_state, base, tokenizer, examples)
                            print(f"[{domain}] {eval_set} Result: {res[:2]}...") # Print first 2 for check
                            
                        elif eval_set in ["humaneval", "mbpp"]:
                            # Use code evaluator
                            # Note: eval_humaneval_with_adapter expects prompts list
                            prompts = [ex["input"] for ex in examples] if isinstance(examples, list) else examples
                            res = eval_humaneval_with_adapter(adapter_state, base, tokenizer, prompts)
                            print(f"[{domain}] {eval_set} Result: {len(res)} samples generated.")
                            
                        elif eval_set in ["boolq", "piqa", "siqa", "hellaswag", "winogrande", "arc_e", "arc_c", "obqa"]:
                            # Use commonsense evaluator
                            # Note: eval_boolq is generic enough for accuracy on multiple choice if format matches
                            acc = eval_boolq(adapter_state, base, tokenizer, examples)
                            print(f"[{domain}] {eval_set} Accuracy: {acc:.4f}")
                        
                        else:
                            print(f"Unknown eval set: {eval_set}")

        print(f"Completed domain {domain}")

if __name__=="__main__":
    main()
