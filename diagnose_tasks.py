import os
import torch
import json
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer
from models.llama_wrapper import FedSubspaceModelWrapper
from models.lora_utils import extract_lora_shapes
import re
from subspace.projection import RandomSubspaceProjection
from subspace.utils import unflatten_lora_params

os.environ["CUDA_VISIBLE_DEVICES"] = "5"

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

# Config
cfg_path = "configs/fedsubspace_flan.yaml"
cfg = yaml.safe_load(open(cfg_path, "r"))
checkpoint_path = "outputs/fedsubspace_flan/theta_round19.pt"
clients_to_check = [3, 6] # Commonsense, Reading Comp

print(f"Loading base model: {cfg['model']['path']}")
# Force CPU to avoid OOM
torch_dtype = torch.float32
device = "cpu"

base = AutoModelForCausalLM.from_pretrained(cfg["model"]["path"], torch_dtype=torch_dtype, device_map=device)
tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["path"])
tokenizer.pad_token_id = tokenizer.eos_token_id or 0

# Setup Subspace
lora_shapes = extract_lora_shapes(base, target_modules=cfg["lora"]["target_modules"], r=cfg["lora"]["r"])

# Load Checkpoint
print(f"Loading checkpoint: {checkpoint_path}")
theta = torch.load(checkpoint_path, map_location="cpu")
adapter_state = decode_adapter(theta, lora_shapes, seed=cfg["subspace"]["seed"], device=device)

def compute_rouge1(prediction, reference):
    def get_tokens(text):
        return re.findall(r'\w+', text.lower())
    pred_tokens = get_tokens(prediction)
    ref_tokens = get_tokens(reference)
    if not pred_tokens or not ref_tokens: return 0.0
    from collections import Counter
    pred_counts = Counter(pred_tokens)
    ref_counts = Counter(ref_tokens)
    overlap = sum((pred_counts & ref_counts).values())
    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)
    if precision + recall == 0: return 0.0
    return 2 * (precision * recall) / (precision + recall)

# Evaluate
with AdapterInjector(base, adapter_state):
    for cid in clients_to_check:
        test_path = os.path.join(cfg["data"]["root"], f"test_{cid}.json")
        if not os.path.exists(test_path): continue
        
        with open(test_path, "r") as f:
            test_data = json.load(f)
        
        # Check first 3 samples
        print(f"\n=== Checking Client {cid} ===")
        scores = []
        for i, ex in enumerate(test_data[:3]):
            raw_input = ex["input"]
            prompt = f"### Instruction:\n{raw_input}\n\n### Response:\n"
            
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = base.generate(**inputs, max_new_tokens=20, pad_token_id=tokenizer.eos_token_id)
            
            input_len = inputs["input_ids"].shape[1]
            pred = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
            if "###" in pred:
                pred = pred.split("###")[0]
            pred = pred.strip()
            
            score = compute_rouge1(pred, ex["output"])
            scores.append(score)
            
            print(f"Sample {i}:")
            print(f"  Ref : {ex['output']}")
            print(f"  Pred: {pred}")
            print(f"  Score: {score:.4f}")
