
import os
import sys
import torch
import json
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

# Add current directory to path to import local modules
sys.path.append(os.getcwd())

from models.llama_wrapper_alt import FedDualSubspaceModelWrapper
from data.dataset_tasks import GenericGenDataset

os.environ["CUDA_VISIBLE_DEVICES"] = "6"

def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

from models.lora_utils import extract_lora_shapes

def main():
    # 1. Configuration
    config_path = "configs/fedsubspace_flan.yaml"
    checkpoint_path = "outputs/fed_alt_ortho/theta_global_round7.pt"
    
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found: {checkpoint_path}")
        return

    cfg = load_config(config_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f">> Loading Base Model: {cfg['model']['path']}")
    # Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["path"], use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    # Load Base Model
    base_model = AutoModelForCausalLM.from_pretrained(
        cfg["model"]["path"],
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    # Extract LoRA Shapes
    print(">> Extracting LoRA Shapes...")
    lora_shapes = extract_lora_shapes(base_model, cfg["lora"]["target_modules"], cfg["lora"]["r"])
    
    # Initialize Wrapper
    print(">> Initializing FedDualSubspaceModelWrapper...")
    model = FedDualSubspaceModelWrapper(
        base_model,
        lora_shapes=lora_shapes,
        d_s=cfg["subspace"]["dim"],
        seed=cfg["subspace"]["seed"],
        target_modules=cfg["lora"]["target_modules"]
    )
    
    # Load Checkpoint
    print(f">> Loading Checkpoint: {checkpoint_path}")
    state = torch.load(checkpoint_path, map_location=device)
    
    # Load Global Theta
    if isinstance(state, dict):
        theta = state['theta']
    else:
        theta = state
        
    # For diagnosis, we only care about the Global Model performance
    # So we load theta into adapter_global and use it.
    # Note: In FedALT, the global model uses adapter_global.
    model.adapter_global.theta_s.data.copy_(theta.to(device))
    
    # Disable Local Adapter for this test to see pure Global Model performance
    # Or we can set gate_l to 0.
    for module in model.modules():
        if hasattr(module, "gate_l"):
            module.gate_l.data.fill_(0.0)
            
    model.eval()
    
    # 2. Define Tasks to Diagnose
    # Client 5: Struct to Text (Generation)
    # Client 3: Commonsense Reasoning (Selection/Generation)
    # Client 6: Reading Comprehension (Selection/Generation)
    
    tasks_to_check = [
        (5, "Struct to Text (CommonGen)"),
        (3, "Commonsense Reasoning (Hellaswag)"),
        (6, "Reading Comprehension (OpenBookQA)")
    ]
    
    for cid, task_name in tasks_to_check:
        print(f"\n{'='*40}")
        print(f"Diagnosing Task: {task_name} (Client {cid})")
        print(f"{'='*40}")
        
        data_path = os.path.join(cfg["data"]["root"], f"client_{cid}.json")
        if not os.path.exists(data_path):
            print(f"Data file not found: {data_path}")
            continue
            
        with open(data_path, "r") as f:
            data = json.load(f)
            
        # Take a few random samples
        import random
        random.seed(42)
        samples = random.sample(data, 5)
        
        for i, ex in enumerate(samples):
            input_text = ex["input"]
            target_text = ex["output"]
            
            prompt = f"### Instruction:\n{input_text}\n\n### Response:\n"
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            
            with torch.no_grad():
                outputs = base_model.generate(
                    **inputs, 
                    max_new_tokens=100, 
                    pad_token_id=tokenizer.eos_token_id,
                    do_sample=False, # Greedy decoding for deterministic output
                    num_beams=1
                )
            
            input_len = inputs["input_ids"].shape[1]
            pred = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
            
            # Clean up prediction
            if "###" in pred:
                pred = pred.split("###")[0]
            pred = pred.strip()
            
            print(f"\n[Sample {i+1}]")
            print(f"Input:   {input_text}")
            print(f"Target:  {target_text}")
            print(f"Predict: {pred}")
            print("-" * 20)

if __name__ == "__main__":
    main()
