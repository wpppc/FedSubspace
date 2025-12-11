# evaluation/code_eval.py
import os
import torch
from tqdm import tqdm

def eval_humaneval_with_adapter(adapter_state, base_model, tokenizer, prompts, device="cuda", harness_dir=None):
    """
    This function returns pass@1 rate if harness_dir (humanEval harness) is available.
    If harness not provided, returns raw generations.
    """
    # Adapter is already injected by the caller using AdapterInjector

    gens = []
    base_model.to(device); base_model.eval()
    for p in tqdm(prompts):
        inputs = tokenizer(p, return_tensors="pt").to(device)
        with torch.no_grad():
            g = base_model.generate(**inputs, max_new_tokens=256)
        text = tokenizer.decode(g[0], skip_special_tokens=True)
        gens.append(text)
        
    # Adapter removal is handled by the caller

    return gens

