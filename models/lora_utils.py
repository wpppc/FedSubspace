# models/lora_utils.py

import torch.nn as nn
import torch

def extract_lora_shapes(model, target_modules, r):
    """
    自动从 base model 中寻找 target_modules（如 q_proj/k_proj/...），
    并返回它们对应 LoRA A/B 的 shape。
    Example:
        q_proj: W ∈ R[out, in]
        LoRA:
            A: R[r, in]
            B: R[out, r]
    返回：{ "layer_name.lora_A": (A_shape, B_shape) }
    """
    shapes = {}
    for name, module in model.named_modules():
        if any(t in name for t in target_modules):
            if isinstance(module, nn.Linear):
                out_dim, in_dim = module.weight.shape

                A_shape = torch.Size([r, in_dim])
                B_shape = torch.Size([out_dim, r])

                shapes[f"{name}.lora"] = (A_shape, B_shape)
    return shapes
