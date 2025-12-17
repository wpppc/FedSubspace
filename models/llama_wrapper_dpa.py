import torch
import torch.nn as nn
import math
from transformers import AutoModelForCausalLM

class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, r=8, alpha=16, dropout=0.05):
        super().__init__()
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        
        self.lora_A = nn.Parameter(torch.zeros(r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))
        self.dropout = nn.Dropout(p=dropout)
        
        # Init
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        # 混合精度计算，防止溢出
        x_f32 = x.to(torch.float32)
        A_f32 = self.lora_A.to(torch.float32)
        B_f32 = self.lora_B.to(torch.float32)
        
        out = (self.dropout(x_f32) @ A_f32.T @ B_f32.T) * self.scaling
        return out.to(x.dtype)

class DualLoRALinear(nn.Module):
    def __init__(self, original_module, r=8, alpha=16, dropout=0.05):
        super().__init__()
        self.original_module = original_module
        
        # Global Adapter (Server Aggregated)
        self.adapter_global = LoRALinear(original_module.in_features, original_module.out_features, r, alpha, dropout)
        
        # Local Adapter (Client Personalized)
        self.adapter_local = LoRALinear(original_module.in_features, original_module.out_features, r, alpha, dropout)
        
        # Mixing Weights (Fixed 0.5 for standard FedDPA)
        self.lam_g = 0.5
        self.lam_l = 0.5

    def forward(self, x):
        base_out = self.original_module(x)
        global_out = self.adapter_global(x)
        local_out = self.adapter_local(x)
        
        return base_out + self.lam_g * global_out + self.lam_l * local_out

class FedDPAModelWrapper(nn.Module):
    def __init__(self, base_model_path, lora_r=8, lora_alpha=16, lora_dropout=0.05, target_modules=None, torch_dtype=torch.float16):
        super().__init__()
        
        # [关键修复] 强制使用 eager 模式防止 RecursionError
        print(f">> [Model] Loading Base Model with eager attention...", flush=True)
        self.base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch_dtype,
            device_map="auto",
            attn_implementation="eager"
        )
        
        # Enable GC
        self.base_model.gradient_checkpointing_enable()
        self.base_model.enable_input_require_grads()
        
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.target_modules = target_modules or ["q_proj", "v_proj"]
        
        self.replace_layers()
        
        # Freeze Base
        for n, p in self.base_model.named_parameters():
            if "lora_" not in n:
                p.requires_grad = False

    def replace_layers(self):
        modules_to_replace = {}
        for name, module in self.base_model.named_modules():
            if any(t in name for t in self.target_modules) and isinstance(module, nn.Linear):
                modules_to_replace[name] = module
        
        for name, module in modules_to_replace.items():
            if '.' in name:
                parent_name, child_name = name.rsplit('.', 1)
                parent = self.base_model.get_submodule(parent_name)
            else:
                parent_name = ''; child_name = name; parent = self.base_model
            
            new_module = DualLoRALinear(module, self.lora_r, self.lora_alpha, self.lora_dropout)
            setattr(parent, child_name, new_module)

    def forward(self, input_ids, attention_mask=None, labels=None):
        return self.base_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

    # --- Parameter Helpers ---
    def get_global_state_dict(self):
        # Extract keys containing 'adapter_global'
        return {k: v.cpu() for k, v in self.state_dict().items() if "adapter_global" in k}

    def get_local_state_dict(self):
        # Extract keys containing 'adapter_local'
        return {k: v.cpu() for k, v in self.state_dict().items() if "adapter_local" in k}
    
    def load_global_state_dict(self, state_dict):
        self.load_state_dict(state_dict, strict=False)

    def load_local_state_dict(self, state_dict):
        self.load_state_dict(state_dict, strict=False)