
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

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
        # Ensure LoRA weights are on the same device and dtype as input
        device = x.device
        dtype = x.dtype
        
        if self.lora_A.device != device:
            self.lora_A.data = self.lora_A.data.to(device)
            self.lora_B.data = self.lora_B.data.to(device)
            
        # Cast to float32 for stability, similar to standard LoRA implementations
        x_f32 = x.to(torch.float32)
        A_f32 = self.lora_A.to(torch.float32)
        B_f32 = self.lora_B.to(torch.float32)
        
        out = (self.dropout(x_f32) @ A_f32.T @ B_f32.T) * self.scaling
        return out.to(dtype)

class DualLoRALinear(nn.Module):
    def __init__(self, original_module, r=8, alpha=16, dropout=0.05):
        super().__init__()
        self.original_module = original_module
        self.in_features = original_module.in_features
        self.out_features = original_module.out_features
        
        # Global Adapter (Aggregated)
        self.adapter_global = LoRALinear(self.in_features, self.out_features, r, alpha, dropout)
        
        # Local Adapter (Personalized)
        self.adapter_local = LoRALinear(self.in_features, self.out_features, r, alpha, dropout)
        
        # Mixing Weight (Fixed 0.5 as per FedDPA default)
        self.alpha_g = 0.5
        self.alpha_l = 0.5

    def forward(self, x):
        # Base output
        out = self.original_module(x)
        
        # Global LoRA
        out_g = self.adapter_global(x)
        
        # Local LoRA
        out_l = self.adapter_local(x)
        
        # Mix
        out = out + (self.alpha_g * out_g + self.alpha_l * out_l).to(out.dtype)
        
        return out

class FedDPAModelWrapper(nn.Module):
    def __init__(self, base_model, lora_r=8, lora_alpha=16, lora_dropout=0.05, target_modules=None):
        super().__init__()
        self.base_model = base_model
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.target_modules = target_modules or ["q_proj", "v_proj"]
        
        self.replace_layers()

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
                parent_name = ''
                child_name = name
                parent = self.base_model
            
            new_module = DualLoRALinear(module, self.lora_r, self.lora_alpha, self.lora_dropout)
            setattr(parent, child_name, new_module)

    def forward(self, input_ids, attention_mask=None, labels=None):
        return self.base_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    
    def get_global_params(self):
        params = []
        for module in self.base_model.modules():
            if isinstance(module, DualLoRALinear):
                params.append(module.adapter_global.lora_A)
                params.append(module.adapter_global.lora_B)
        return params

    def get_local_params(self):
        params = []
        for module in self.base_model.modules():
            if isinstance(module, DualLoRALinear):
                params.append(module.adapter_local.lora_A)
                params.append(module.adapter_local.lora_B)
        return params
    
    def load_global_state_dict(self, state_dict):
        # Load only global adapters
        # state_dict keys should match the structure
        # We assume state_dict is the full model state dict or just the adapters?
        # For simplicity, let's assume we pass a dict of {layer_name: {'A': ..., 'B': ...}}
        # Or better, just use load_state_dict with strict=False if keys match.
        
        # But here we want to load specifically into adapter_global
        # Let's assume state_dict contains keys like "base_model.model.layers.0.self_attn.q_proj.adapter_global.lora_A"
        self.load_state_dict(state_dict, strict=False)

    def get_global_state_dict(self):
        return {k: v for k, v in self.state_dict().items() if "adapter_global" in k}

    def get_local_state_dict(self):
        return {k: v for k, v in self.state_dict().items() if "adapter_local" in k}
