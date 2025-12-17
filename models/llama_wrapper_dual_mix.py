
import torch
import torch.nn as nn
import torch.nn.functional as F
from subspace.adapter import SubspaceLoRAAdapter

class DualMixLoRALinear(nn.Module):
    def __init__(self, original_module, layer_name, adapter_g, adapter_l):
        super().__init__()
        self.original_module = original_module
        self.layer_name = layer_name
        # Store adapters in lists to avoid registering them as submodules
        self.adapter_g_ref = [adapter_g]
        self.adapter_l_ref = [adapter_l]
        
        # Dual Gates for Independent Control
        # Initialize to 0.1 to avoid deadlock and allow gradients to flow to both adapters.
        self.gate_g = nn.Parameter(torch.tensor(0.1))
        self.gate_l = nn.Parameter(torch.tensor(0.1))

    def forward(self, x):
        # Original output
        out = self.original_module(x)
        
        # Compute deltas for Global
        key = f"{self.layer_name}.lora"
        deltas_g = self.adapter_g_ref[0].get_layer_deltas(key)
        
        # Compute deltas for Local
        deltas_l = self.adapter_l_ref[0].get_layer_deltas(key)
        
        if deltas_g is not None and deltas_l is not None:
            # Global LoRA output
            A_g, B_g = deltas_g
            x_f32 = x.to(torch.float32)
            A_g_f32 = A_g.to(torch.float32)
            B_g_f32 = B_g.to(torch.float32)
            lora_out_g = (x_f32 @ A_g_f32.T) @ B_g_f32.T
            
            # Local LoRA output
            A_l, B_l = deltas_l
            A_l_f32 = A_l.to(torch.float32)
            B_l_f32 = B_l.to(torch.float32)
            lora_out_l = (x_f32 @ A_l_f32.T) @ B_l_f32.T
            
            # Gated Combination
            # out = base + tanh(gate_g) * global + tanh(gate_l) * local
            out = out + torch.tanh(self.gate_g) * lora_out_g.to(out.dtype) + torch.tanh(self.gate_l) * lora_out_l.to(out.dtype)
            
        return out

class FedDualMixModelWrapper(nn.Module):
    def __init__(self, base_model, lora_shapes, d_s, seed=42, target_modules=None):
        super().__init__()
        self.base_model = base_model
        self.target_modules = target_modules
        
        # Initialize Global and Local Adapters RANDOMLY (init_zeros=False)
        # This ensures gradients exist. Gates will handle the zero-start.
        self.adapter_global = SubspaceLoRAAdapter(lora_shapes, d_s, seed, device=base_model.device, init_zeros=False)
        self.adapter_local = SubspaceLoRAAdapter(lora_shapes, d_s, seed, device=base_model.device, init_zeros=False)
        
        self.replace_layers()

    def replace_layers(self):
        modules_to_replace = {}
        for name, module in self.base_model.named_modules():
            if any(t in name for t in self.target_modules) and isinstance(module, nn.Linear):
                modules_to_replace[name] = module
        
        for name, module in modules_to_replace.items():
            # Find parent
            if '.' in name:
                parent_name, child_name = name.rsplit('.', 1)
                parent = self.base_model.get_submodule(parent_name)
            else:
                parent_name = ''
                child_name = name
                parent = self.base_model
            
            new_module = DualMixLoRALinear(module, name, self.adapter_global, self.adapter_local)
            setattr(parent, child_name, new_module)

    def forward(self, input_ids, attention_mask=None, labels=None):
        # Ensure thetas require grad
        if not self.adapter_global.theta_s.requires_grad:
             self.adapter_global.theta_s.requires_grad_(True)
        if not self.adapter_local.theta_s.requires_grad:
             self.adapter_local.theta_s.requires_grad_(True)

        # Run base model
        outputs = self.base_model(input_ids=input_ids,
                                  attention_mask=attention_mask,
                                  labels=labels)

        return outputs
    
    def get_gate_params(self):
        gates = []
        for module in self.base_model.modules():
            if isinstance(module, DualMixLoRALinear):
                gates.append(module.gate_g)
                gates.append(module.gate_l)
        return gates
