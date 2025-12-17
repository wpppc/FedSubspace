
import torch
import torch.nn as nn
import torch.nn.functional as F
from subspace.adapter import SubspaceLoRAAdapter

class DualSubspaceLoRALinear(nn.Module):
    def __init__(self, original_module, layer_name, adapter_global, adapter_local):
        super().__init__()
        self.original_module = original_module
        self.layer_name = layer_name
        
        # Store adapters in lists to avoid auto-registration issues with Trainer
        self.adapter_global_ref = [adapter_global]
        self.adapter_local_ref = [adapter_local]
        
        # Dual Gates
        # gate_g: controls contribution of global knowledge (frozen reference)
        # gate_l: controls contribution of local knowledge (trainable expert)
        
        # Asymmetric Gating Strategy:
        # Global Gate: Fixed at 1.0 (Always On) to prevent catastrophic forgetting
        self.gate_g = nn.Parameter(torch.tensor(1.0), requires_grad=False)
        
        # Local Gate: Trainable.
        # FedAlt & Fed-LESS: Initialize to 0.0 (Strict Zero)
        # This ensures Round 0 starts from pure Base Model.
        self.gate_l = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        # Original output
        out = self.original_module(x)
        
        key = f"{self.layer_name}.lora"
        
        # --- Global Branch (Frozen) ---
        deltas_g = self.adapter_global_ref[0].get_layer_deltas(key)
        if deltas_g is not None:
            Ag, Bg = deltas_g
            # Ensure we don't backprop through global (double safety, though theta_g should be requires_grad=False)
            Ag, Bg = Ag.detach(), Bg.detach()
            
            x_f32 = x.to(torch.float32)
            Ag_f32 = Ag.to(torch.float32)
            Bg_f32 = Bg.to(torch.float32)
            
            lora_out_g = (x_f32 @ Ag_f32.T) @ Bg_f32.T
            # Global is always on (gate_g is fixed 1.0)
            out = out + self.gate_g * lora_out_g.to(out.dtype)

        # --- Local Branch (Trainable) ---
        deltas_l = self.adapter_local_ref[0].get_layer_deltas(key)
        if deltas_l is not None:
            Al, Bl = deltas_l
            
            x_f32 = x.to(torch.float32)
            Al_f32 = Al.to(torch.float32)
            Bl_f32 = Bl.to(torch.float32)
            
            lora_out_l = (x_f32 @ Al_f32.T) @ Bl_f32.T
            # Local starts at 0 and grows (tanh(gate_l))
            out = out + torch.tanh(self.gate_l) * lora_out_l.to(out.dtype)
        
        return out

class FedDualSubspaceModelWrapper(nn.Module):
    def __init__(self, base_model, lora_shapes, d_s, seed=42, target_modules=None):
        super().__init__()
        self.base_model = base_model
        self.target_modules = target_modules
        
        # Two adapters sharing the same projection seed (P)
        # This ensures they project to the same "semantic" LoRA space
        # Global Adapter: Initialize to Zeros (Identity) because gate_g is fixed at 1.0
        self.adapter_global = SubspaceLoRAAdapter(lora_shapes, d_s, seed, device=base_model.device, init_zeros=True)
        # Local Adapter: Initialize to Random (False)
        # We want to learn pure residuals from 0.
        self.adapter_local = SubspaceLoRAAdapter(lora_shapes, d_s, seed, device=base_model.device, init_zeros=False)
        
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
            
            new_module = DualSubspaceLoRALinear(module, name, self.adapter_global, self.adapter_local)
            setattr(parent, child_name, new_module)

    def forward(self, input_ids, attention_mask=None, labels=None):
        # Safety: Ensure Global is frozen, Local is trainable
        if self.adapter_global.theta_s.requires_grad:
            self.adapter_global.theta_s.requires_grad_(False)
        if not self.adapter_local.theta_s.requires_grad:
            self.adapter_local.theta_s.requires_grad_(True)

        outputs = self.base_model(input_ids=input_ids,
                                  attention_mask=attention_mask,
                                  labels=labels)
        return outputs
    
    def get_trainable_params(self):
        # Return theta_local and all gates
        # Filter only those that require grad
        params = []
        if self.adapter_local.theta_s.requires_grad:
            params.append(self.adapter_local.theta_s)
            
        for module in self.base_model.modules():
            if isinstance(module, DualSubspaceLoRALinear):
                if module.gate_g.requires_grad:
                    params.append(module.gate_g)
                if module.gate_l.requires_grad:
                    params.append(module.gate_l)
        return params
