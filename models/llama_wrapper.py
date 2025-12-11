# models/llama_wrapper.py

import torch
import torch.nn as nn
import torch.nn.functional as F

from subspace.adapter import SubspaceLoRAAdapter

class SubspaceLoRALinear(nn.Module):
    def __init__(self, original_module, layer_name, adapter):
        super().__init__()
        self.original_module = original_module
        self.layer_name = layer_name
        # Store adapter in a list to avoid registering it as a submodule
        # This prevents Trainer from trying to move it (and failing with meta tensor error)
        # while still allowing access to it for forward pass.
        self.adapter_ref = [adapter]
        
        # Adaptive Zero-Initialization Gate
        # Initialize to 0.0 to start from pure Base Model.
        self.gate = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        # Original output
        out = self.original_module(x)
        
        # LoRA output
        # Compute deltas on the fly to support gradient checkpointing
        key = f"{self.layer_name}.lora"
        # Access adapter via the list
        deltas = self.adapter_ref[0].get_layer_deltas(key)
        
        if deltas is not None:
            A, B = deltas
            
            # Compute in float32 for numerical stability
            x_f32 = x.to(torch.float32)
            A_f32 = A.to(torch.float32)
            B_f32 = B.to(torch.float32)
            
            lora_out = (x_f32 @ A_f32.T) @ B_f32.T
            
            # Apply gate
            
            # Apply gate
            out = out + torch.tanh(self.gate) * lora_out.to(out.dtype)
        
        return out

class FedSubspaceModelWrapper(nn.Module):
    def __init__(self, base_model, lora_shapes, d_s, seed=42, target_modules=None):
        super().__init__()
        self.base_model = base_model
        self.target_modules = target_modules
        self.adapter = SubspaceLoRAAdapter(lora_shapes, d_s, seed, device=base_model.device)
        
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
            
            new_module = SubspaceLoRALinear(module, name, self.adapter)
            setattr(parent, child_name, new_module)

    def forward(self, input_ids, attention_mask=None, labels=None):
        # Ensure theta_s requires grad (safety check)
        if not self.adapter.theta_s.requires_grad:
             self.adapter.theta_s.requires_grad_(True)

        # Run base model (which now uses SubspaceLoRALinear layers)
        outputs = self.base_model(input_ids=input_ids,
                                  attention_mask=attention_mask,
                                  labels=labels)

        return outputs
    
    def get_gate_params(self):
        gates = []
        for module in self.base_model.modules():
            if isinstance(module, SubspaceLoRALinear):
                gates.append(module.gate)
        return gates
