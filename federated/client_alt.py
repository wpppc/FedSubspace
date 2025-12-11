
import torch
from torch.optim import AdamW
import os
import transformers
import shutil

class FedAltClient:
    def __init__(self, client_id, model, tokenizer, dataloader, output_dir, local_epochs=1, max_steps=-1, lr=5e-4, device="cuda", data_collator=None, dtype=torch.float16):
        self.client_id = client_id
        self.model = model  # FedDualSubspaceModelWrapper
        self.tokenizer = tokenizer
        self.dataloader = dataloader
        self.output_dir = output_dir
        self.local_epochs = local_epochs
        self.max_steps = max_steps
        self.lr = lr
        self.device = device
        self.data_collator = data_collator
        self.dtype = dtype
        
        # Persistent Local Vector Path
        self.local_state_path = os.path.join(output_dir, f"client_{client_id}_local_state.pt")

    def load_vectors(self, global_state):
        # 1. Load Global Vector (Frozen)
        # global_state can be just theta (first round) or dict (subsequent rounds)
        if isinstance(global_state, dict):
            global_theta = global_state['theta']
            # Load Global Gates if available
            if 'gates' in global_state:
                for name, val in global_state['gates'].items():
                    # name is like "model.layers.0.self_attn.q_proj.gate_g"
                    # We need to find the module
                    # This is tricky because names are absolute.
                    # Let's iterate modules and match names.
                    # Actually, simpler: we know the structure.
                    pass 
                    # TODO: Implement loading global gates properly if we want to sync them.
                    # For now, let's assume global gates are part of the aggregation but 
                    # maybe we don't overwrite local instances? 
                    # Wait, FedALT usually keeps Global Model identical across clients.
                    # So we MUST load global gates.
                    
                    # Helper to set parameter by name
                    self._set_param_by_name(name, val)
        else:
            global_theta = global_state

        self.model.adapter_global.theta_s.data.copy_(global_theta.to(self.device))
        self.model.adapter_global.theta_s.requires_grad_(False)
        
        # 2. Load Local State (Trainable Theta + Gates)
        if os.path.exists(self.local_state_path):
            state = torch.load(self.local_state_path, map_location=self.device)
            self.model.adapter_local.theta_s.data.copy_(state['theta'])
            
            # Load Local Gates
            if 'gates' in state:
                for name, val in state['gates'].items():
                    self._set_param_by_name(name, val)
        else:
            # First round or missing local state
            # Check if global_theta is Zero (Identity). If so, keep Local as Random (from init) to avoid Deadlock.
            # If Global is learned (non-zero), initialize Local from Global.
            if global_theta.abs().sum() == 0:
                # Global is Zero (Identity). Keep Local as Random (initialized in wrapper).
                pass
            else:
                # Global has knowledge. Initialize Local from Global.
                self.model.adapter_local.theta_s.data.copy_(global_theta.to(self.device))
            
            # Gates remain at default initialization (0.0 for gate_l)
            
        self.model.adapter_local.theta_s.requires_grad_(True)

    def _set_param_by_name(self, name, value):
        # name example: "model.layers.0.self_attn.q_proj.gate_g"
        try:
            # Split into module path and param name
            if '.' in name:
                module_path, param_name = name.rsplit('.', 1)
                module = self.model.get_submodule(module_path)
            else:
                module = self.model
                param_name = name
            
            if hasattr(module, param_name):
                p = getattr(module, param_name)
                p.data.copy_(value.to(self.device))
        except Exception as e:
            print(f"Warning: Failed to load param {name}: {e}")

    def train(self):
        """
        Manual training loop to avoid Trainer/Accelerate recursion issues.
        """
        from tqdm import tqdm
        
        # Optimizer: Only train Local Theta + Gates
        trainable_params = self.model.get_trainable_params()
        opt = AdamW(trainable_params, lr=self.lr)
        
        self.model.train()
        
        # Setup scaler for fp16 (disable for bfloat16 as it has enough range)
        scaler = torch.amp.GradScaler('cuda', enabled=(self.dtype == torch.float16))
        
        steps = 0
        total_steps = len(self.dataloader) * self.local_epochs
        if self.max_steps > 0:
            total_steps = min(total_steps, self.max_steps)
            
        progress_bar = tqdm(range(total_steps), desc=f"Client {self.client_id} (ALT)", unit="step")

        for epoch in range(self.local_epochs):
            for batch in self.dataloader:
                # Move batch to device
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)
                
                opt.zero_grad()
                
                with torch.amp.autocast('cuda', dtype=self.dtype):
                    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    loss = outputs.loss
                
                if torch.isnan(loss):
                    # print(f"Warning: NaN loss detected at Client {self.client_id}. Skipping step.")
                    opt.zero_grad()
                    del outputs, loss
                    continue

                scaler.scale(loss).backward()
                
                # Unscale gradients to check their true norm
                scaler.unscale_(opt)
                
                # Clip gradients
                total_norm = torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                
                if not scaler.is_enabled():
                    if torch.isnan(total_norm) or torch.isinf(total_norm):
                        # print(f"Warning: NaN/Inf gradient detected at Client {self.client_id}. Skipping step.")
                        opt.zero_grad()
                        del outputs, loss
                        continue

                # Calculate grad norm for local theta_s (monitoring)
                grad_norm = 0.0
                if self.model.adapter_local.theta_s.grad is not None:
                    grad_norm += self.model.adapter_local.theta_s.grad.norm().item()
                
                # Also check gates (just sample one for logging)
                for m in self.model.modules():
                    if hasattr(m, 'gate_l') and m.gate_l.grad is not None:
                        grad_norm += m.gate_l.grad.norm().item()
                        break 
                
                scaler.step(opt)
                scaler.update()
                
                # Explicitly delete tensors to free memory
                current_loss = loss.item()
                del outputs, loss
                
                steps += 1
                progress_bar.set_postfix({"loss": f"{current_loss:.4f}", "gnorm": f"{grad_norm:.4f}"})
                progress_bar.update(1)
                
                if self.max_steps > 0 and steps >= self.max_steps:
                    break
            if self.max_steps > 0 and steps >= self.max_steps:
                break
        
        progress_bar.close()
        
        # Save updated local state (Theta + Gates)
        local_gates = {}
        for name, module in self.model.named_modules():
            if hasattr(module, "gate_l"): # DualSubspaceLoRALinear
                local_gates[f"{name}.gate_l"] = module.gate_l.detach().cpu()
                # We also save gate_g locally? No, gate_g is global.
                # But wait, if we train gate_g locally (do we?), we should aggregate it.
                # In FedALT, usually only Local is trained?
                # Let's check get_trainable_params in wrapper.
                # It returns gate_g AND gate_l. So we train BOTH.
                # So we must save BOTH to send to server, but for LOCAL persistence,
                # we mainly care about restoring the state.
                local_gates[f"{name}.gate_g"] = module.gate_g.detach().cpu()

        state_to_save = {
            'theta': self.model.adapter_local.theta_s.detach().cpu(),
            'gates': local_gates
        }
        torch.save(state_to_save, self.local_state_path)
        
        # Cleanup
        opt.zero_grad(set_to_none=True)
        del opt
        del scaler
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        
        # Do NOT remove output dir if it contains persistent files
        # if os.path.exists(self.output_dir):
        #    shutil.rmtree(self.output_dir, ignore_errors=True)

    def get_update_for_server(self):
        # Return Theta and Gates for aggregation
        gates = {}
        for name, module in self.model.named_modules():
            if hasattr(module, "gate_g"):
                gates[f"{name}.gate_g"] = module.gate_g.detach().cpu()
                gates[f"{name}.gate_l"] = module.gate_l.detach().cpu()
        
        return {
            'theta': self.model.adapter_local.theta_s.detach().cpu().clone(),
            'gates': gates
        }
