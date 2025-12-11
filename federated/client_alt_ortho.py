
import torch
from torch.optim import AdamW
import os
import transformers
import shutil

class FedAltClientOrtho:
    def __init__(self, client_id, model, tokenizer, dataloader, output_dir, local_epochs=1, max_steps=-1, lr=5e-4, device="cuda", data_collator=None, dtype=torch.float16, lambda_ortho=0.1):
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
        self.lambda_ortho = lambda_ortho
        
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
                # FIX: Do NOT copy Global to Local. Reset Local to Random Noise to avoid "Destructive Rotation".
                # We want Local to learn the *residual* (what Global missed), not start from Global.
                self.model.adapter_local.theta_s.data.normal_(0, 0.02)
                
                # Reset Gate to 0 to ensure we start from Global (Base + Global + 0*Local)
                # This allows Local to grow gradually via gradients.
                for module in self.model.modules():
                    if hasattr(module, "gate_l"):
                        module.gate_l.data.fill_(0.0)
            
            # Gates remain at default initialization (0.0 for gate_l)
            
        self.model.adapter_local.theta_s.requires_grad_(True)

    def _set_param_by_name(self, name, value):
        try:
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
        Manual training loop with Orthogonality Regularization.
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
            
        progress_bar = tqdm(range(total_steps), desc=f"Client {self.client_id} (ALT+Ortho)", unit="step")

        for epoch in range(self.local_epochs):
            for batch in self.dataloader:
                # Move batch to device
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)
                
                opt.zero_grad()
                
                with torch.amp.autocast('cuda', dtype=self.dtype):
                    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    task_loss = outputs.loss
                    
                    # --- Orthogonality Regularization ---
                    theta_l = self.model.adapter_local.theta_s
                    theta_g = self.model.adapter_global.theta_s.detach()
                    
                    norm_l = torch.norm(theta_l)
                    norm_g = torch.norm(theta_g)
                    
                    ortho_loss = torch.tensor(0.0, device=self.device)
                    if norm_l > 1e-6 and norm_g > 1e-6:
                        cos_sim = torch.dot(theta_l, theta_g) / (norm_l * norm_g)
                        ortho_loss = self.lambda_ortho * (cos_sim ** 2)
                    
                    loss = task_loss + ortho_loss
                
                if torch.isnan(loss):
                    opt.zero_grad()
                    del outputs, loss, task_loss, ortho_loss
                    continue

                scaler.scale(loss).backward()
                
                scaler.unscale_(opt)
                total_norm = torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                
                if not scaler.is_enabled():
                    if torch.isnan(total_norm) or torch.isinf(total_norm):
                        opt.zero_grad()
                        del outputs, loss, task_loss, ortho_loss
                        continue

                grad_norm = 0.0
                # Check gradients for both theta_l and gate_l
                if self.model.adapter_local.theta_s.grad is not None:
                    grad_norm += self.model.adapter_local.theta_s.grad.norm().item()
                
                # Also check gates (just sample one for logging)
                # Finding the first gate_l to check its grad
                for m in self.model.modules():
                    if hasattr(m, 'gate_l') and m.gate_l.grad is not None:
                        grad_norm += m.gate_l.grad.norm().item()
                        break # Just add one gate's norm to indicate activity
                
                scaler.step(opt)
                scaler.update()
                
                current_loss = loss.item()
                current_ortho = ortho_loss.item()
                del outputs, loss, task_loss, ortho_loss
                
                steps += 1
                progress_bar.set_postfix({
                    "loss": f"{current_loss:.4f}", 
                    "ortho": f"{current_ortho:.4f}",
                    "gnorm": f"{grad_norm:.4f}"
                })
                progress_bar.update(1)
                
                if self.max_steps > 0 and steps >= self.max_steps:
                    break
            if self.max_steps > 0 and steps >= self.max_steps:
                break
        
        progress_bar.close()
        
        # Save updated local state (Theta + Gates)
        local_gates = {}
        for name, module in self.model.named_modules():
            if hasattr(module, "gate_l"): 
                local_gates[f"{name}.gate_l"] = module.gate_l.detach().cpu()
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
