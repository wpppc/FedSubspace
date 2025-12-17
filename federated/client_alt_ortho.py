
import torch
from torch.optim import AdamW
import os
import transformers
import shutil

class FedAltClientOrtho:
    def __init__(self, client_id, model, tokenizer, dataloader, output_dir, local_epochs=1, max_steps=-1, lr=5e-4, device="cuda", data_collator=None, dtype=torch.float16, lambda_ortho=0.1, gradient_accumulation_steps=1):
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
        self.gradient_accumulation_steps = gradient_accumulation_steps
        
        # Persistent Local Vector Path
        self.local_state_path = os.path.join(output_dir, f"client_{client_id}_local_state.pt")

    def load_vectors(self, global_state):
        # 1. Load Global (不变)
        if isinstance(global_state, dict):
            global_theta = global_state['theta']
            if 'gates' in global_state:
                for name, val in global_state['gates'].items():
                    if ".gate_g" in name:
                        self._set_param_by_name(name, val)
        else:
            global_theta = global_state

        # 加载并冻结 Global
        self.model.adapter_global.theta_s.data.copy_(global_theta.to(self.device))
        self.model.adapter_global.theta_s.requires_grad_(False)
        
        # ==============================================================
        # 2. [终极修正] 安全的随机初始化 (Safe Random Initialization)
        # ==============================================================
        
        # A. 先生成高斯噪声 (提供动力)
        self.model.adapter_local.theta_s.data.normal_(0, 0.02)
        
        # B. [关键一步] 强制正交化 (Gram-Schmidt)
        # 如果 Global 不是零向量，我们就把 Local 中与 Global 平行的分量减掉
        # 这样初始的 Ortho Loss 绝对为 0，不会产生破坏性梯度
        theta_g = self.model.adapter_global.theta_s.data
        theta_l = self.model.adapter_local.theta_s.data
        
        norm_g = torch.norm(theta_g)
        if norm_g > 1e-6:
            # proj = (v_l . v_g) / (v_g . v_g) * v_g
            projection = torch.dot(theta_l, theta_g) / (norm_g ** 2) * theta_g
            theta_l.sub_(projection) # 原地减去投影
            
        # C. 强制 Gate 归零 (保护输出)
        # R0: Base + 0 + 0 = Base (Score 41)
        # R1: Base + Global + 0 = Better (Score > 41)
        for module in self.model.modules():
            if hasattr(module, "gate_l"):
                module.gate_l.data.fill_(0.0)
                
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

        opt.zero_grad()

        for epoch in range(self.local_epochs):
            for step, batch in enumerate(self.dataloader):
                # Move batch to device
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)
                
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
                    
                    loss = (task_loss + ortho_loss) / self.gradient_accumulation_steps
                
                if torch.isnan(loss):
                    opt.zero_grad()
                    del outputs, loss, task_loss, ortho_loss
                    continue

                scaler.scale(loss).backward()
                
                if (step + 1) % self.gradient_accumulation_steps == 0 or (step + 1) == len(self.dataloader):
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
                    opt.zero_grad()
                    
                    current_loss = loss.item() * self.gradient_accumulation_steps
                    current_ortho = ortho_loss.item()
                    del outputs, loss, task_loss, ortho_loss
                    
                    steps += 1
                    progress_bar.set_postfix({
                        "loss": f"{current_loss:.4f}", 
                        "ortho": f"{current_ortho:.8f}",
                        "gnorm": f"{grad_norm:.4f}",
                        "ng": f"{norm_g:.2f}",
                        "nl": f"{norm_l:.2f}"
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
        gate_values = [] # Store tanh(gate_l) values
        
        for name, module in self.model.named_modules():
            if hasattr(module, "gate_g"):
                gates[f"{name}.gate_g"] = module.gate_g.detach().cpu()
            
            if hasattr(module, "gate_l"):
                gates[f"{name}.gate_l"] = module.gate_l.detach().cpu()
                # Collect effective gate value
                gate_val = torch.tanh(module.gate_l).detach().item()
                gate_values.append(gate_val)
        
        # [Scale Mismatch Fix]
        # Bake-in the average gate scale into the vector
        if len(gate_values) > 0:
            avg_gate_scale = sum(gate_values) / len(gate_values)
        else:
            avg_gate_scale = 0.0
            
        print(f">> [Client {self.client_id}] Average Gate Scale: {avg_gate_scale:.4f}")

        # Get raw theta
        raw_theta = self.model.adapter_local.theta_s.detach().cpu().clone()
        
        # Scale it so Server receives the "Effective Residual"
        # Next round Global Gate is 1.0, so this matches the magnitude.
        scaled_theta = raw_theta * avg_gate_scale
        
        return {
            'theta': scaled_theta,
            'gates': gates
        }
