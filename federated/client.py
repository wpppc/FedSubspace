# federated/client.py

import torch
from torch.optim import AdamW

import os
import torch
import transformers
from copy import deepcopy
from torch.optim import AdamW

import shutil

class FedSubspaceClient:
    """
    Professional version:
    - base model fully shared and frozen
    - only train theta_s
    - use HF Trainer for stability
    - save only theta_s update
    """

    def __init__(self, client_id, model, tokenizer, dataloader, output_dir, local_epochs=1, max_steps=-1, lr=5e-4, device="cuda", data_collator=None, gradient_accumulation_steps=1, lr_scheduler_type="linear", warmup_ratio=0.0, dtype=torch.float16):
        self.client_id = client_id
        self.model = model  # same shared base model wrapper
        self.tokenizer = tokenizer
        self.dataloader = dataloader
        self.output_dir = output_dir
        self.local_epochs = local_epochs
        self.max_steps = max_steps
        self.device = device
        self.data_collator = data_collator
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.lr_scheduler_type = lr_scheduler_type
        self.warmup_ratio = warmup_ratio
        self.dtype = dtype

        # Optimizer for theta_s AND gates
        self.optim_params = [self.model.adapter.theta_s] + self.model.get_gate_params()
        
        # Ensure it requires grad
        self.model.adapter.theta_s.requires_grad_(True)
        for g in self.model.get_gate_params():
            g.requires_grad_(True)
            
        # Re-create optimizer every time to ensure it tracks the correct parameter object
        # self.opt = AdamW(self.optim_params, lr=lr) 
        self.lr = lr

    def train(self):
        """
        Manual training loop to avoid Trainer/Accelerate recursion issues.
        """
        from tqdm import tqdm
        
        # Create optimizer
        opt = AdamW([self.model.adapter.theta_s] + self.model.get_gate_params(), lr=self.lr)
        
        self.model.train()
        
        # Setup scaler for fp16 (disable for bfloat16 as it has enough range)
        scaler = torch.amp.GradScaler('cuda', enabled=(self.dtype == torch.float16))
        
        steps = 0
        total_steps = len(self.dataloader) * self.local_epochs
        if self.max_steps > 0:
            total_steps = min(total_steps, self.max_steps)
            
        progress_bar = tqdm(range(total_steps), desc=f"Client {self.client_id}", unit="step")
        
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
                total_norm = torch.nn.utils.clip_grad_norm_(self.optim_params, max_norm=1.0)
                
                if not scaler.is_enabled():
                    if torch.isnan(total_norm) or torch.isinf(total_norm):
                        # print(f"Warning: NaN/Inf gradient detected at Client {self.client_id}. Skipping step.")
                        opt.zero_grad()
                        del outputs, loss
                        continue

                # Calculate grad norm for theta_s (monitoring)
                grad_norm = 0.0
                if self.model.adapter.theta_s.grad is not None:
                    grad_norm = self.model.adapter.theta_s.grad.norm().item()
                
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

        # Cleanup
        opt.zero_grad(set_to_none=True)
        del opt
        del scaler
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        
        # Force clear cache
        torch.cuda.ipc_collect()

    def get_theta(self):
        return self.model.adapter.theta_s.detach().cpu().clone()

    def load_theta(self, theta):
        self.model.adapter.theta_s.data.copy_(theta.to(self.device))
