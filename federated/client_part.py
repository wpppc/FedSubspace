import torch
from torch.optim import AdamW
from tqdm import tqdm
from federated.client import FedSubspaceClient

class FedPartClient(FedSubspaceClient):
    def __init__(self, client_id, model, tokenizer, dataloader, output_dir, 
                 local_epochs=1, lr=1e-4, device="cuda", data_collator=None, 
                 dtype=torch.float32, batch_size=8, gradient_accumulation_steps=1,
                 group_id=0):
        super().__init__(client_id, model, tokenizer, dataloader, output_dir,
                         local_epochs, lr, device, data_collator, dtype,
                         batch_size, gradient_accumulation_steps)
        self.group_id = group_id
        
        # Create Gradient Mask
        # Assuming theta_s is 1D tensor
        dim = self.model.adapter.theta_s.shape[0]
        mid = dim // 2
        self.mask = torch.zeros(dim, device=self.device)
        
        if self.group_id == 0:
            # Group 0: Logic (First Half)
            self.mask[:mid] = 1.0
        else:
            # Group 1: Semantic (Second Half)
            self.mask[mid:] = 1.0
            
    def train(self):
        self.model.train()
        # Ensure theta_s requires grad
        self.model.adapter.theta_s.requires_grad_(True)
        
        optimizer = AdamW([self.model.adapter.theta_s], lr=self.lr)
        
        # Mixed Precision Scaler
        scaler = torch.amp.GradScaler('cuda')
        
        global_step = 0
        
        for epoch in range(self.local_epochs):
            # pbar = tqdm(self.dataloader, desc=f"Client {self.client_id} Epoch {epoch+1}/{self.local_epochs}", leave=False)
            for step, batch in enumerate(self.dataloader):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                with torch.amp.autocast('cuda', dtype=self.dtype):
                    outputs = self.model(**batch)
                    loss = outputs.loss
                    loss = loss / self.gradient_accumulation_steps
                
                scaler.scale(loss).backward()
                
                if (step + 1) % self.gradient_accumulation_steps == 0 or (step + 1) == len(self.dataloader):
                    # --- Apply Gradient Masking ---
                    if self.model.adapter.theta_s.grad is not None:
                        self.model.adapter.theta_s.grad *= self.mask
                    # ------------------------------
                    
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_([self.model.adapter.theta_s], 1.0)
                    
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    global_step += 1
                
                # del outputs, loss
