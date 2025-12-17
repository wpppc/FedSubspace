import torch
from torch.optim import AdamW
import os
from tqdm import tqdm

class FedTTClient:
    def __init__(self, client_id, model, dataloader, output_dir, 
                 local_epochs=2, lr=2e-4, device="cuda", dtype=torch.float16,
                 gradient_accumulation_steps=1):
        self.client_id = client_id
        self.model = model
        self.dataloader = dataloader
        self.output_dir = output_dir
        self.local_epochs = local_epochs
        self.lr = lr
        self.device = device
        self.dtype = dtype
        self.gradient_accumulation_steps = gradient_accumulation_steps

    def train(self):
        # Only train TT parameters
        params = [p for n, p in self.model.named_parameters() if p.requires_grad]
        optimizer = AdamW(params, lr=self.lr)
        
        self.model.train()
        scaler = torch.amp.GradScaler('cuda', enabled=(self.dtype == torch.float16))
        
        optimizer.zero_grad()
        
        for epoch in range(self.local_epochs):
            with tqdm(self.dataloader, desc=f"Client {self.client_id} (FedTT)", leave=False) as pbar:
                for step, batch in enumerate(pbar):
                    input_ids = batch["input_ids"].to(self.device)
                    attention_mask = batch["attention_mask"].to(self.device)
                    labels = batch["labels"].to(self.device)
                    
                    with torch.amp.autocast('cuda', dtype=self.dtype):
                        outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
                        loss = outputs.loss / self.gradient_accumulation_steps
                    
                    if torch.isnan(loss):
                        optimizer.zero_grad()
                        continue

                    scaler.scale(loss).backward()
                    
                    if (step + 1) % self.gradient_accumulation_steps == 0 or (step + 1) == len(self.dataloader):
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(params, 1.0)
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad()
                        
                        pbar.set_postfix(loss=f"{loss.item() * self.gradient_accumulation_steps:.4f}")

    def get_update_for_server(self):
        return self.model.get_trainable_state_dict()