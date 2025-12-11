
import torch
from torch.optim import AdamW
import os
import transformers
import shutil

class FedDPAClient:
    def __init__(self, client_id, model, tokenizer, dataloader, output_dir, local_epochs=1, lr=3e-4, device="cuda", data_collator=None, dtype=torch.float16):
        self.client_id = client_id
        self.model = model  # FedDPAModelWrapper
        self.tokenizer = tokenizer
        self.dataloader = dataloader
        self.output_dir = output_dir
        self.local_epochs = local_epochs
        self.lr = lr
        self.device = device
        self.data_collator = data_collator
        self.dtype = dtype
        
        # Persistent Local State Path
        self.local_state_path = os.path.join(output_dir, f"client_{client_id}_local_state.pt")

    def load_state(self, global_state_dict):
        # 1. Load Global State (Aggregated)
        # global_state_dict contains keys for adapter_global
        self.model.load_state_dict(global_state_dict, strict=False)
        
        # 2. Load Local State (Personalized)
        if os.path.exists(self.local_state_path):
            local_state = torch.load(self.local_state_path, map_location=self.device)
            self.model.load_state_dict(local_state, strict=False)
        else:
            # First round: Initialize local = global? 
            # Or initialize local randomly (already done by init)?
            # FedDPA usually initializes local = global at start.
            # Let's copy global weights to local for a good start.
            self._init_local_from_global()
            
    def _init_local_from_global(self):
        # Copy adapter_global params to adapter_local
        for module in self.model.base_model.modules():
            if hasattr(module, "adapter_global") and hasattr(module, "adapter_local"):
                module.adapter_local.lora_A.data.copy_(module.adapter_global.lora_A.data)
                module.adapter_local.lora_B.data.copy_(module.adapter_global.lora_B.data)

    def train(self):
        from tqdm import tqdm
        import bitsandbytes as bnb
        
        # Optimizer: Train BOTH Global and Local adapters
        # FedDPA updates both during local training.
        params = []
        params.extend(self.model.get_global_params())
        params.extend(self.model.get_local_params())
        
        # Use 8-bit AdamW to save memory
        opt = bnb.optim.AdamW8bit(params, lr=self.lr)
        
        self.model.train()
        
        # Setup scaler
        # Only use scaler for float16. bfloat16 has enough range.
        use_scaler = (self.dtype == torch.float16)
        scaler = torch.amp.GradScaler('cuda', enabled=use_scaler)
        
        steps = 0
        total_steps = len(self.dataloader) * self.local_epochs
        
        # progress_bar = tqdm(range(total_steps), desc=f"Client {self.client_id} (FedDPA)", unit="step")
        
        print(f"  [Client {self.client_id}] Starting training for {self.local_epochs} epochs ({total_steps} steps)...", flush=True)

        for epoch in range(self.local_epochs):
            for batch_idx, batch in enumerate(self.dataloader):
                # Move batch to device
                # Ensure inputs are on the same device as the model
                input_ids = batch["input_ids"].to(self.model.base_model.device)
                attention_mask = batch["attention_mask"].to(self.model.base_model.device)
                labels = batch["labels"].to(self.model.base_model.device)
                
                # Use autocast with the correct dtype
                with torch.amp.autocast('cuda', dtype=self.dtype):
                    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    loss = outputs.loss
                
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                opt.zero_grad()
                
                # progress_bar.update(1)
                # progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
                steps += 1
                
                if steps % 10 == 0:
                    print(f"  [Client {self.client_id}] Step {steps}/{total_steps} Loss: {loss.item():.4f}", flush=True)

        
        # Save Local State
        local_state = self.model.get_local_state_dict()
        torch.save(local_state, self.local_state_path)

    def get_update_for_server(self):
        # Return only Global State
        return self.model.get_global_state_dict()
