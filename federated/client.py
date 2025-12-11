# federated/client.py

import torch
from torch.optim import AdamW

import os
import torch
import transformers
from copy import deepcopy
from torch.optim import AdamW

class FedSubspaceClient:
    """
    Professional version:
    - base model fully shared and frozen
    - only train theta_s
    - use HF Trainer for stability
    - save only theta_s update
    """

    def __init__(self, client_id, model, tokenizer, dataloader, output_dir, local_epochs=1, lr=5e-4, device="cuda"):
        self.client_id = client_id
        self.model = model  # same shared base model wrapper
        self.tokenizer = tokenizer
        self.dataloader = dataloader
        self.output_dir = output_dir
        self.local_epochs = local_epochs
        self.device = device

        # Optimizer ONLY for theta_s
        self.optim_params = [self.model.adapter.theta_s]
        self.opt = AdamW(self.optim_params, lr=lr)

    def train(self):
        """
        Use a simple HF Trainer wrapper for professional training.
        """

        def collate_fn(batch):
            return {
                "input_ids": torch.tensor([b["input_ids"] for b in batch]),
                "attention_mask": torch.tensor([b["attention_mask"] for b in batch]),
                "labels": torch.tensor([b["labels"] for b in batch]),
            }

        args = transformers.TrainingArguments(
            output_dir=os.path.join(self.output_dir, f"client_{self.client_id}"),
            per_device_train_batch_size=4,
            gradient_accumulation_steps=1,
            learning_rate=self.opt.param_groups[0]["lr"],
            num_train_epochs=self.local_epochs,
            optim="adamw_torch",
            fp16=True,
            logging_steps=10,
            save_strategy="no",
            report_to=[]
        )

        trainer = transformers.Trainer(
            model=self.model,
            args=args,
            train_dataset=self.dataloader.dataset,
            data_collator=collate_fn,
            tokenizer=self.tokenizer,
        )

        trainer.train()

    def get_theta(self):
        return self.model.adapter.theta_s.detach().cpu().clone()

    def load_theta(self, theta):
        self.model.adapter.theta_s.data.copy_(theta.to(self.device))
