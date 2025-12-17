
import torch
import transformers
import os
from transformers import Trainer

class DualMixTrainer(Trainer):
    def __init__(self, lambda_reg=0.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lambda_reg = lambda_reg

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(**inputs)
        loss = outputs.loss if isinstance(outputs, dict) else outputs[0]
        
        if self.lambda_reg > 0:
            # Access local theta from the model wrapper
            # Assuming model is FedDualMixModelWrapper
            theta_local = model.adapter_local.theta_s
            # L2 Regularization: 0.01 * (theta_l ** 2).mean()
            l2_loss = (theta_local ** 2).mean()
            loss += self.lambda_reg * l2_loss
            
        return (loss, outputs) if return_outputs else loss

class FedDualMixClient:
    def __init__(self, client_id, model, tokenizer, dataloader, output_dir, 
                 local_epochs=1, lr=5e-4, device="cuda", data_collator=None, 
                 dtype=torch.float16, batch_size=2, gradient_accumulation_steps=1,
                 lambda_reg=0.0):
        self.client_id = client_id
        self.model = model
        self.tokenizer = tokenizer
        self.dataloader = dataloader
        self.output_dir = output_dir
        self.local_epochs = local_epochs
        self.device = device
        self.data_collator = data_collator
        self.dtype = dtype
        self.lr = lr
        self.batch_size = batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.lambda_reg = lambda_reg

    def train(self):
        def collate_fn(batch):
            return {
                "input_ids": torch.tensor([b["input_ids"] for b in batch]),
                "attention_mask": torch.tensor([b["attention_mask"] for b in batch]),
                "labels": torch.tensor([b["labels"] for b in batch]),
            }

        args = transformers.TrainingArguments(
            output_dir=os.path.join(self.output_dir, f"client_{self.client_id}"),
            per_device_train_batch_size=self.batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            learning_rate=self.lr,
            num_train_epochs=self.local_epochs,
            optim="adamw_torch",
            fp16=(self.dtype == torch.float16),
            bf16=(self.dtype == torch.bfloat16),
            logging_steps=10,
            save_strategy="no",
            report_to=[],
            use_cpu=False,
            remove_unused_columns=False
        )

        trainer = DualMixTrainer(
            lambda_reg=self.lambda_reg,
            model=self.model,
            args=args,
            train_dataset=self.dataloader.dataset,
            data_collator=self.data_collator if self.data_collator else collate_fn,
            tokenizer=self.tokenizer,
        )

        trainer.train()

    def get_theta_global(self):
        return self.model.adapter_global.theta_s.detach().cpu().clone()

    def get_theta_local(self):
        return self.model.adapter_local.theta_s.detach().cpu().clone()
    
    def get_alphas(self):
        # Return a dict of gate values for local storage
        gates = {}
        for n, m in self.model.named_modules():
            if hasattr(m, 'gate_g'):
                gates[f"{n}.gate_g"] = m.gate_g.detach().cpu().clone()
            if hasattr(m, 'gate_l'):
                gates[f"{n}.gate_l"] = m.gate_l.detach().cpu().clone()
        return gates

    def load_theta_global(self, theta):
        self.model.adapter_global.theta_s.data.copy_(theta.to(self.device))

    def load_theta_local(self, theta):
        self.model.adapter_local.theta_s.data.copy_(theta.to(self.device))
        
    def load_alphas(self, alphas_dict):
        for n, m in self.model.named_modules():
            if hasattr(m, 'gate_g') and f"{n}.gate_g" in alphas_dict:
                m.gate_g.data.copy_(alphas_dict[f"{n}.gate_g"].to(self.device))
            if hasattr(m, 'gate_l') and f"{n}.gate_l" in alphas_dict:
                m.gate_l.data.copy_(alphas_dict[f"{n}.gate_l"].to(self.device))
