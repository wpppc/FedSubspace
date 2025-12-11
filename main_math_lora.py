# main_math_lora.py
import os, yaml, torch, json, csv, time, gc, random
from copy import deepcopy
from torch.utils.data import DataLoader
from tqdm import tqdm
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from peft import get_peft_model, LoraConfig, TaskType, get_peft_model_state_dict, set_peft_model_state_dict

from data.partition_tasks import partition_task
from data.format_fns import format_metamathqa
from data.dataset_tasks import GenericGenDataset, GenCollator

from evaluation.math_eval import generate_answers, is_correct

# ============================================================
#                Federated LoRA Classes
# ============================================================

class FedLoRAServer:
    def __init__(self, initial_state_dict):
        self.global_state_dict = {k: v.cpu().clone() for k, v in initial_state_dict.items()}

    def aggregate(self, state_dicts, sizes):
        if not state_dicts: return None
        
        total_size = sum(sizes)
        
        # Initialize accumulator
        avg_state = {k: torch.zeros_like(v) for k, v in state_dicts[0].items()}
        
        for state, size in zip(state_dicts, sizes):
            weight = size / total_size
            for k, v in state.items():
                if k in avg_state:
                    avg_state[k] += v.cpu() * weight
        
        self.global_state_dict = avg_state
        return avg_state

class FedLoRAClient:
    def __init__(self, client_id, model, tokenizer, dataloader, output_dir, local_epochs=1, max_steps=-1, lr=2e-4, device="cuda", data_collator=None, gradient_accumulation_steps=1, lr_scheduler_type="linear", warmup_ratio=0.0):
        self.client_id = client_id
        self.model = model # PeftModel
        self.tokenizer = tokenizer
        self.dataloader = dataloader
        self.output_dir = output_dir
        self.local_epochs = local_epochs
        self.max_steps = max_steps
        self.lr = lr
        self.device = device
        self.data_collator = data_collator
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.lr_scheduler_type = lr_scheduler_type
        self.warmup_ratio = warmup_ratio

    def train(self, global_state_dict):
        # Load global state
        set_peft_model_state_dict(self.model, global_state_dict)
        
        # Ensure only LoRA params are trainable
        self.model.print_trainable_parameters()

        def default_collate_fn(batch):
            return {
                "input_ids": torch.tensor([b["input_ids"] for b in batch]),
                "attention_mask": torch.tensor([b["attention_mask"] for b in batch]),
                "labels": torch.tensor([b["labels"] for b in batch]),
            }

        args = TrainingArguments(
            output_dir=os.path.join(self.output_dir, f"client_{self.client_id}"),
            per_device_train_batch_size=4,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            learning_rate=self.lr,
            num_train_epochs=self.local_epochs,
            max_steps=self.max_steps,
            optim="adamw_torch",
            fp16=True,
            logging_steps=10,
            save_strategy="no",
            report_to=[],
            remove_unused_columns=False,
            lr_scheduler_type=self.lr_scheduler_type,
            warmup_ratio=self.warmup_ratio,
            gradient_checkpointing=True 
        )

        trainer = Trainer(
            model=self.model,
            args=args,
            train_dataset=self.dataloader.dataset,
            data_collator=self.data_collator if self.data_collator else default_collate_fn,
            tokenizer=self.tokenizer,
        )

        trainer.train()
        
        # Get local state
        local_state = get_peft_model_state_dict(self.model)
        
        # Cleanup
        del trainer
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        
        return {k: v.cpu() for k, v in local_state.items()}

# ============================================================
#                Main Logic
# ============================================================

def get_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        num_params = param.numel()
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel
        if param.__class__.__name__ == "Params4bit":
            num_params = num_params * 2
        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params
    return trainable_params, all_param

def eval_gsm8k_peft(model, tokenizer, examples, device="cuda"):
    gens = generate_answers(model, tokenizer, examples, device=device)
    correct = 0
    for i, g in enumerate(gens):
        gold = examples[i]["output"]
        if is_correct(g, gold):
            correct += 1
    acc = correct / len(examples) if examples else 0.0
    return acc, f"Accuracy: {acc:.4f} ({correct}/{len(examples)})"

def main(cfg_path="configs/fedsubspace_multi_domain.yaml"):
    cfg = yaml.safe_load(open(cfg_path,"r"))
    os.makedirs(cfg["output_dir"], exist_ok=True)

    # Load Model
    print(f"Loading base model: {cfg['model']['path']}")
    base = AutoModelForCausalLM.from_pretrained(cfg["model"]["path"], torch_dtype=torch.float16)
    base.to("cuda")
    base.gradient_checkpointing_enable()
    base.enable_input_require_grads()
    
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["path"])
    tokenizer.pad_token_id = tokenizer.eos_token_id or 0

    # Setup PEFT LoRA
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, 
        inference_mode=False, 
        r=cfg["lora"]["r"], 
        lora_alpha=cfg["lora"]["r"] * 2, # Standard practice: alpha = 2*r
        lora_dropout=0.05,
        target_modules=cfg["lora"]["target_modules"]
    )
    model = get_peft_model(base, peft_config)
    model.print_trainable_parameters()

    trainable_params, all_param = get_trainable_parameters(model)
    print(f"Trainable Params: {trainable_params:,}")
    print(f"All Params: {all_param:,}")
    print(f"Trainable Ratio: {100 * trainable_params / all_param:.4f}%")

    # --- CSV Logging Setup ---
    csv_file = os.path.join(cfg["output_dir"], "experiment_results_lora.csv")
    if not os.path.exists(csv_file):
        with open(csv_file, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Timestamp", "BaseModel", "PartitionStrategy", "Domain", "Round", "Task", "Metric", "Value", "TrainableParams", "AllParams"])

    # Math Task Config (Same as main_math.py)
    task_cfg = {
        "name":"metamathqa",
        "hf_dataset": cfg["datasets"]["metamathqa"]["hf_name"],
        "train_split": cfg["datasets"]["metamathqa"].get("train_split","train"),
        "format_fn": format_metamathqa,
        "format_eval": False,
        "eval_sets": {
            "gsm8k": (cfg["datasets"]["gsm8k"]["hf_name"], cfg["datasets"]["gsm8k"].get("split","test"), "main"),
            "math": (cfg["datasets"]["math"]["hf_name"], cfg["datasets"]["math"].get("split","test"))
        }
    }

    # Partition (Reuse existing partition if available)
    out_root = os.path.join(cfg["data"]["root"], "metamathqa")
    if not os.path.exists(out_root) or not any(n.startswith("local_training_") for n in os.listdir(out_root)):
        print("Partitioning MetaMathQA...")
        partition_task(task_cfg, cfg["data"]["root"], cfg["data"]["num_clients"],
                       strategy=cfg["data"]["partition_strategy"],
                       alpha=cfg["data"]["partition_alpha"], seed=cfg["subspace"]["seed"])

    # Server
    initial_state = get_peft_model_state_dict(model)
    server = FedLoRAServer(initial_state)
    
    # Clients
    clients = []
    task_dir = os.path.join(cfg["data"]["root"], "metamathqa")
    for cid in range(cfg["data"]["num_clients"]):
        path = os.path.join(task_dir, f"local_training_{cid}.json")
        ds = GenericGenDataset(path)
        collator = GenCollator(tokenizer, max_len=cfg["data"]["cutoff_len"], train_on_inputs=cfg["data"]["train_on_inputs"])
        dl = DataLoader(ds, batch_size=cfg["train"]["batch_size"], shuffle=True, collate_fn=lambda b: collator(b))
        client = FedLoRAClient(client_id=cid, model=model, tokenizer=tokenizer, dataloader=dl,
                               output_dir=cfg["output_dir"], local_epochs=cfg["train"]["local_epochs"],
                               max_steps=cfg["train"].get("max_steps", -1),
                               lr=2e-4, # Use standard LoRA LR, 2e-3 is too high
                               device="cuda", data_collator=collator,
                               gradient_accumulation_steps=cfg["train"].get("gradient_accumulation_steps", 1),
                               lr_scheduler_type=cfg["train"].get("lr_scheduler_type", "linear"),
                               warmup_ratio=cfg["train"].get("warmup_ratio", 0.0))
        clients.append(client)

    # Training Loop
    rounds = cfg["federated"]["rounds"]
    num_selected = max(1, int(cfg["data"]["num_clients"] * cfg["federated"]["client_fraction"]))

    for r in range(rounds):
        print(f"--- Math LoRA Round {r} ---")
        
        # Randomly select clients, but seeded for reproducibility across experiments
        random.seed(cfg["subspace"]["seed"] + r)
        selected_cids = random.sample(range(cfg["data"]["num_clients"]), num_selected)
        selected_cids.sort()
        print(f"Selected Clients: {selected_cids}")
        
        local_states = []
        sizes = []
        
        for cid in selected_cids:
            print(f"Training Client {cid}...")
            c = clients[cid]
            # Train returns the local state dict
            state = c.train(server.global_state_dict)
            local_states.append(state)
            sizes.append(len(c.dataloader.dataset))
            
            gc.collect()
            torch.cuda.empty_cache()
        
        # Aggregate
        new_state = server.aggregate(local_states, sizes)
        
        # Save global model
        # We can save the adapter
        set_peft_model_state_dict(model, new_state)
        model.save_pretrained(os.path.join(cfg["output_dir"], f"math_lora_round{r}"))

        # Eval
        if cfg["eval"]["enabled"] and (r % cfg["eval"].get("eval_every",1) == 0):
            print(f"Evaluating LoRA Round {r}...")
            
            for eval_set in ["gsm8k", "math"]:
                eval_file = os.path.join(task_dir, f"global_eval_{eval_set}.json")
                if os.path.exists(eval_file):
                    with open(eval_file, 'r') as f: raw_examples = json.load(f)
                    
                    examples = []
                    for ex in raw_examples:
                        if "question" in ex and "answer" in ex: 
                            examples.append({"input": ex["question"], "output": ex["answer"]})
                        elif "problem" in ex and "solution" in ex: 
                            examples.append({"input": ex["problem"], "output": ex["solution"]})
                        elif "query" in ex and "response" in ex:
                            examples.append({"input": ex["query"], "output": ex["response"]})
                            
                    if examples:
                        max_samples = cfg["eval"].get("max_samples", None)
                        if max_samples and max_samples > 0 and len(examples) > max_samples:
                            examples = examples[:max_samples]
                            
                        print(f"Evaluating on {len(examples)} examples...")
                        acc, res_str = eval_gsm8k_peft(model, tokenizer, examples)
                        print(f"[Math LoRA] {eval_set} Result: {res_str}")
                        
                        with open(csv_file, mode='a', newline='') as f:
                            writer = csv.writer(f)
                            writer.writerow([
                                time.strftime("%Y-%m-%d %H:%M:%S"),
                                os.path.basename(cfg["model"]["path"]),
                                cfg["data"]["partition_strategy"],
                                "Math",
                                r,
                                eval_set,
                                "Accuracy",
                                acc,
                                trainable_params,
                                all_param
                            ])

if __name__=="__main__":
    main()
