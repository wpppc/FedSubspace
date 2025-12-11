# main.py
import os
import yaml
import torch
from torch.utils.data import DataLoader
from copy import deepcopy

# -------- Base LLM / Tokenizer --------
from transformers import AutoTokenizer, AutoModelForCausalLM

# -------- FedSubspace Components --------
from models.lora_utils import extract_lora_shapes
from models.llama_wrapper import FedSubspaceModelWrapper
from subspace.projection import RandomSubspaceProjection
from subspace.utils import unflatten_lora_params

# -------- Federated Components --------
from federated.client import FedSubspaceClient
from federated.server import FedSubspaceServer

# -------- Data Pipeline --------
from data.partition_glue import partition_glue
from data.dataset import LocalJSONDataset, GlueClientDataset
from data.collator import DataCollatorForGLUE

# -------- Evaluation --------
from evaluation.glue_eval import inject_adapter_and_eval


# ============================================================
#                Decode θ_s → LoRA Adapter Weights
# ============================================================

def build_meta_from_shapes(lora_shapes):
    meta = []
    for name, (A_shape, B_shape) in lora_shapes.items():
        meta.append((f"{name}.A", list(A_shape), A_shape.numel()))
        meta.append((f"{name}.B", list(B_shape), B_shape.numel()))
    return meta


def decode_adapter(theta_s, lora_shapes, seed, device="cpu"):
    # compute D
    D = sum(A.numel() + B.numel() for A, B in lora_shapes.values())

    P = RandomSubspaceProjection(D, len(theta_s), seed=seed, device=device)
    theta_D = P.project(theta_s.to(device)).cpu()

    meta = build_meta_from_shapes(lora_shapes)
    return unflatten_lora_params(theta_D, meta)


# ============================================================
#                     Build Clients for GLUE
# ============================================================

def build_clients_glue(cfg, shared_model, tokenizer):
    data_cfg = cfg["data"]
    root = data_cfg["root"]
    num_clients = data_cfg["num_clients"]
    task = data_cfg["glue_task"]

    # Partition GLUE if needed
    if not os.path.exists(root) or not any(f.startswith("local_training_") for f in os.listdir(root)):
        print("Partitioning GLUE dataset...")
        partition_glue(
            task=task,
            num_clients=num_clients,
            out_dir=root,
            strategy=data_cfg.get("partition_strategy", "dirichlet"),
            alpha=data_cfg.get("partition_alpha", 0.5),
            seed=cfg["subspace"]["seed"]
        )

    clients = []
    for cid in range(num_clients):
        path = os.path.join(root, f"local_training_{cid}.json")
        raw_ds = LocalJSONDataset(path)
        examples = raw_ds.data

        ds_client = GlueClientDataset(
            examples,
            tokenizer,
            task=task,
            cutoff_len=data_cfg["cutoff_len"],
            train_on_inputs=data_cfg["train_on_inputs"]
        )

        dl = DataLoader(
            ds_client,
            batch_size=cfg["train"]["batch_size"],
            shuffle=True,
            collate_fn=DataCollatorForGLUE(tokenizer)
        )

        # critical: each client has its own θ_s, but shared base model wrapper object
        client_model = shared_model   # same object, but θ_s inside is different per client

        client = FedSubspaceClient(
            client_id=cid,
            model=client_model,
            tokenizer=tokenizer,
            dataloader=dl,
            output_dir=cfg["output_dir"],
            local_epochs=cfg["train"]["local_epochs"],
            lr=cfg["train"]["lr"],
            device="cuda"
        )
        clients.append(client)

    return clients


# ============================================================
#                     Federated Training Loop
# ============================================================

def main(cfg_path="configs/fedsubspace.yaml"):
    # Load config
    cfg = yaml.safe_load(open(cfg_path, "r"))
    os.makedirs(cfg["output_dir"], exist_ok=True)

    # ============================================================
    #                     Load Base LLM (Shared)
    # ============================================================

    print("Loading base model:", cfg["model"]["path"])
    base_model = AutoModelForCausalLM.from_pretrained(
        cfg["model"]["path"],
        torch_dtype=torch.float16,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["path"])
    tokenizer.pad_token_id = tokenizer.eos_token_id or 0

    # ------------------- freeze base model ---------------------
    for p in base_model.parameters():
        p.requires_grad = False

    # ============================================================
    #               Extract LoRA Shapes (for decode)
    # ============================================================

    lora_shapes = extract_lora_shapes(
        base_model,
        target_modules=cfg["lora"]["target_modules"],
        r=cfg["lora"]["r"]
    )

    # Wrap base model with FedSubspace logic
    shared_model = FedSubspaceModelWrapper(
        base_model=base_model,
        lora_shapes=lora_shapes,
        d_s=cfg["subspace"]["dim"],
        seed=cfg["subspace"]["seed"],
        target_modules=cfg["lora"]["target_modules"],
    )

    # ============================================================
    #               Build Clients (GLUE Mode)
    # ============================================================

    if cfg["data"]["mode"] == "glue":
        clients = build_clients_glue(cfg, shared_model, tokenizer)
    else:
        raise NotImplementedError

    # ============================================================
    #                        Build Server
    # ============================================================

    server = FedSubspaceServer(cfg["subspace"]["dim"])

    # ============================================================
    #                     Training Round Loop
    # ============================================================

    R = cfg["federated"]["rounds"]
    client_fraction = cfg["federated"].get("client_fraction", 1.0)
    num_clients = len(clients)

    for r in range(R):
        print(f"\n========== Round {r} ==========")

        # ------------------ client selection --------------------
        if client_fraction >= 1.0:
            selected = list(range(num_clients))
        else:
            k = max(1, int(num_clients * client_fraction))
            selected = torch.randperm(num_clients)[:k].tolist()
        print("Selected:", selected)

        # ------------------- local training ---------------------
        thetas = []
        sizes = []

        for cid in selected:
            client = clients[cid]
            # sync global theta
            client.load_theta(server.global_theta)

            print(f"Client {cid} training...")
            client.train()

            thetas.append(client.get_theta())
            sizes.append(len(client.dataloader.dataset))

        # ------------------- FedAvg aggregation ------------------
        new_global = server.aggregate(thetas, sizes)

        # Save θ_s
        torch.save(new_global, f"{cfg['output_dir']}/theta_round{r}.pt")

        # ------------------- Decode adapter ----------------------
        adapter_state = decode_adapter(
            theta_s=new_global,
            lora_shapes=lora_shapes,
            seed=cfg["subspace"]["seed"],
            device=base_model.device
        )
        torch.save(adapter_state, f"{cfg['output_dir']}/adapter_round{r}.pt")

        # ------------------ Optional GLUE eval ------------------
        if cfg["data"]["mode"] == "glue":
            print("Running GLUE evaluation...")
            res = inject_adapter_and_eval(
                adapter_state,
                base_model_name=cfg["model"]["path"],
                task=cfg["data"]["glue_task"]
            )
            print(f"[Round {r}] GLUE Eval:", res)

    print("Training finished.")


if __name__ == "__main__":
    main()
