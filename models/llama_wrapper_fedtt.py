import torch
import torch.nn as nn
import math
from transformers import AutoModelForCausalLM

class TTLowRankLinear(nn.Module):
    """
    Implements a Linear Layer where the weight matrix is parameterized by TT-Format.
    Based on FedTT (ACL 2025).
    """
    def __init__(self, in_features, out_features, tt_rank=8, input_factors=None, output_factors=None, init_zero=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.tt_rank = tt_rank
        
        # 1. Determine Factors (Hardcoded for LLaMA-7B alignment as per paper Table 10)
        # 4096 -> [16, 16, 16], 64 -> [4, 4, 4]
        if input_factors is None:
            if in_features == 4096: input_factors = [16, 16, 16]
            elif in_features == 64: input_factors = [4, 4, 4]
            else: raise ValueError(f"Unsupported in_features {in_features} for Auto-TT")
            
        if output_factors is None:
            if out_features == 4096: output_factors = [16, 16, 16]
            elif out_features == 64: output_factors = [4, 4, 4]
            else: raise ValueError(f"Unsupported out_features {out_features} for Auto-TT")

        self.input_factors = input_factors
        self.output_factors = output_factors
        self.num_cores = len(input_factors)
        
        # 2. Initialize TT-Cores
        # Cores shapes: [r_{k-1}, m_k, n_k, r_k]
        # Boundary ranks are 1
        ranks = [1] + [tt_rank] * (self.num_cores - 1) + [1]
        self.cores = nn.ParameterList()
        
        for k in range(self.num_cores):
            shape = (ranks[k], input_factors[k], output_factors[k], ranks[k+1])
            core = torch.empty(shape)
            # Init strategy: Normal for most, Zero for last if requested (to start as identity)
            if init_zero and k == self.num_cores - 1:
                nn.init.zeros_(core)
            else:
                nn.init.xavier_normal_(core)
            self.cores.append(nn.Parameter(core))

    def get_weight_matrix(self):
        """Reconstructs the full weight matrix from TT cores."""
        # W = G1 x G2 x G3 ...
        # This reconstruction is cheap for Adapter sizes (e.g. 4096x64)
        
        # Start with the first core: [1, m1, n1, r1] -> [m1*n1, r1]
        curr = self.cores[0].view(-1, self.cores[0].shape[-1])
        
        for k in range(1, self.num_cores):
            # Next core: [rk, mk, nk, r_{k+1}] -> [rk, mk*nk*r_{k+1}]
            core_flat = self.cores[k].view(self.cores[k].shape[0], -1)
            # Contract: [..., rk] @ [rk, ...]
            curr = curr @ core_flat 
            # Reshape for next iteration
            # curr becomes [m1...mk * n1...nk, r_{k+1}]
            curr = curr.view(-1, self.cores[k].shape[-1])
            
        # Final reshape to [In, Out] (PyTorch Linear expects [Out, In], so we transpose later)
        # Note: TT usually constructs [Row, Col], we mapped In->Row, Out->Col
        W = curr.view(self.in_features, self.out_features)
        return W

    def forward(self, x):
        # Reconstruct W on the fly: [In, Out]
        W = self.get_weight_matrix()
        
        # Linear layer: x @ W
        # x: [B, S, In], W: [In, Out] -> [B, S, Out]
        return x @ W

class FedTTAdapterBlock(nn.Module):
    """
    Standard Adapter with TT-Linear layers.
    Structure: Down-Project (TT) -> Act -> Up-Project (TT)
    """
    def __init__(self, in_features, bottleneck_dim=64, tt_rank=8, dropout=0.05):
        super().__init__()
        # Down: 4096 -> 64
        self.down_proj = TTLowRankLinear(
            in_features, bottleneck_dim, tt_rank=tt_rank, 
            input_factors=[16, 16, 16], output_factors=[4, 4, 4], init_zero=False
        )
        self.act = nn.GELU()
        # Up: 64 -> 4096
        # Init zero to ensure the adapter starts as Identity
        self.up_proj = TTLowRankLinear(
            bottleneck_dim, in_features, tt_rank=tt_rank,
            input_factors=[4, 4, 4], output_factors=[16, 16, 16], init_zero=True
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.down_proj(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.up_proj(x)
        return residual + x

class FedTTModelWrapper(nn.Module):
    def __init__(self, base_model_path, bottleneck_dim=64, tt_rank=8, target_modules=None, torch_dtype=torch.float16):
        super().__init__()
        
        # [Safety] Use eager attention to avoid recursion bugs
        print(f">> [FedTT] Loading Base Model...", flush=True)
        self.base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch_dtype,
            device_map="auto",
            attn_implementation="eager"
        )
        
        self.base_model.gradient_checkpointing_enable()
        self.base_model.enable_input_require_grads()
        
        self.target_modules = target_modules or ["q_proj", "v_proj"]
        self.tt_rank = tt_rank
        self.bottleneck_dim = bottleneck_dim
        
        self.inject_adapters()
        
        # Freeze Base Model
        for n, p in self.base_model.named_parameters():
            if "tt_adapter" not in n:
                p.requires_grad = False
            else:
                p.requires_grad = True

    def inject_adapters(self):
        modules_to_replace = {}
        for name, module in self.base_model.named_modules():
            if any(t in name for t in self.target_modules) and isinstance(module, nn.Linear):
                modules_to_replace[name] = module
        
        for name, module in modules_to_replace.items():
            parent_name, child_name = name.rsplit('.', 1)
            parent = self.base_model.get_submodule(parent_name)
            
            # Create a Wrapped Module that includes the original Linear + Adapter
            # We replace the Linear layer with a compound module? 
            # No, easier to keep Linear frozen and ADD the adapter output.
            # But to do that cleanly in HF structure, we usually subclass or hook.
            # For simplicity: We use the "Side-Branch" implementation inside a custom Linear.
            
            original_linear = module
            new_module = TTLinearWithAdapter(original_linear, self.bottleneck_dim, self.tt_rank)
            setattr(parent, child_name, new_module)

    def forward(self, input_ids, attention_mask=None, labels=None):
        return self.base_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    
    def get_trainable_state_dict(self):
        return {k: v.cpu() for k, v in self.state_dict().items() if "tt_adapter" in k}

    def load_trainable_state_dict(self, state_dict):
        self.load_state_dict(state_dict, strict=False)

class TTLinearWithAdapter(nn.Module):
    def __init__(self, original_linear, bottleneck_dim, tt_rank):
        super().__init__()
        self.original_linear = original_linear
        self.tt_adapter = FedTTAdapterBlock(
            original_linear.in_features, 
            bottleneck_dim=bottleneck_dim, 
            tt_rank=tt_rank
        )
        
    def forward(self, x):
        # Base Path (Frozen)
        with torch.no_grad():
            base_out = self.original_linear(x)
        
        # Adapter Path (Trainable)
        # Note: FedTT adds adapter output to the base output (Residual connection is inside AdapterBlock relative to its input)
        # Wait, standard Adapter adds to the output of the sub-layer. 
        # FedTT paper Fig 1(b) shows: Output = Linear(x) + Adapter(x) (Parallel/Side-tuning) 
        # or Output = Linear(x) + Adapter(Linear(x))? 
        # Fig 1(b) shows "Tensorized Adapter" is added to the "Output Hidden States".
        # Usually adapters are sequential or parallel. LoRA is parallel.
        # Given "Tensorized Adapter... integrated... same as LoRA", we treat it as Parallel.
        
        adapter_out = self.tt_adapter(x) - x # Remove residual inside block if we want pure additive
        # Actually, let's just make the adapter output additive
        # Our FedTTAdapterBlock returns x + adapter(x).
        # We want: Out = Base(x) + Adapter(x). 
        # So we use the adapter on x.
        
        # Let's adjust:
        # FedTTAdapterBlock above calculates: x + Down->Up(x).
        # We need Down->Up(x) to add to Base(x).
        
        adapter_signal = self.tt_adapter.up_proj(
            self.tt_adapter.dropout(
                self.tt_adapter.act(
                    self.tt_adapter.down_proj(x)
                )
            )
        )
        
        return base_out + adapter_signal