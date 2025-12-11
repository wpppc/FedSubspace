# subspace/adapter.py

import torch
import torch.nn as nn
from subspace.projection import RandomSubspaceProjection

class SubspaceLoRAAdapter(nn.Module):
    """
    Replace LoRA's A, B parameters with a vector θ_s,
    and generate ΔW through projection.
    """
    def __init__(self, shapes, d_s, seed, device="cpu", init_zeros=False):
        """
        shapes: dict {layer_name: (shape_A, shape_B)}
        flat_dim = total number of LoRA parameters
        """
        super().__init__()
        self.device = device

        # Total D for θ_D
        self.shapes = shapes
        self.sizes = {k: A.numel() + B.numel() for k, (A, B) in shapes.items()}
        self.offsets = {}

        offset = 0
        for k in shapes:
            self.offsets[k] = offset
            offset += shapes[k][0].numel() + shapes[k][1].numel()
        self.D = offset

        # parameters
        self.theta_s = nn.Parameter(
            torch.zeros(d_s, dtype=torch.float32, device=device)
        )
        # Initialize with larger variance to match LoRA initialization scale
        # LoRA A is usually initialized with Kaiming Uniform (std ~ 0.01)
        # Since we project, we need slightly larger std in theta_s
        if not init_zeros:
            nn.init.normal_(self.theta_s, mean=0.0, std=0.02)

        self.proj = RandomSubspaceProjection(self.D, d_s, seed=seed, device=device)

    def forward(self):
        """
        Returns a dict {layer_name: ΔW_tensor}
        """
        theta_D = self.proj.project(self.theta_s)
        results = {}
        for layer_name, (A_shape, B_shape) in self.shapes.items():
            off = self.offsets[layer_name]
            A_num = A_shape.numel()
            B_num = B_shape.numel()

            A = theta_D[off : off+A_num].reshape(A_shape)
            B = theta_D[off+A_num : off+A_num+B_num].reshape(B_shape)

            results[layer_name] = (A, B)
        return results

    def get_layer_deltas(self, layer_name):
        """
        Compute deltas for a specific layer on the fly.
        """
        if layer_name not in self.shapes:
            return None
            
        off = self.offsets[layer_name]
        A_shape, B_shape = self.shapes[layer_name]
        A_num = A_shape.numel()
        B_num = B_shape.numel()
        
        # Project only the needed slice
        # The slice in theta_D corresponds to [off, off + A_num + B_num]
        theta_D_slice = self.proj.project_slice(self.theta_s, off, off + A_num + B_num)
        
        A = theta_D_slice[:A_num].reshape(A_shape)
        B = theta_D_slice[A_num:].reshape(B_shape)
        
        # Ensure they retain gradient history
        # print(f"DEBUG: get_layer_deltas {layer_name} theta_s.grad_fn={self.theta_s.grad_fn} slice.grad_fn={theta_D_slice.grad_fn}")
        
        return A, B
