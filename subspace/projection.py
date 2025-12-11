# subspace/projection.py

import torch
import numpy as np

class RandomSubspaceProjection:
    """
    Efficient implementation of S * theta_s and S^T * grad_thetaD
    using random grouping (index_map).
    """
    def __init__(self, D, d_s, seed=42, device="cpu"):
        self.D = D
        self.d_s = d_s
        self.seed = seed
        self.device = device

        rng = np.random.RandomState(seed)
        self.index_map = torch.tensor(
            rng.randint(0, d_s, size=D),
            dtype=torch.long, device=device
        )

        counts = torch.bincount(self.index_map, minlength=d_s).float()
        self.norm = 1.0 / torch.sqrt(torch.clamp(counts, min=1.0))

    def project(self, theta_s):
        """
        θ_D = S θ_s  (vector of length D)
        each position i gets:
        θ_D[i] = θ_s[index_map[i]] * norm[index_map[i]]
        """
        return theta_s[self.index_map] * self.norm[self.index_map]

    def project_slice(self, theta_s, start, end):
        """
        Project only a slice of indices [start, end).
        """
        indices = self.index_map[start:end]
        norms = self.norm[indices]
        return theta_s[indices] * norms

    def backproject(self, grad_thetaD):
        """
        θ_s_grad = S^T grad(θ_D)
        using scatter_add
        """
        grad = torch.zeros(self.d_s, device=self.device)
        grad.index_add_(0, self.index_map, grad_thetaD * self.norm[self.index_map])
        return grad
