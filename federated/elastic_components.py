import torch
import copy
from federated.client import FedSubspaceClient
from federated.server import FedSubspaceServer

class ElasticFedSubspaceClient(FedSubspaceClient):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.active_dim = None # Will be set before training

    def set_active_dim(self, dim):
        self.active_dim = dim

    def train(self):
        """
        Elastic training: Only update the first `active_dim` parameters.
        """
        self.model.train()
        
        # Ensure we are working with the subspace parameters
        # The optimizer was initialized in __init__, but we need to make sure
        # it respects the gradient masking or slicing.
        
        # Since the optimizer holds references to parameters, and we don't want to 
        # reconstruct the optimizer every round (to keep momentum if needed, though 
        # in FL we often reset), we will use a gradient hook to zero out gradients
        # for indices >= active_dim.
        
        theta_s = self.model.adapter.theta_s
        
        # Define the hook
        def grad_mask_hook(grad):
            if self.active_dim is None or self.active_dim >= grad.shape[0]:
                return grad
            
            # Create a mask: 1 for [:active_dim], 0 for [active_dim:]
            # More efficient: just slice and zero
            # Clone to avoid in-place modification issues if any
            new_grad = grad.clone()
            new_grad[self.active_dim:] = 0.0
            return new_grad

        # Register hook
        handle = theta_s.register_hook(grad_mask_hook)

        try:
            # Standard training loop
            for epoch in range(self.local_epochs):
                for batch in self.dataloader:
                    batch = {k: v.to(self.device) for k, v in batch.items()}
                    outputs = self.model(**batch)
                    loss = outputs.loss
                    loss.backward()
                    
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    
                    if self.max_steps > 0:
                        # We don't track global steps here strictly for simplicity in this snippet
                        pass
        finally:
            # Remove the hook to avoid side effects if reused
            handle.remove()

    def get_theta(self):
        # Return the full theta, but the tail should be unchanged (or zero if initialized zero)
        # The server needs to know our active_dim to perform segmented aggregation
        return self.model.adapter.theta_s.detach().cpu().clone()

class ElasticFedSubspaceServer(FedSubspaceServer):
    def aggregate(self, thetas, sizes, active_dims):
        """
        Segmented Aggregation.
        thetas: list of full-size theta vectors
        sizes: list of dataset sizes (weights)
        active_dims: list of active dimensions for each client
        """
        if not thetas: return None
        
        # Ensure all on CPU
        thetas = [t.cpu() for t in thetas]
        
        full_dim = thetas[0].shape[0]
        aggregated_theta = torch.zeros(full_dim)
        
        # We need to aggregate index by index (or segment by segment)
        # To be efficient, we can process segments defined by the sorted unique values of active_dims
        
        # Sort unique dimensions to define segments
        # e.g. dims = [100, 100, 200, 500] -> segments: 0-100, 100-200, 200-500
        unique_dims = sorted(list(set(active_dims)))
        if unique_dims[-1] < full_dim:
            unique_dims.append(full_dim) # Handle the tail if no one updated it (though it should be 0)
            
        current_idx = 0
        for end_idx in unique_dims:
            if end_idx <= current_idx: continue
            
            # Identify clients who are active in this segment [current_idx, end_idx)
            # Client k is active if active_dims[k] >= end_idx
            # (Since active_dims[k] is the cutoff, they cover everything up to that point)
            
            segment_weights = []
            segment_updates = []
            
            for i, dim in enumerate(active_dims):
                if dim >= end_idx:
                    segment_weights.append(sizes[i])
                    segment_updates.append(thetas[i][current_idx:end_idx])
            
            if segment_weights:
                total_weight = sum(segment_weights)
                weighted_sum = torch.zeros(end_idx - current_idx)
                for w, update in zip(segment_weights, segment_updates):
                    weighted_sum += w * update
                
                aggregated_theta[current_idx:end_idx] = weighted_sum / total_weight
            else:
                # No one updated this segment, keep previous global (or 0)
                # In this implementation, we are returning the NEW global theta.
                # If we want to keep old values for untouched parts, we need self.global_theta
                # But usually we update in place.
                # Let's assume we retain the old value if no update.
                aggregated_theta[current_idx:end_idx] = self.global_theta[current_idx:end_idx].cpu()

            current_idx = end_idx
            
        self.global_theta = aggregated_theta
        return self.global_theta
