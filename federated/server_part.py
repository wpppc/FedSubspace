import torch
from federated.server import FedSubspaceServer

class FedPartServer(FedSubspaceServer):
    def aggregate_part(self, updates, sizes, client_ids):
        """
        Aggregates updates based on partitioning strategy.
        Group 0 (Logic): Clients [1, 3, 6, 7] -> Update First Half
        Group 1 (Semantic): Clients [0, 2, 4, 5] -> Update Second Half
        """
        # Define Groups
        group0_cids = [1, 3, 6, 7]
        group1_cids = [0, 2, 4, 5]
        
        # Identify indices in the updates list
        g0_indices = [i for i, cid in enumerate(client_ids) if cid in group0_cids]
        g1_indices = [i for i, cid in enumerate(client_ids) if cid in group1_cids]
        
        # Assume all updates have same shape
        dim = updates[0].shape[0]
        mid = dim // 2
        
        new_theta = torch.zeros_like(updates[0])
        
        # --- Aggregate Group 0 (Logic) -> First Half ---
        if g0_indices:
            total_s = sum([sizes[i] for i in g0_indices])
            # Weighted Sum of the FIRST HALF
            weighted_sum = sum([updates[i][:mid] * sizes[i] for i in g0_indices])
            new_theta[:mid] = weighted_sum / total_s
        else:
            # If no updates from G0, keep old value
            new_theta[:mid] = self.global_theta[:mid]
            
        # --- Aggregate Group 1 (Semantic) -> Second Half ---
        if g1_indices:
            total_s = sum([sizes[i] for i in g1_indices])
            # Weighted Sum of the SECOND HALF
            weighted_sum = sum([updates[i][mid:] * sizes[i] for i in g1_indices])
            new_theta[mid:] = weighted_sum / total_s
        else:
            # If no updates from G1, keep old value
            new_theta[mid:] = self.global_theta[mid:]
            
        self.global_theta = new_theta
        return new_theta
