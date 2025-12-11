# federated/server.py

import torch

class FedSubspaceServer:
    def __init__(self, d_s):
        self.global_theta = torch.zeros(d_s)

    def aggregate(self, updates, sizes, update_state=True):
        """
        Aggregates updates which can be:
        1. A list of tensors (FedSubspace)
        2. A list of dicts containing tensors (FedALT with Gates)
        """
        total = sum(sizes)
        
        # Case 1: List of Tensors
        if isinstance(updates[0], torch.Tensor):
            result = sum([theta * (n/total) for theta,n in zip(updates, sizes)])
            if update_state:
                self.global_theta = result.clone()
            return result
            
        # Case 2: List of Dicts (FedALT)
        elif isinstance(updates[0], dict):
            aggregated = {}
            keys = updates[0].keys()
            
            for k in keys:
                # Recursively aggregate if nested (e.g. 'gates' dict)
                if isinstance(updates[0][k], dict):
                    aggregated[k] = {}
                    sub_keys = updates[0][k].keys()
                    for sub_k in sub_keys:
                        val = sum([u[k][sub_k] * (n/total) for u,n in zip(updates, sizes)])
                        aggregated[k][sub_k] = val
                else:
                    # Aggregate tensor
                    val = sum([u[k] * (n/total) for u,n in zip(updates, sizes)])
                    aggregated[k] = val
            
            # Update global state if it matches structure
            if update_state:
                if hasattr(self, 'global_state'):
                    self.global_state = aggregated
                else:
                    # For backward compatibility, if 'theta' is the main key
                    if 'theta' in aggregated:
                        self.global_theta = aggregated['theta'].clone()
            
            return aggregated
        
        else:
            raise ValueError("Unsupported update type for aggregation")
