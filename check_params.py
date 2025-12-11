
import torch
import torch.nn as nn
from models.llama_wrapper import FedSubspaceModelWrapper
from transformers import AutoModelForCausalLM

class MockAdapter:
    def __init__(self):
        self.theta_s = nn.Parameter(torch.randn(100))

def check():
    print("Loading model...")
    # We don't need to load the real huge model, just a mock or small one if possible.
    # But FedSubspaceModelWrapper expects a base_model.
    # Let's just mock the base model structure.
    
    class MockBaseModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.device = "cpu"
            self.layer = nn.Linear(10, 10)
            
    base = MockBaseModel()
    
    print("Initializing Wrapper...")
    # We need to mock SubspaceLoRAAdapter too or let it run.
    # Let's try to import the real one.
    
    # We need lora_shapes.
    lora_shapes = {"layer.lora": (torch.Size([10, 10]), torch.Size([10, 10]))}
    
    model = FedSubspaceModelWrapper(base, lora_shapes, d_s=100, target_modules=["layer"])
    
    print("Checking parameters...")
    found_theta = False
    for name, param in model.named_parameters():
        print(f"Param: {name}, Requires Grad: {param.requires_grad}")
        if "theta_s" in name:
            found_theta = True
            
    if found_theta:
        print("SUCCESS: theta_s is registered.")
    else:
        print("FAILURE: theta_s is NOT registered.")

if __name__ == "__main__":
    check()
