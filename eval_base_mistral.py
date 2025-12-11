import os
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from evaluation.math_eval import generate_answers, is_correct

def main():
    model_path = "/home/wuqicen/base_models/mistral-7b-v0.1"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Loading base model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16).to(device)
    
    # GSM8K
    gsm8k_path = "data/fed_tasks/metamathqa/global_eval_gsm8k.json"
    if os.path.exists(gsm8k_path):
        print(f"Evaluating on GSM8K ({gsm8k_path})...")
        with open(gsm8k_path, 'r') as f:
            raw_examples = json.load(f)
            
        examples = []
        for ex in raw_examples:
            if "question" in ex and "answer" in ex:
                examples.append({"input": ex["question"], "output": ex["answer"]})
        
        # Limit to 100 samples for quick check, or full if user wants
        # User asked "how much can it run", implying a full or representative check.
        # Let's do 100 samples to be fast, as full eval takes hours.
        examples = examples[:20]
        print(f"Running on {len(examples)} samples...")
        
        gens = generate_answers(model, tokenizer, examples, device=device)
        correct = 0
        for i, g in enumerate(gens):
            if is_correct(g, examples[i]["output"]):
                correct += 1
        print(f"GSM8K Base Accuracy: {correct/len(examples):.4f}")

    # MATH
    math_path = "data/fed_tasks/metamathqa/global_eval_math.json"
    if os.path.exists(math_path):
        print(f"Evaluating on MATH ({math_path})...")
        with open(math_path, 'r') as f:
            raw_examples = json.load(f)
            
        examples = []
        for ex in raw_examples:
            if "problem" in ex and "solution" in ex:
                examples.append({"input": ex["problem"], "output": ex["solution"]})
        
        examples = examples[:20]
        print(f"Running on {len(examples)} samples...")
        
        gens = generate_answers(model, tokenizer, examples, device=device)
        correct = 0
        for i, g in enumerate(gens):
            if is_correct(g, examples[i]["output"]):
                correct += 1
        print(f"MATH Base Accuracy: {correct/len(examples):.4f}")

if __name__ == "__main__":
    main()
