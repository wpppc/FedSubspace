# evaluation/math_eval.py
import torch
from tqdm import tqdm
from evaluation.math_utils import math_equal, last_boxed_only_string, remove_boxed

def generate_answers(model, tokenizer, examples, device="cuda", max_new_tokens=256):
    model.to(device)
    model.eval()
    out = []
    for ex in tqdm(examples):
        prompt = ex["input"]
        # Ensure prompt is formatted as instruction if needed, but main_math.py seems to pass raw question.
        # Ideally we should wrap it in the instruction template used during training.
        # But for now let's assume the caller handles it or the model is robust.
        # Actually, main_math.py passes raw question. The model might expect a template.
        # However, let's stick to the current logic of just generating.
        
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            gen = model.generate(**inputs, max_new_tokens=max_new_tokens)
        txt = tokenizer.decode(gen[0], skip_special_tokens=True)
        # Remove the prompt from the output to get just the generated answer
        # This is important because the model might repeat the prompt
        if txt.startswith(prompt):
            txt = txt[len(prompt):]
        out.append(txt)
    return out

import re

def extract_answer(text):
    # 1. Try GSM8K style "####"
    if "####" in text:
        return text.split("####")[-1].strip()
    
    # 2. Try \boxed{} (MATH style)
    boxed = last_boxed_only_string(text)
    if boxed:
        return remove_boxed(boxed)
    
    # 3. Fallback: look for "The answer is"
    if "The answer is" in text:
        return text.split("The answer is")[-1].strip()
        
    # 4. Fallback: look for last number (risky for MATH but okay for simple numeric)
    matches = re.findall(r'-?\d+\.?\d*', text)
    if matches:
        return matches[-1]
    
    return text # Return full text if nothing found, let grader handle it

def is_correct(pred, gold):
    pred_ans = extract_answer(pred)
    # Gold might also need extraction if it's full text, but usually it's the answer field.
    # In main_math.py:
    # GSM8K: output is "answer" (full solution with ####) -> we need to extract from gold too?
    # MATH: output is "solution" (full solution with \boxed{}) -> we need to extract from gold too?
    # MetaMathQA: output is "response" -> full solution?
    
    # Let's try to extract from gold as well, just in case.
    gold_ans = extract_answer(gold)
    
    return math_equal(pred_ans, gold_ans)

def eval_gsm8k_with_adapter(adapter_state, base_model, tokenizer, examples, device="cuda"):
    """
    adapter_state: dict of LoRA deltas (decoded)
    base_model: HF model object (AutoModelForCausalLM)
    examples: list of {"input":..., "output":...} for GSM8K test
    """
    # Note: adapter injection is handled by AdapterInjector context manager in main script.
    # We just generate answers here.

    gens = generate_answers(base_model, tokenizer, examples, device=device)
    
    # Compute accuracy
    correct = 0
    for i, g in enumerate(gens):
        gold = examples[i]["output"]
        if is_correct(g, gold):
            correct += 1
            
    acc = correct / len(examples) if examples else 0.0
    
    return acc, f"Accuracy: {acc:.4f} ({correct}/{len(examples)})"

