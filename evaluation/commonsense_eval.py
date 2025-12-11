# evaluation/commonsense_eval.py
import torch
import numpy as np
from tqdm import tqdm

def get_log_prob_of_continuation(model, tokenizer, prefix, continuation, device="cuda"):
    """
    Computes the log probability of the continuation given the prefix.
    P(continuation | prefix)
    """
    full_text = prefix + continuation
    enc_full = tokenizer(full_text, return_tensors="pt").to(device)
    
    # Lengths
    enc_prefix = tokenizer(prefix, return_tensors="pt").to(device)
    len_prefix = enc_prefix.input_ids.shape[1]
    len_full = enc_full.input_ids.shape[1]
    
    input_ids = enc_full.input_ids
    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits # [1, seq_len, vocab]
        
    # Shift logits: logits[i] predicts input_ids[i+1]
    # We want logits for the continuation part.
    # Continuation starts at input_ids[len_prefix].
    # Its logit is at logits[len_prefix-1].
    
    start_logit_idx = max(0, len_prefix - 1)
    end_logit_idx = len_full - 1
    
    if start_logit_idx >= end_logit_idx:
        return -9999.0

    relevant_logits = logits[0, start_logit_idx:end_logit_idx, :] # [continuation_len, vocab]
    relevant_ids = input_ids[0, start_logit_idx+1:end_logit_idx+1] # [continuation_len]
    
    log_probs = torch.nn.functional.log_softmax(relevant_logits, dim=-1)
    token_log_probs = log_probs.gather(dim=-1, index=relevant_ids.unsqueeze(-1)).squeeze(-1)
    
    return token_log_probs.sum().item()

def parse_example(ex, task_name):
    """
    Returns:
        prompt: str
        choices: list of str
        label_idx: int (index of correct choice)
    """
    task = task_name.lower()
    
    if "boolq" in task:
        # {"question":..., "passage":..., "answer": True/False}
        passage = ex.get("passage", "")
        question = ex.get("question", "")
        prompt = f"{passage}\nQuestion: {question}?\nAnswer:"
        choices = ["No", "Yes"] # False=0, True=1 usually
        label = ex.get("answer", False)
        label_idx = 1 if label else 0
        return prompt, choices, label_idx

    elif "piqa" in task:
        # {"goal":..., "sol1":..., "sol2":..., "label": 0/1}
        prompt = ex.get("goal", "")
        choices = [ex.get("sol1", ""), ex.get("sol2", "")]
        label_idx = int(ex.get("label", 0))
        return prompt, choices, label_idx

    elif "siqa" in task or "social_i_qa" in task:
        # {"context":..., "question":..., "answerA":..., "answerB":..., "answerC":..., "label": "1"/"2"/"3"}
        ctx = ex.get("context", "")
        q = ex.get("question", "")
        prompt = f"{ctx}\n{q}\nAnswer:"
        choices = [ex.get("answerA",""), ex.get("answerB",""), ex.get("answerC","")]
        l = str(ex.get("label", "1"))
        label_idx = int(l) - 1
        return prompt, choices, label_idx

    elif "hellaswag" in task:
        # {"ctx":..., "endings": [str, str, str, str], "label": 0-3}
        prompt = ex.get("ctx", "")
        choices = ex.get("endings", [])
        label = ex.get("label", "")
        label_idx = int(label)
        return prompt, choices, label_idx

    elif "winogrande" in task:
        # {"sentence": "The trophy doesn't fit... because _ is too big", "option1":..., "option2":..., "answer": "1"/"2"}
        sent = ex.get("sentence", "")
        # Simple prompt strategy: just append the option to the sentence (replacing _ if possible)
        if "_" in sent:
             prompt = sent.split("_")[0] 
        else:
             prompt = sent
        choices = [ex.get("option1", ""), ex.get("option2", "")]
        l = str(ex.get("answer", "1"))
        label_idx = int(l) - 1
        return prompt, choices, label_idx

    elif "arc" in task: # arc_e, arc_c
        # {"question":..., "choices": {"text": [...], "label": ["A","B","C","D"]}, "answerKey": "A"}
        prompt = ex.get("question", "")
        c_dict = ex.get("choices", {})
        texts = c_dict.get("text", [])
        labels = c_dict.get("label", [])
        ans = ex.get("answerKey", "A")
        try:
            label_idx = labels.index(ans)
        except:
            label_idx = 0
        return prompt, texts, label_idx

    elif "obqa" in task or "openbookqa" in task:
        # {"question_stem":..., "choices": {"text":..., "label":...}, "answerKey":...}
        prompt = ex.get("question_stem", "")
        c_dict = ex.get("choices", {})
        texts = c_dict.get("text", [])
        labels = c_dict.get("label", [])
        ans = ex.get("answerKey", "A")
        try:
            label_idx = labels.index(ans)
        except:
            label_idx = 0
        return prompt, texts, label_idx

    return None, [], 0

def eval_commonsense(model, tokenizer, examples, task_name="boolq", device="cuda"):
    model.eval()
    correct = 0
    total = 0
    
    for ex in tqdm(examples, desc=f"Eval {task_name}"):
        prompt, choices, label_idx = parse_example(ex, task_name)
        if prompt is None or not choices: continue 
        
        scores = []
        for ch in choices:
            score = get_log_prob_of_continuation(model, tokenizer, prompt, " " + ch, device=device)
            scores.append(score)
            
        pred_idx = np.argmax(scores)
        if pred_idx == label_idx:
            correct += 1
        total += 1
        
    return correct / total if total > 0 else 0.0


