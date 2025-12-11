# data/format_fns.py

def format_metamathqa(item):
    # MetaMathQA: {"query": ..., "response": ...}
    prompt = item.get("query", "")
    target = item.get("response", "")
    return {"input": prompt, "output": target}

def format_codefeedback(item):
    # CodeFeedback: {"query": ..., "answer": ...}
    prompt = item.get("query", "")
    target = item.get("answer", "")
    return {"input": prompt, "output": target}

def format_commonsense(item):
    # Commonsense170K: {"instruction": ..., "output": ...}
    # Sometimes input is empty, sometimes not.
    instruction = item.get("instruction", "")
    inp = item.get("input", "")
    if inp:
        prompt = f"{instruction}\n{inp}"
    else:
        prompt = instruction
    target = item.get("output", "")
    return {"input": prompt, "output": target}
