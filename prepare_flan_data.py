import os
import json
import random
import pandas as pd
from datasets import load_dataset

# Configuration
OUTPUT_DIR = "data/fed_tasks/flan_experiment"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Task definitions
# (Dataset Name, Subset Name, Split, Input Column(s), Output Column, Task Type)
TASKS = [
    # 1. Sentiment Analysis
    {"name": "sentiment140", "subset": None, "split": "train", "type": "sentiment"},
    # 2. NLI
    {"name": "snli", "subset": None, "split": "train", "type": "nli"},
    # 3. Text Classification
    {"name": "ag_news", "subset": None, "split": "train", "type": "topic"},
    # 4. Commonsense Reasoning (Using Hellaswag as Story Cloze is gated)
    {"name": "Rowan/hellaswag", "subset": None, "split": "train", "type": "commonsense"},
    # 5. Paraphrase Detection
    {"name": "glue", "subset": "mrpc", "split": "train", "type": "paraphrase"},
    # 6. Struct to Text
    {"name": "common_gen", "subset": None, "split": "train", "type": "struct2text"},
    # 7. Reading Comprehension
    {"name": "openbookqa", "subset": "main", "split": "train", "type": "qa"},
    # 8. Coreference Resolution
    {"name": "definite_pronoun_resolution", "subset": None, "split": "train", "type": "coref"},
]

SAMPLES_PER_CLIENT = 600
TEST_SAMPLES = 300

def format_example(example, task_type):
    inp, out = "", ""
    
    if task_type == "sentiment":
        # sentiment140: text, sentiment (0=neg, 2=neu, 4=pos)
        text = example["text"]
        label = example["sentiment"]
        inp = f"Classify the sentiment of the following tweet as positive or negative:\nTweet: {text}"
        # Map 0->Negative, 4->Positive. Drop neutral if needed or map. 
        # sentiment140 usually has 0 and 4.
        if label == 0: out = "Negative"
        elif label == 4: out = "Positive"
        else: out = "Neutral"
        
    elif task_type == "nli":
        # snli: premise, hypothesis, label (0: entailment, 1: neutral, 2: contradiction)
        prem = example["premise"]
        hyp = example["hypothesis"]
        label = example["label"]
        inp = f"Premise: {prem}\nHypothesis: {hyp}\nDoes the premise entail the hypothesis? Answer with Entailment, Neutral, or Contradiction."
        mapping = {0: "Entailment", 1: "Neutral", 2: "Contradiction"}
        if label in mapping:
            out = mapping[label]
        else:
            return None # Skip invalid labels
            
    elif task_type == "topic":
        # ag_news: text, label (0: World, 1: Sports, 2: Business, 3: Sci/Tech)
        text = example["text"]
        label = example["label"]
        inp = f"Classify the following news article into a category (World, Sports, Business, Sci/Tech):\nArticle: {text}"
        mapping = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}
        out = mapping[label]
        
    elif task_type == "commonsense":
        # hellaswag: ctx, endings (list), label (index)
        ctx = example["ctx"]
        endings = example["endings"]
        label = int(example["label"]) if isinstance(example["label"], str) else example["label"]
        # Change: Output text instead of index to match FedDPA style
        inp = f"Complete the story with the most plausible ending:\nContext: {ctx}\nOptions:\n- {endings[0]}\n- {endings[1]}\n- {endings[2]}\n- {endings[3]}"
        out = endings[label]
        
    elif task_type == "paraphrase":
        # glue/mrpc: sentence1, sentence2, label (0: no, 1: yes)
        s1 = example["sentence1"]
        s2 = example["sentence2"]
        label = example["label"]
        inp = f"Are the following two sentences paraphrases of each other?\nSentence 1: {s1}\nSentence 2: {s2}\nAnswer Yes or No."
        out = "Yes" if label == 1 else "No"
        
    elif task_type == "struct2text":
        # common_gen: concepts (list), target
        concepts = ", ".join(example["concepts"])
        target = example["target"]
        inp = f"Generate a sentence that includes all the following concepts: {concepts}"
        out = target
        
    elif task_type == "qa":
        # openbookqa: question_stem, choices (dict: text, label), answerKey
        q = example["question_stem"]
        choices = example["choices"] # {'text': [...], 'label': [...]}
        labels = choices['label']
        texts = choices['text']
        
        # Map answerKey (A,B,C,D) to index
        label_map = {l: i for i, l in enumerate(labels)}
        ans_idx = label_map.get(example["answerKey"], -1)
        
        options_str = "\n".join([f"- {t}" for t in texts])
        inp = f"Answer the question given the options:\nQuestion: {q}\nOptions:\n{options_str}"
        # Change: Output text instead of label letter
        if ans_idx != -1:
            out = texts[ans_idx]
        else:
            out = "" # Should not happen usually
        
    elif task_type == "coref":
        # definite_pronoun_resolution: sentence, pronoun, candidate (list), label (index)
        # This dataset structure: sentence, pronoun, candidates, label
        sent = example["sentence"]
        pronoun = example["pronoun"]
        candidates = example["candidates"]
        label = example["label"]
        
        c_str = ", ".join(candidates)
        inp = f"In the sentence: '{sent}'\nWho does the pronoun '{pronoun}' refer to? Candidates: {c_str}"
        out = candidates[label]

    return {"input": inp, "output": out}

def process_task(task_idx, task_info):
    print(f"Processing Task {task_idx}: {task_info['name']} ({task_info['type']})")
    try:
        dataset = load_dataset(task_info['name'], task_info['subset'], split=task_info['split'], trust_remote_code=True)
    except Exception as e:
        print(f"Error loading {task_info['name']}: {e}")
        return
    
    # Shuffle and select
    # We need SAMPLES_PER_CLIENT for training + TEST_SAMPLES for eval
    total_needed = SAMPLES_PER_CLIENT + TEST_SAMPLES
    
    # Convert to list to shuffle easily (dataset might be large, so take a larger slice then shuffle if needed, 
    # but for these datasets, they fit in memory usually. Or use dataset.shuffle())
    dataset = dataset.shuffle(seed=42)
    
    train_data = []
    test_data = []
    
    count = 0
    for ex in dataset:
        formatted = format_example(ex, task_info['type'])
        if formatted and formatted['output']: # Check validity
            if len(train_data) < SAMPLES_PER_CLIENT:
                train_data.append(formatted)
            elif len(test_data) < TEST_SAMPLES:
                test_data.append(formatted)
            else:
                break
    
    # Save Client Train Data
    client_file = os.path.join(OUTPUT_DIR, f"client_{task_idx}.json")
    with open(client_file, "w", encoding="utf-8") as f:
        json.dump(train_data, f, indent=2, ensure_ascii=False)
    print(f"  Saved {len(train_data)} train samples to {client_file}")
    
    # Save Test Data (We can aggregate these later or keep separate)
    # For now, let's save a separate test file for this client/task
    test_file = os.path.join(OUTPUT_DIR, f"test_{task_idx}.json")
    with open(test_file, "w", encoding="utf-8") as f:
        json.dump(test_data, f, indent=2, ensure_ascii=False)
    print(f"  Saved {len(test_data)} test samples to {test_file}")

def main():
    for i, task in enumerate(TASKS):
        process_task(i, task)
        
    # Create a global eval file combining all test sets
    global_eval = []
    for i in range(len(TASKS)):
        test_file = os.path.join(OUTPUT_DIR, f"test_{i}.json")
        if os.path.exists(test_file):
            with open(test_file, "r") as f:
                data = json.load(f)
                # Add task source info if needed, or just extend
                global_eval.extend(data)
    
    global_eval_path = os.path.join(OUTPUT_DIR, "global_eval.json")
    with open(global_eval_path, "w", encoding="utf-8") as f:
        json.dump(global_eval, f, indent=2, ensure_ascii=False)
    print(f"Saved Global Eval set with {len(global_eval)} samples to {global_eval_path}")

if __name__ == "__main__":
    main()
