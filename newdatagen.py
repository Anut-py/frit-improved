import torch
from datasets import load_dataset
from copy import deepcopy
import random
import traceback
from tqdm import tqdm
from augmentation import various_traces, generate_cot_completion
from intervention import intervention
from model.model import load_aligned_model, load_tokenizer
from util import prompt_model_answer

model = None
tokenizer = None

dataset_sources = {
    "gsm8k": load_dataset("gsm8k", "main")["train"],
    "svamp": load_dataset("ChilleD/SVAMP")["train"],
    "strategyqa": load_dataset("ChilleD/StrategyQA")["train"],
    "commonsenseqa": load_dataset("ChilleD/CommonsenseQA")["train"],
    "scibench": load_dataset("xw27/scibench")["train"],
    "asdiv": load_dataset("nguyen-brat/asdiv")["train"],
}

def format_entry(entry, dataset_name):
    if dataset_name == "gsm8k":
        return entry["question"]
    elif dataset_name == "svamp":
        return entry["question_concat"]
    elif dataset_name == "strategyqa":
        return entry["facts"] + " " + entry["question"]
    elif dataset_name == "commonsenseqa":
        return entry["question_concat"]
    elif dataset_name == "scibench":
        return entry["problem_text"]
    elif dataset_name == "asdiv":
        return entry["question"]
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

def format_answer(entry, dataset_name):
    if dataset_name == "gsm8k":
        return entry["answer"].split("#### ")[1]
    elif dataset_name == "svamp":
        return entry["Answer"]
    elif dataset_name == "strategyqa":
        return entry["answer"]
    elif dataset_name == "commonsenseqa":
        return entry["answerKey"]
    elif dataset_name == "scibench":
        return entry["answer_number"]
    elif dataset_name == "asdiv":
        return entry["answer"][0].split(" ")[0]
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

def generate_preliminary_answer(prompt, temp, debug):
    return generate_cot_completion(prompt, [], model, tokenizer, temperature=temp, debug=debug)

def format_cot(cot):
    return "\n".join(cot[0] + [f'Answer: {cot[1]}'])

def make_sft_example(prompt, n=10, prelim_temp=0.5, various_temp=1.6, debug=0):
    try:
        global model, tokenizer
        if not model:
            model = load_aligned_model(trainable=False)
            tokenizer = load_tokenizer()

        preliminary = generate_preliminary_answer(prompt, prelim_temp, debug)
        samples = various_traces(prompt, preliminary[0], preliminary[1], model, tokenizer, temperature=various_temp, debug=debug, n=n)
        return {
            "prompt": prompt,
            "original": preliminary,
            "samples": samples
        }
    except Exception as e:
        print(traceback.format_exc())
        print(f"Making dpo example failed with error: {e}")
        return None
