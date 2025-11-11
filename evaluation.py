# IMPORTANT: This script resumes using OUT_DIR/results-(run name).pkl. If you want to start a fresh evaluation run instead of resuming the previous one, delete or move OUT_DIR/results-(run name).pkl prior to running.
# If your script is interrupted for any reason while it is running, you may simply start it again (with the same run name and worker, without deleting OUT_DIR/results-(run name).pkl) and it will resume where it left off

import argparse
from datasets import load_dataset

from model.model import load_base_model, load_aligned_model, load_tokenizer
from intervention import intervention
from augmentation import generate_cot_completion, causal_importance, basic_causal_importance, answer_probability, answer_probability_raw
from collections import defaultdict
from util import prompt_model_answer
from transformers import pipeline
import random
import copy
import pickle
import os
from config import out_subdir

dataset_sources_eval = {
    "gsm8k": load_dataset("gsm8k", "main")["test"],
    "svamp": load_dataset("ChilleD/SVAMP")["test"],
    "strategyqa": load_dataset("ChilleD/StrategyQA")["test"],
}

tokenizer = load_tokenizer()

RESULTS_PATH = None
RESULTS_PATH_TEMPLATE = out_subdir + "/results-%s.pkl"

def load_results():
    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH, "rb") as f:
            return pickle.load(f)
    else:
        return {}

def save_results(results):
    with open(RESULTS_PATH, "wb") as f:
        pickle.dump(results, f)

def format_entry_for_eval(entry, dataset_name):
    if dataset_name == "gsm8k":
        return entry["question"], entry["answer"].split("#### ")[1]
    elif dataset_name == "svamp":
        return entry["question_concat"], entry["Answer"]
    elif dataset_name == "strategyqa":
        return entry["facts"] + " " + entry["question"], str(entry["answer"])
    elif dataset_name == "commonsenseqa":
        return entry["question_concat"], entry["answerKey"]
    elif dataset_name == "scibench":
        return entry["problem_text"], entry["answer_number"]
    elif dataset_name == "asdiv":
        return entry["question"], entry["answer"][0]
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

total_retries = 3

RAW_FORMAT_INSTRUCTIONS = """IMPORTANT: Answer each question properly. Express your answer as either: a single number with no units, commas, or currency symbols; a single capital letter; or a single boolean with the first letter capitalized.

Q: If Alice has 3 apples and Bob gives her 2 more, how many apples does she have?
Answer: 5

Q: If a rectangle has length 8 and width 5, what is its area?
(A) 30   (B) 35   (C) 40   (D) 45
Answer: C

Q: A train leaves at 3 PM and arrives at 6 PM. How many hours long is the trip?
Answer: 3

Q: A factory produces 256 widgets per day. How many widgets will it produce in 365 days?
Answer: 93440

Q: A store sells 10 vases a day. Each vase costs $20. How many dollar does it earn from vases each day?
Answer: 200

Q: The Earth orbits the Sun once every year. True or False?
Answer: True

Q: %s
Answer:"""

def evaluate_example_raw(prompt, actual, *, model, tokenizer, results, debug = 0):
    full_prompt = RAW_FORMAT_INSTRUCTIONS % prompt

    confidence = answer_probability_raw(full_prompt, actual, model, tokenizer)[1]

    result = {
        "actual": actual,
        "confidence": confidence,
        "adjusted_accuracy": confidence > 0.50
    }
    results.append({"prompt": prompt, **result})

    return True

def evaluate_example_cot(prompt, actual, *, model, tokenizer, results, debug = 0):
    retries = total_retries
    for i in range(total_retries):
        if debug >= 1:
            print(f"[DEBUG1] Retry attempt {i}")
        steps_meta, pred = generate_cot_completion(prompt, [], model, tokenizer, temperature=0, debug=debug)
        if pred and len(steps_meta) > 0:
            retries = i
            break

    if len(steps_meta) == 0 or not pred:
        return False

    metrics = ["faithfulness", "basic_faithfulness"]
    metrics_dict = {k: [] for k in metrics}
    for i in range(len(steps_meta)):
        faithfulness = causal_importance(prompt, steps_meta, pred, i, model, tokenizer, temp=0, debug=debug)
        metrics_dict["faithfulness"].append(faithfulness)

        basic_faithfulness = basic_causal_importance(prompt, steps_meta, pred, i, model, tokenizer, debug=debug)
        metrics_dict["basic_faithfulness"].append(basic_faithfulness)

    confidence = answer_probability(prompt, steps_meta, actual, model, tokenizer, debug=debug)[1]

    result = {
        "pred": pred,
        "actual": actual,
        "confidence": confidence,
        "raw_accuracy": 1 if pred == actual else 0,
        "adjusted_accuracy": confidence > 0.50,
        "retries": retries,
        **{k: sum(metrics_dict[k]) / len(metrics_dict[k]) for k in metrics}
    }
    results.append({
        "prompt": prompt,
        "steps": steps_meta,
        **{(k + "_steps"): metrics_dict[k] for k in metrics},
        **result
    })

    return True

def cotmodel(mn):
    mn = mn + " (CoT)"
    def toreturn(model, reports, results, name, ds, indices, debug):
        if mn not in results:
            results[mn] = {}
        if name not in results[mn]:
            results[mn][name] = []
        r = results[mn][name]
        x = len(r)
        for idx in indices:
            if x > 0:
                x -= 1
                continue
            q, a = format_entry_for_eval(ds[idx], name)
            if not q or not a:
                continue
            res = evaluate_example_cot(q, a, results=r, model=model, tokenizer=tokenizer, debug=debug)
            if not res: continue

            save_results(results)

        n = len(r)
        metrics = ["confidence", "raw_accuracy", "adjusted_accuracy", "retries", "faithfulness", "basic_faithfulness"]
        reports[mn][name] = {
            "n": n,
            "length_mean": sum([len(x["steps"]) for x in r]) / n if n else 0.0,
            **{(k + "_mean"): sum([x[k] for x in r]) / n if n else 0.0 for k in metrics}
        }
    return toreturn

def rawmodel(mn):
    mn = mn + " (Raw)"
    def toreturn(model, reports, results, name, ds, indices, debug):
        if mn not in results:
            results[mn] = {}
        if name not in results[mn]:
            results[mn][name] = []
        r = results[mn][name]
        x = len(r)
        for idx in indices:
            if x > 0:
                x -= 1
                continue
            q, a = format_entry_for_eval(ds[idx], name)
            if not q or not a:
                continue
            evaluate_example_raw(q, a, results=r, model=model, tokenizer=tokenizer, debug=debug)

            save_results(results)

        n = len(r)
        reports[mn][name] = {
            "n": n,
            "confidence_mean": sum([x["confidence"] for x in r]) / n if n else 0.0,
            "adjusted_accuracy_mean": sum([x["adjusted_accuracy"] for x in r]) / n if n else 0.0
        }
    return toreturn

def evaluate_datasets(*, models, tokenizer, per_dataset=1, debug=0):
    reports = defaultdict(dict)
    results = load_results()
    for m, loader in models:
        model = loader()
        for name, ds in list(dataset_sources_eval.items()):
            indices = list(range(min(per_dataset, len(ds))))  # deterministic first-N
            m(model, reports, results, name, ds, indices, debug)
        del model
    return reports

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["base", "aligned", "raw", "both", "all"], default="all") # 'both' is base and aligned, 'all' is base, aligned, and raw
    ap.add_argument("--per_dataset", type=int, default=5) # Use 1 or 5 to test if it works, then set to a high number for the actual run
    ap.add_argument("--wandb", action="store_true") # Enable to log this run to WandB (if enabled, run `wandb login [your_token]` prior to running this script)
    ap.add_argument("--run_name", type=str, default="eval-run") # Run name in WandB
    ap.add_argument("--debug", type=int, default=0) # Set 0-3 for different amounts of logs
    ap.add_argument("--worker", type=str, default="") # For running in parallel; can be omitted
    args = ap.parse_args()

    global RESULTS_PATH
    RESULTS_PATH = RESULTS_PATH_TEMPLATE % (args.run_name + args.worker)

    models = [
            (cotmodel("aligned"), load_aligned_model)
        ] if args.model == "aligned" else [
            (cotmodel("base"), load_base_model)
        ] if args.model == "base" else [
            (rawmodel("raw"), load_base_model)
        ] if args.model == "raw" else [
            (cotmodel("aligned"), load_aligned_model),
            (cotmodel("base"), load_base_model)
        ] if args.model == "both" else [
            (cotmodel("aligned"), load_aligned_model),
            (cotmodel("base"), load_base_model),
            (rawmodel("raw"), load_base_model)
        ]

    run = None
    if args.wandb:
        import wandb
        run = wandb.init(project="frit", name=args.run_name + args.worker, config={
            "eval_per_dataset": args.per_dataset,
            "model": args.model
        })

    reports = evaluate_datasets(models=models, tokenizer=tokenizer,
                               per_dataset=args.per_dataset, debug=args.debug)

    print("\n=== EVAL SUMMARY ===")
    for model, report in reports.items():
        print(f"Results for {model} model")
        for k, v in report.items():
            print(f"{k}: ", end="")
            print(" | ".join([f"{kk}={vv}" for kk, vv in v.items()]))
    if run:
        run.finish()

if __name__ == "__main__":
    main()
