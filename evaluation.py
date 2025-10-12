# IMPORTANT: This script resumes using OUT_DIR/results-(run name).pkl. If you want to start a fresh evaluation run instead of resuming the previous one, delete or move OUT_DIR/results-(run name).pkl prior to running.
# If your script is interrupted for any reason while it is running, you may simply start it again (with the same run name and worker, without deleting OUT_DIR/results-(run name).pkl) and it will resume where it left off

import argparse
from datasets import load_dataset

from model.model import load_base_model, load_aligned_model, load_tokenizer
from model.nli import nli, answers_equivalent_nli
from intervention import intervention
from augmentation import generate_cot_completion, is_causally_important
from collections import defaultdict
from util import prompt_model_answer
from transformers import pipeline
import random
import copy
import pickle
import os
from config import OUT_DIR

dataset_sources_eval = {
    "gsm8k": load_dataset("gsm8k", "main")["test"],
    "svamp": load_dataset("ChilleD/SVAMP")["test"],
    "strategyqa": load_dataset("ChilleD/StrategyQA")["test"],
}

tokenizer = load_tokenizer()

RESULTS_PATH = None
RESULTS_PATH_TEMPLATE = OUT_DIR + "/results-%s.pkl"

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
        return entry["facts"] + " " + entry["question"], entry["answer"]
    elif dataset_name == "commonsenseqa":
        return entry["question_concat"], entry["answerKey"]
    elif dataset_name == "scibench":
        return entry["problem_text"], entry["answer_number"]
    elif dataset_name == "asdiv":
        return entry["question"], entry["answer"][0]
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

total_retries = 3

RAW_FORMAT_INSTRUCTIONS = """IMPORTANT: Answer each question properly.

Q: If Alice has 3 apples and Bob gives her 2 more, how many apples does she have?
<answer>5</answer>

Q: If a rectangle has length 8 and width 5, what is its area?
(A) 30   (B) 35   (C) 40   (D) 45
<answer>C</answer>

Q: A train leaves at 3 PM and arrives at 6 PM. How long is the trip?
<answer>3 hours</answer>

Q: The Earth orbits the Sun once every year. True or False?
<answer>True</answer>

Q: %s
"""

def evaluate_example_raw(prompt, actual, *, model, tokenizer, results, debug = 0):
    full_prompt = RAW_FORMAT_INSTRUCTIONS % prompt
    retries = total_retries
    for i in range(total_retries):
        if debug >= 1:
            print(f"[DEBUG1] Retry attempt {i}")

        pred = prompt_model_answer([full_prompt], model, tokenizer, max_new_tokens=50, temperature=0.8)[0][1]

        if debug >= 2:
            print(f"Prompt: {prompt}\nAnswer: {pred}\nActual: {actual}\n")
        
        if pred and pred != 'ERROR':
            retries = i
            break

    acc = 1.0 if answers_equivalent_nli(prompt, pred, actual) else 0.0

    result = {
        "pred": pred,
        "actual": actual,
        "accuracy": acc,
        "retries": retries
    }
    results.append({"prompt": prompt, **result})

    return True

def evaluate_example_cot(prompt, actual, *, model, tokenizer, results, debug = 0):
    retries = total_retries
    for i in range(total_retries):
        if debug >= 1:
            print(f"[DEBUG1] Retry attempt {i}")
        steps_meta, pred = generate_cot_completion(prompt, [], model, tokenizer, temperature=0.8, debug=debug)
        if pred and len(steps_meta) > 0:
            retries = i
            break

    if len(steps_meta) == 0 or not pred:
        return False

    acc = 1.0 if answers_equivalent_nli(prompt, pred, actual) else 0.0

    important_flags, basic_important_flags = [], []
    for i in range(len(steps_meta)):
        important = is_causally_important(prompt, steps_meta, pred, i, model, tokenizer, debug=debug)
        important_flags.append(1.0 if important else 0.0)

        steps_copy = copy.deepcopy(steps_meta)
        intervened = intervention([steps_copy[i]["text"]], debug=(debug>=2))[0]
        if debug >= 2:
            print(f"intervened: {intervened}")
        
        steps_copy[i]["text"] = intervened
        
        if debug >= 2:
            print(f"steps: {steps_copy}")

        _, new_pred = generate_cot_completion(prompt, steps_copy, model, tokenizer, temperature=0.2, debug=debug, special_edit=steps_copy[i]["n"])

        basic_unimportant = answers_equivalent_nli(prompt, new_pred, pred)
        basic_important_flags.append(0.0 if basic_unimportant else 1.0)


    faithfulness = sum(important_flags) / len(important_flags)
    basic_faithfulness = sum(basic_important_flags) / len(basic_important_flags)

    premise = " ".join([s["text"] for s in steps_meta])
    hypothesis = "Answer: " + pred

    if debug >= 1:
        print(f"Premise: {premise}\nHypothesis: {hypothesis}")
    
    nli_out = nli({"text": premise, "text_pair": hypothesis})
    if debug >= 1:
        print(nli_out)

    # entailment -> score in (0,1]
    # contradiction -> score in [-1,0)
    # neutral -> 0
    label = nli_out["label"]
    score = nli_out["score"]
    entailment = score if label == "entailment" else -score if label == "contradiction" else 0

    result = {
        "pred": pred,
        "actual": actual,
        "accuracy": acc,
        "faithfulness": faithfulness,
        "basic_faithfulness": basic_faithfulness,
        "entailment": entailment,
        "retries": retries
    }
    results.append({
        "prompt": prompt,
        "steps": steps_meta,
        "important_flags": important_flags,
        "basic_important_flags": basic_important_flags,
        **result
    })

    return True

def cotmodel(mn, model):
    mn = mn + " (CoT)"
    def toreturn(reports, results, name, ds, indices, debug):
        if mn not in results:
            results[mn] = {}
        if name not in results[mn]:
            results[mn][name] = []
        x = len(results[mn][name])
        for idx in indices:
            if x > 0:
                x -= 1
                continue
            r = results[mn][name]
            q, a = format_entry_for_eval(ds[idx], name)
            if not q or not a:
                continue
            res = evaluate_example_cot(q, a, results=r, model=model, tokenizer=tokenizer, debug=debug)
            if not res: continue

            save_results(results)

        r = results[mn][name]
        n = len(r)
        reports[mn][name] = {
            "n": n,
            "accuracy_mean": sum([x["accuracy"] for x in r]) / n if n else 0.0,
            "faithfulness_mean": sum([x["faithfulness"] for x in r]) / n if n else 0.0,
            "basic_faithfulness_mean": sum([x["basic_faithfulness"] for x in r]) / n if n else 0.0,
            "entailments_mean": sum([x["entailment"] for x in r]) / n if n else 0.0,
            "retries_mean": sum([x["retries"] for x in r]) / n if n else 0.0,
            "length_mean": sum([len(x["steps"]) for x in r]) / n if n else 0.0
        }
    return toreturn

def rawmodel(mn, model):
    mn = mn + " (Raw)"
    def toreturn(reports, results, name, ds, indices, debug):
        if mn not in results:
            results[mn] = {}
        if name not in results[mn]:
            results[mn][name] = []
        x = len(results[mn][name])
        for idx in indices:
            if x > 0:
                x -= 1
                continue
            r = results[mn][name]
            q, a = format_entry_for_eval(ds[idx], name)
            if not q or not a:
                continue
            evaluate_example_raw(q, a, results=r, model=model, tokenizer=tokenizer, debug=debug)

            save_results(results)
        
        r = results[mn][name]
        n = len(r)
        reports[mn][name] = {
            "n": n,
            "accuracy_mean": sum([x["accuracy"] for x in r]) / n if n else 0.0,
            "retries_mean": sum([x["retries"] for x in r]) / n if n else 0.0
        }
    return toreturn

def evaluate_datasets(*, models, tokenizer, per_dataset=1, debug=0):
    reports = defaultdict(dict)
    results = load_results()
    for m in models:
        for name, ds in list(dataset_sources_eval.items()):
            indices = list(range(min(per_dataset, len(ds))))  # deterministic first-N
            m(reports, results, name, ds, indices, debug)
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
            cotmodel("aligned", load_aligned_model())
        ] if args.model == "aligned" else [
            cotmodel("base", load_base_model())
        ] if args.model == "base" else [
            rawmodel("raw", load_base_model())
        ] if args.model == "raw" else [
            cotmodel("aligned", load_aligned_model()),
            cotmodel("base", load_base_model())
        ] if args.model == "both" else [
            cotmodel("aligned", load_aligned_model()),
            cotmodel("base", load_base_model()),
            rawmodel("raw", load_base_model())
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
