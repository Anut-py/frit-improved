import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--worker', type=int, help='ID of this worker')
args = parser.parse_args()
worker = args.worker

if worker is None:
    print("worker flag is required")
    sys.exit(1)

import os
import ijson
import json
from glob import glob
from transformers import PreTrainedModel, PreTrainedTokenizer, StoppingCriteria
import torch
import time
from datetime import datetime

from pathlib import Path
import functools
import math
from tqdm import tqdm

def log(x):
    now = datetime.now()
    formatted = now.strftime("%Y-%m-%d %H:%M:%S")
    print(f"{formatted} {x}", flush=True)

# add the parent (or any) directory to sys.path
parent = Path().resolve().parent  # one level up
sys.path.insert(0, str(parent))

from model.model import load_tokenizer, load_base_model

tokenizer = load_tokenizer()
base_model = load_base_model(True)

wikidata_dir = "/workspace/corpora/wikidata"

def iterate_labels(func, batch_size=1, offset=0, max_count=None):
    pattern = os.path.join(wikidata_dir, 'labels', '*.jsonl')
    count = 0
    batch = []
    batch_num = 0
    for filepath in sorted(glob(pattern)):
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                if count < offset:
                    count += 1
                    if count % batch_size == 0:
                        batch_num += 1
                    continue
                rec = json.loads(line)
                batch.append((count, rec))
                count += 1

                if len(batch) >= batch_size:
                    func(batch, batch_num)
                    batch = []
                    batch_num += 1

                if max_count and count >= max_count:
                    if len(batch):
                        func(batch, batch_num)
                    return
    if len(batch):
        func(batch, batch_num)

import urllib.request
import urllib.parse
import json
import gc
            

query = """
SELECT ?subjectLabel ?propertyLabel ?propertyDescription ?objectLabel WHERE {
  VALUES ?subject { %s }
  
  ?subject ?property ?object .
  FILTER(STRSTARTS(STR(?object), STR(wd:Q)))
    
  ?subject rdfs:label ?subjectLabel .
  FILTER(LANG(?subjectLabel) = "en")
  
  ?p wikibase:directClaim ?property .
  FILTER(STRSTARTS(STR(?property), STR(wdt:P)))
  ?p rdfs:label ?propertyLabel .
  FILTER(?p != wd:P1889) # Don't want "different from" relations because those clutter up the data
  FILTER(LANG(?propertyLabel) = "en")
  OPTIONAL {
    ?p schema:description ?propertyDescription .
    FILTER(LANG(?propertyDescription) = "en")
  }

  ?object rdfs:label ?objectLabel .
  FILTER(LANG(?objectLabel) = "en")
}
"""

def get_qid_relations(qids):
    qid_str = " ".join(["wd:" + qid for qid in qids])
    full_query = query % qid_str
    
    url = "https://query.wikidata.org/sparql"

    data = urllib.parse.urlencode({"query": full_query}).encode("utf-8")
    headers = {
        "Accept": "application/sparql-results+json",
        "Content-Type": "application/x-www-form-urlencoded"
    }

    try:
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")
        with urllib.request.urlopen(req) as resp:
            data = json.load(resp)
            return [{
                    "subject": b["subjectLabel"]["value"],
                    "property": b["propertyLabel"]["value"],
                    "propertyDescription": b["propertyDescription"]["value"] if "propertyDescription" in b else None,
                    "object": b["objectLabel"]["value"]
                } for b in data["results"]["bindings"]]
    except:
        return []

class AnswerEOS(StoppingCriteria):
    def __init__(self):
        self.batch_size = -1
        self.seen = []
        self.total = 0
    
    def __call__(self, input_ids, scores, **kwargs):
        if self.batch_size == -1:
            self.batch_size = input_ids.shape[0]
            self.seen = [False] * self.batch_size
        for i in range(self.batch_size):
            if self.seen[i]: continue
            txt = tokenizer.decode(input_ids[i, -10:])
            if "</answer>" in txt:
                self.seen[i] = True
                self.total += 1
        return self.total == self.batch_size

def most_mem_device():
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    gc.collect()
    
    max_free_mem = 0
    best_device = None
    
    for i in range(torch.cuda.device_count()):
        free_mem, _ = torch.cuda.mem_get_info(i)
        if free_mem > max_free_mem:
            max_free_mem = free_mem
            best_device = i

    return best_device

@torch.no_grad()
def prompt_model(prompts: list[str], max_new_tokens: int, debug: bool = False):
    start = time.time()
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=False).to(f"cuda:{most_mem_device()}")
    end = time.time()
    if debug: log(f"tokenized inputs in {(end - start):0.3f}s")

    start = time.time()
    outputs = base_model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        stopping_criteria=[AnswerEOS()],
        eos_token_id=None,
        temperature=None,
        top_p=None,
        top_k=None
    )
    end = time.time()
    if debug: log(f"generated outputs in {(end - start):0.3f}s")

    start = time.time()
    result = tokenizer.batch_decode(outputs[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    end = time.time()
    if debug: log(f"decoded outputs in {(end - start):0.3f}s")

    def extract(res):
        try:
            return res.split("<answer>")[1].split("</answer>")[0].strip()
        except:
            return "ERROR"
    return [(r, extract(r)) for r in result]

to_sentence_prompt_template = """You will be given a subject, a property, a description of that property, and an object, all referring to a Wikidata statement. Think step-by-step about how to turn them into a natural, fluent English sentence that conveys the meaning of the statement. Then write the final sentence, wrapped in <answer>...</answer> tags. Do not include any extra commentary.

Subject: %s  
Property: %s  
Property Description: %s  
Object: %s

Step-by-step reasoning:"""

to_sentence_no_desc_prompt_template = """You will be given a subject, a property, and an object, all referring to a Wikidata statement. Think step-by-step about how to turn them into a natural, fluent English sentence that conveys the meaning of the statement. Then write the final sentence, wrapped in <answer>...</answer> tags. Do not include any extra commentary.

Subject: %s  
Property: %s  
Object: %s

Step-by-step reasoning:"""

read_batch_size = 200
prompt_batch_size = 80
completed_file = f"completed_relations_{worker}"
out_file = f"data/relations/relations_{worker}.log"

@torch.no_grad()
def iteration(xs, i):
    log(f"starting batch {i}")
    
    rels = get_qid_relations([x[1]['qid'] for x in xs])

    log(f"fetched relations for batch {i}")
    
    if len(rels) == 0:
        log(f"batch has no relations, skipping")
        return

    def gen_prompt(result):
        if result["propertyDescription"] is None:
            return to_sentence_no_desc_prompt_template % (result["subject"], result["property"], result["object"])
        else:
            return to_sentence_prompt_template % (result["subject"], result["property"], result["propertyDescription"], result["object"])

    prompts = [gen_prompt(result) for result in rels]
    batched_prompts = [
        prompts[(prompt_batch * prompt_batch_size):((prompt_batch + 1) * prompt_batch_size)]
        for prompt_batch in range(math.ceil(len(prompts) / prompt_batch_size))]

    log(f"number of prompts: {len(prompts)}")
    # log(f"got batched prompts: {batched_prompts}")
    log(f"number of prompt batches: {len(batched_prompts)}")

    written = 0
    for batch in tqdm(batched_prompts):
        ans = prompt_model(batch, 200, False)

        with open(out_file, "a") as f:
            for _, sentence in ans:
                if sentence == "ERROR" or len(sentence) <= 10 or len(sentence) >= 300 or not (sentence.endswith(".") or sentence.endswith(".\"")):
                    continue
                f.write(f"{sentence}\n")
                written += 1
    log(f"wrote {written} lines")

    completed = xs[-1][0] + 1

    with open(completed_file, "w") as f:
        f.write(str(completed))

    log(f"done with batch {i}")

log(f"starting as worker {worker}")
with open(completed_file, "r") as f:
    completed = int(f.readline().strip())
    log(f"offset is {completed}")

iterate_labels(iteration, read_batch_size, offset=completed)
