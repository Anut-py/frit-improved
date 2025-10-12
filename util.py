from transformers import StoppingCriteria
import torch
import time
import re

def extract_steps(text):
    ls = text.split("\n")
    
    steps = []
    for l in ls:
        if not l.strip():
            continue
        if l.startswith("Answer: "):
            return (steps, l[8:].strip())
        steps.append(l.strip())
    return (steps, None)

class AnswerEOS(StoppingCriteria):
    def __init__(self, tokenizer, stopping_str="\n\n"):
        self.batch_size = -1
        self.seen = []
        self.total = 0
        self.tokenizer = tokenizer
        self.stopping_str = stopping_str
    
    def __call__(self, input_ids, scores, **kwargs):
        if self.batch_size == -1:
            self.batch_size = input_ids.shape[0]
            self.seen = [False] * self.batch_size
        for i in range(self.batch_size):
            if self.seen[i]: continue
            txt = self.tokenizer.decode(input_ids[i, -10:])
            if self.stopping_str in txt:
                self.seen[i] = True
                self.total += 1
        return self.total == self.batch_size

default_args = {"max_new_tokens": 2000, "do_sample": True, "temperature": 0.5}
@torch.no_grad()
def prompt_model_answer(prompts: list[str], model, tokenizer, *, debug: int = 0, **model_args):
    start = time.time()
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=False).to("cuda")
    end = time.time()
    if debug >= 2: print(f"tokenized inputs in {(end - start):0.3f}s")

    start = time.time()
    full_args = default_args | model_args
    outputs = model.generate(
        **inputs,
        **full_args,
        stopping_criteria=[AnswerEOS(tokenizer)],
        eos_token_id=None
    )
    end = time.time()
    if debug >= 2: print(f"generated outputs in {(end - start):0.3f}s")

    start = time.time()
    result = tokenizer.batch_decode(outputs[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    end = time.time()
    if debug >= 2: print(f"decoded outputs in {(end - start):0.3f}s")
    
    def extract(res):
        try:
            return res.split("Answer: ")[1].split("\n\n")[0].strip()
        except:
            return "ERROR"
    return [(r, extract(r)) for r in result]
