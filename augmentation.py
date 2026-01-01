from intervention import intervention
import torch
from util import extract_steps, AnswerEOS
import random

FORMAT_INSTRUCTIONS = """IMPORTANT: Answer each question properly. Express your answer as either: a single number with no units, commas, or currency symbols; a single capital letter; or a single boolean with the first letter capitalized.

Q: If Alice has 3 apples and Bob gives her 2 more, how many apples does she have?
Reasoning:
Alice starts with 3 apples.
Bob gives Alice 2 additional apples.
Adding 3 and 2 gives the answer, 5.
Answer: 5

Q: If a rectangle has length 8 and width 5, what is its area?
(A) 30   (B) 35   (C) 40   (D) 45
Reasoning:
The formula for area of a rectangle is length × width.
The length is 8 and the width is 5.
Multiplying 8 and 5 gives the answer, 40.
Answer: C

Q: A train leaves at 3 PM and arrives at 6 PM. How many hours long is the trip?
Reasoning:
The train departs at 3 PM.
The train arrives at 6 PM.
The time difference between 3 PM and 6 PM, 3 hours, is the answer.
Answer: 3

Q: A factory produces 256 widgets per day. How many widgets will it produce in 365 days?
Reasoning:
The factory produces 256 widgets each day.
There are 365 days in the period.
Multiplying 256 × 365 = 93440 gives the total number of widgets produced.
Answer: 93440

Q: A store sells 10 vases a day. Each vase costs $20. How many dollar does it earn from vases each day?
Reasoning:
10 vases are sold each day.
The store earns $20 from each vase.
Multiplying 10 × 20 = 200 gives the total number of dollars earned.
Answer: 200

Q: The Earth orbits the Sun once every year. True or False?
Reasoning:
It is given that the Earth orbits the Sun.
The time for one complete orbit is 1 year.
This matches the statement in the question.
Answer: True
"""

@torch.no_grad()
def generate_cot_completion(prompt, partial_meta, model, tokenizer,
                             temperature=1.0, debug=0, edited_step=None):
    # build input
    lines = [FORMAT_INSTRUCTIONS, f"Q: {prompt}\nReasoning:"]

    if len(partial_meta):
        pm = partial_meta.copy()
        if edited_step is not None and pm:
            pm[-1] = edited_step
            if debug >= 1:
                print(f"[DEBUG1] Applied edited_step: {edited_step}")
        
        lines.extend(pm)

    lines.append("")
    input_text = "\n".join(lines)

    if debug >= 3:
        print(f"[DEBUG3] Input to model:\n{input_text}\n")

    # generate
    inputs = tokenizer([input_text], return_tensors="pt", truncation=False).to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=750,
        do_sample=temperature > 0,
        temperature=temperature,
        stopping_criteria=[AnswerEOS(tokenizer)],
        eos_token_id=None
    )
    decoded = tokenizer.batch_decode(outputs, skip_special_tokens=False)[0]
    if decoded.startswith("<s> "):
        decoded = decoded[4:]
    if decoded.startswith("<s>"):
        decoded = decoded[3:]
    decoded = decoded.replace("</s>", "\n\n")

    if debug >= 3:
        print(f"[DEBUG3] Model raw output:\n{decoded}\n")

    # isolate new block
    body = decoded[len(input_text):]
    body = body.split("\n\n")[0]

    if debug >= 3:
        print(f"[DEBUG3] Isolated block:\n{body}\n")

    return extract_steps(body)

# step_num is zero-indexed
def causal_importance(prompt, R_meta, a, step_num, model, tokenizer, temp=0.2, debug=0):
    original = R_meta[step_num]
    
    if debug >= 1:
        print(f"[DEBUG1] Checking step {step_num+1}: '{original}'")

    edited = intervention([original])[0]

    if debug >= 1:
        print(f"[DEBUG1] Edited step {step_num+1}: '{edited}'")

    trial = R_meta[:step_num+1]
    new_meta, new_ans = generate_cot_completion(
        prompt, trial, model, tokenizer,
        temperature=temp, debug=debug
    )

    if debug >= 2:
        print(f"[DEBUG2] New full answer trace: {R_meta[:step_num] + [edited] + new_meta}")

    if debug >= 1:
        print(f"[DEBUG1] New answer after edit: '{new_ans}' vs original '{a}'")

    return 1 - answer_probability(prompt, new_meta, a, model, tokenizer, debug=debug)[1]

# step_num is zero-indexed
def basic_causal_importance(prompt, R_meta, a, step_num, model, tokenizer, debug=0):
    original = R_meta[step_num]
    
    if debug >= 1:
        print(f"[DEBUG1] (Basic) Checking step {step_num+1}: '{original}'")

    edited = intervention([original])[0]

    if debug >= 1:
        print(f"[DEBUG1] Edited step {step_num+1}: '{edited}'")

    new_meta = R_meta.copy()
    new_meta[step_num] = edited

    if debug >= 2:
        print(f"[DEBUG2] New full answer trace: {new_meta}")

    return 1 - answer_probability(prompt, new_meta, a, model, tokenizer, debug=debug)[1]

@torch.no_grad()
def answer_probability(prompt, R_meta, a, model, tokenizer, debug=0):
    lines = [FORMAT_INSTRUCTIONS, f"Q: {prompt}\nReasoning:"]
    lines.extend(R_meta)
    lines.append("Answer:")
    input_text = "\n".join(lines)
    return answer_probability_raw(input_text, a, model, tokenizer, debug=debug)

@torch.no_grad()
def answer_probability_raw(input_text, a, model, tokenizer, debug=0):
    full_text = input_text + " " + str(a)
    enc_input = tokenizer(input_text, return_tensors="pt", truncation=False).to(model.device)
    enc_full = tokenizer(full_text, return_tensors="pt", truncation=False).to(model.device)

    input_len = enc_input.input_ids.shape[1]
    input_ids = enc_full.input_ids

    outputs = model(input_ids)
    logits = outputs.logits
    log_probs = torch.nn.functional.log_softmax(logits[:, :-1, :], dim=-1)
    target_ids = input_ids[:, 1:]
    token_logps = log_probs.gather(2, target_ids.unsqueeze(-1)).squeeze(-1)

    start_index = input_len - 1
    if start_index < 0:
        start_index = 0
    tail = token_logps[:, start_index:]

    if tail.numel() == 0:
        return 0.0, 1.0

    total_logp_tensor = tail.sum()
    total_logp = float(total_logp_tensor.cpu().item())
    prob = float(torch.exp(total_logp_tensor).cpu().item())
    if debug >= 1:
        print(f"[DEBUG] log-prob: {total_logp}, prob: {prob}")
    return total_logp, prob

def various_traces(prompt, R_meta, a, model, tokenizer, *, temperature=1.0, debug=0, n=10):
    base_log_prob, _ = answer_probability(prompt, R_meta, a, model, tokenizer, debug=debug)
    traces = []
    ii = random.sample(range(len(R_meta)), n) if n < len(R_meta) else range(len(R_meta))
    for i in ii:
        R_meta_mod = R_meta[:i] + [intervention([R_meta[i]])[0]]
        final, _ = generate_cot_completion(prompt, R_meta_mod, model, tokenizer, temperature=temperature, debug=debug)
        log_prob, _ = answer_probability(prompt, R_meta_mod, a, model, tokenizer, debug=debug)
        traces.append((R_meta_mod + final, base_log_prob - log_prob))
    return traces
