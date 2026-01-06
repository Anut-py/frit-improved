#!/usr/bin/env python3
import builtins
from datetime import datetime
import pytz

eastern = pytz.timezone('US/Eastern')

print = lambda *args, **kwargs: builtins.print(
    f"[{datetime.now(eastern).strftime('%Y-%m-%d %H:%M:%S')}]",
    *args,
    **{**kwargs, "flush": True}
)

import os
import pickle
import time
import random
import fcntl
from multiprocessing import Process, set_start_method
import json
import math
import torch, gc
import tempfile
import shutil
import logging

LOG = logging.getLogger(__name__)

# user config should provide these
from config import GPUS, PER_GPU, BATCH_SIZE, EPOCHS, LR, GRAD_ACCUM_STEPS, MAX_LENGTH, KL_LAMBDA, GEN_EPOCHS, TARGET_EXAMPLES, out_subdir

# Constants and paths
MONITOR_POLL_SECONDS = 5   # how often main prints datagen progress while workers run

PER_DATASET = math.ceil(TARGET_EXAMPLES / 6)

N_WORKERS = GPUS * PER_GPU
GPU_IDS = list(range(GPUS))
DEBUG = 2

LOCK_RETRY_DELAY = 0.05
LOCK_RETRY_ATTEMPTS = 5
PICKLE_PATH = os.path.join(out_subdir, "datagen.pkl")
PICKLE_ARCHIVE_PATH = os.path.join(out_subdir, "datagen%d.pkl")
EPOCH_PATH = os.path.join(out_subdir, "current_epoch")

SAMPLE_ENTRIES_PATH = os.path.join(out_subdir, "sample_entries.pkl")
WORK_QUEUE_PATH = os.path.join(out_subdir, "work_queue.pkl")

# Utility: free CUDA cache
def clear_cuda():
    gc.collect()
    try:
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    except Exception:
        pass

# ---------- robust pickle helpers ----------
def safe_read_pickle(path):
    """Read pickle, but if file is corrupted try .bak before giving up."""
    try:
        with open(path, "rb") as f:
            try:
                fcntl.flock(f, fcntl.LOCK_SH)
            except Exception:
                pass
            try:
                data = pickle.load(f)
            except EOFError:
                LOG.warning("%s appears truncated/empty; attempting to load backup", path)
                try:
                    fcntl.flock(f, fcntl.LOCK_UN)
                except Exception:
                    pass
                bak = path + ".bak"
                if os.path.exists(bak):
                    with open(bak, "rb") as fb:
                        try:
                            fcntl.flock(fb, fcntl.LOCK_SH)
                        except Exception:
                            pass
                        try:
                            data = pickle.load(fb)
                            LOG.warning("Loaded backup %s instead", bak)
                        except Exception:
                            LOG.error("Backup %s also failed to load", bak)
                            data = []
                        finally:
                            try:
                                fcntl.flock(fb, fcntl.LOCK_UN)
                            except Exception:
                                pass
                else:
                    data = []
            finally:
                try:
                    fcntl.flock(f, fcntl.LOCK_UN)
                except Exception:
                    pass
        return data
    except FileNotFoundError:
        return []
    except Exception:
        LOG.exception("Unexpected error reading %s", path)
        bak = path + ".bak"
        if os.path.exists(bak):
            try:
                with open(bak, "rb") as fb:
                    try:
                        fcntl.flock(fb, fcntl.LOCK_SH)
                    except Exception:
                        pass
                    try:
                        data = pickle.load(fb)
                        LOG.warning("Recovered from backup %s after read error", bak)
                        return data
                    finally:
                        try:
                            fcntl.flock(fb, fcntl.LOCK_UN)
                        except Exception:
                            pass
            except Exception:
                LOG.exception("Backup also failed: %s", bak)
        return []

def atomic_write_pickle(path, obj):
    """Write pickle atomically (temp -> replace), fsync directory."""
    dirpath = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(dirpath, exist_ok=True)
    fd, tmppath = tempfile.mkstemp(dir=dirpath, prefix=os.path.basename(path) + ".tmp.")
    try:
        with os.fdopen(fd, "wb") as tf:
            try:
                # no lock needed on temp
                pickle.dump(obj, tf)
                tf.flush()
                os.fsync(tf.fileno())
            finally:
                pass
        # move temp -> path atomically
        os.replace(tmppath, path)
        # fsync directory to make rename durable
        try:
            dirfd = os.open(dirpath, os.O_DIRECTORY)
            os.fsync(dirfd)
            os.close(dirfd)
        except Exception:
            pass
    finally:
        if os.path.exists(tmppath):
            try:
                os.remove(tmppath)
            except Exception:
                pass

def safe_append_pickle(path, item):
    """
    Robust append:
      - read current (shared/exclusive locking)
      - create temp file and write new list to it
      - fsync temp, then atomically replace target (os.replace)
      - keep a .bak of previous file in case of failure
    Returns new length.
    """
    # Read current list under shared lock first to minimize exclusive lock time
    current = []
    if os.path.exists(path):
        try:
            with open(path, "rb") as rf:
                try:
                    fcntl.flock(rf, fcntl.LOCK_SH)
                except Exception:
                    pass
                try:
                    try:
                        current = pickle.load(rf)
                    except EOFError:
                        # truncated primary; try backup
                        current = safe_read_pickle(path)
                finally:
                    try:
                        fcntl.flock(rf, fcntl.LOCK_UN)
                    except Exception:
                        pass
        except FileNotFoundError:
            current = []

    # Append the new item
    current.append(item)

    # Write to a temp file in same directory
    dirpath = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(dirpath, exist_ok=True)
    fd, tmppath = tempfile.mkstemp(dir=dirpath, prefix=os.path.basename(path) + ".tmp.")
    try:
        with os.fdopen(fd, "wb") as tf:
            try:
                try:
                    fcntl.flock(tf, fcntl.LOCK_EX)
                except Exception:
                    pass
                pickle.dump(current, tf)
                tf.flush()
                os.fsync(tf.fileno())
                try:
                    fcntl.flock(tf, fcntl.LOCK_UN)
                except Exception:
                    pass
            finally:
                pass

        # Before replacing, optionally make a backup of the current file
        bak = path + ".bak"
        if os.path.exists(path):
            try:
                tmpbak = bak + f".tmp.{os.getpid()}"
                os.replace(path, tmpbak)
                os.replace(tmpbak, bak)
            except Exception:
                # fallback: try best-effort copy
                try:
                    shutil.copy2(path, bak)
                except Exception:
                    LOG.exception("Failed to create backup %s; continuing", bak)

        # Atomically replace the target with the temp file
        os.replace(tmppath, path)
        # fsync directory to ensure rename is durable
        try:
            dirfd = os.open(dirpath, os.O_DIRECTORY)
            os.fsync(dirfd)
            os.close(dirfd)
        except Exception:
            pass

        return len(current)
    finally:
        # ensure no stray temp
        if os.path.exists(tmppath):
            try:
                os.remove(tmppath)
            except Exception:
                pass

# ---------- on-disk sample entries + work-queue helpers ----------
def save_sample_entries_if_missing(path, entries):
    """If sample entries file exists, load & return it. Otherwise save entries atomically and return them."""
    if os.path.exists(path):
        try:
            with open(path, "rb") as f:
                try:
                    fcntl.flock(f, fcntl.LOCK_SH)
                except Exception:
                    pass
                try:
                    data = pickle.load(f)
                    return data
                finally:
                    try:
                        fcntl.flock(f, fcntl.LOCK_UN)
                    except Exception:
                        pass
        except Exception:
            LOG.exception("Failed to load existing sample entries; will overwrite")
    # write atomically
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    atomic_write_pickle(path, entries)
    return entries

def init_work_queue_if_missing(path, n):
    """Create work queue with indices [0..n-1] if not already present."""
    if os.path.exists(path):
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    indices = list(range(n))
    random.shuffle(indices)
    atomic_write_pickle(path, indices)

def pop_work_index(path):
    """Atomically pop and return an index from work-queue (or None if empty)."""
    if not os.path.exists(path):
        return None
    try:
        with open(path, "rb+") as f:
            try:
                fcntl.flock(f, fcntl.LOCK_EX)
            except Exception:
                pass
            try:
                f.seek(0)
                try:
                    data = pickle.load(f)
                except EOFError:
                    data = []
                if not data:
                    return None
                idx = data.pop()
                # rewrite truncated file
                f.seek(0)
                f.truncate()
                pickle.dump(data, f)
                f.flush()
                os.fsync(f.fileno())
                return idx
            finally:
                try:
                    fcntl.flock(f, fcntl.LOCK_UN)
                except Exception:
                    pass
    except FileNotFoundError:
        return None
    except Exception:
        LOG.exception("pop_work_index failed on %s; falling back", path)
        data = safe_read_pickle(path)
        if not data:
            return None
        idx = data.pop()
        atomic_write_pickle(path, data)
        return idx

# ---------- simple atomic integer file helpers ----------
def read_number(path: str) -> int:
    """Read an integer from a file. Returns 0 if missing or invalid."""
    if not os.path.exists(path):
        return 0
    try:
        with open(path, "r") as f:
            try:
                fcntl.flock(f, fcntl.LOCK_SH)
            except Exception:
                pass
            try:
                data = f.read().strip()
                return int(data) if data else 0
            finally:
                try:
                    fcntl.flock(f, fcntl.LOCK_UN)
                except Exception:
                    pass
    except Exception:
        return 0

def increment_number(path: str, delta: int = 1) -> int:
    """Atomically increment the number in the file by delta. Returns new value."""
    current = read_number(path)
    new_val = current + delta
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX)
        except Exception:
            pass
        try:
            f.write(str(new_val))
            f.flush()
            os.fsync(f.fileno())
        finally:
            try:
                fcntl.flock(f, fcntl.LOCK_UN)
            except Exception:
                pass
    return new_val

# ---------- worker loop (each worker pops indices and processes sample_entries[idx]) ----------
def worker_loop(worker_id, debug=0, gpu_id=None):
    import builtins
    import newdatagen as datagen  # import inside worker (spawn)
    print = lambda *args, **kwargs: builtins.print(*args, **{**kwargs, "flush": True})

    if gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        print(f"[Worker {worker_id}] pinned to CUDA_VISIBLE_DEVICES={gpu_id}")

    # load sample_entries once from disk (deterministic order)
    sample_entries_local = safe_read_pickle(SAMPLE_ENTRIES_PATH)
    print(f"[Worker {worker_id}] starting, {len(sample_entries_local)} candidates in store")

    while True:
        # stop early if we already hit target
        current = safe_read_pickle(PICKLE_PATH)
        if len(current) >= TARGET_EXAMPLES:
            print(f"[Worker {worker_id}] target reached ({len(current)}), exiting")
            return

        idx = pop_work_index(WORK_QUEUE_PATH)
        if idx is None:
            print(f"[Worker {worker_id}] work queue empty, exiting")
            return

        try:
            prompt_entry = sample_entries_local[idx]
        except Exception:
            print(f"[Worker {worker_id}] invalid index {idx}; skipping")
            continue

        # Each entry in original code is (prompt,) tuple style; adapt if your format differs.
        # If format_entry yields (prompt, other_meta), adjust accordingly.
        try:
            prompt, = prompt_entry
        except Exception:
            # fallback if entry is just the prompt string
            prompt = prompt_entry

        print(f"[Worker {worker_id}] processing \"{prompt}\"")

        sft_example = datagen.make_sft_example(prompt, debug=debug)
        if sft_example is None:
            print(f"[Worker {worker_id}] result was None for index {idx}")
            continue

        try:
            new_len = safe_append_pickle(PICKLE_PATH, sft_example)
            print(f"[Worker {worker_id}] appended example for idx={idx}, total={new_len}")
            if new_len >= TARGET_EXAMPLES:
                print(f"[Worker {worker_id}] reached target after appending ({new_len})")
                return
        except Exception as e:
            print(f"[Worker {worker_id}] append failed for idx={idx}: {e}")
            # optionally requeue idx on failure; currently we skip
            continue

    # end while

# ---------- SFT runner (main-process only) ----------
def run_sft(gen_epoch, gpu_id=None):
    from copy import deepcopy

    data = safe_read_pickle(PICKLE_PATH)
    data = [({**y, 'samples': [(z, w, i) for i, (z, w) in enumerate(y['samples']) if len(z) != i + 1]}) for y in data if len(y['samples'])]

    pmap = dict()
    dsmap = dict()
    
    for k, v in dataset_sources.items():
        # if "qa" in k: continue
        for vv in v:
            dsmap[format_entry(vv, k)] = k
            # high faithfulness, low accuracy
            # pmap[format_entry(vv, k)] = (format_answer(vv, k), (0.8 if "gsm" in k else 0.4 if "qa" in k else 0.8), (1.2 if "qa" in k else 0.75))
            # medium faithfulness, high accuracy
            # pmap[format_entry(vv, k)] = (format_answer(vv, k), (1.5 if "gsm" in k else 0.4 if "qa" in k else 0.8), (1.2 if "qa" in k else 0.75))
            # bestest2
            # pmap[format_entry(vv, k)] = (format_answer(vv, k), (1.5 if "gsm" in k else 0.4 if "qa" in k else 0.8), (1.0 if "qa" in k else 1.0))
            # bestest3
            # pmap[format_entry(vv, k)] = (format_answer(vv, k), (1.0 if "gsm" in k else 0.4 if "qa" in k else 0.8), (1.0 if "qa" in k else 1.0))
            # bestest4
            # pmap[format_entry(vv, k)] = (format_answer(vv, k), (1.0 if "gsm" in k else 0.4 if "qa" in k else 0.8), (1.0 if "qa" in k else 1.0))
            pmap[format_entry(vv, k)] = (format_answer(vv, k), (1.8 if "gsm" in k else 0.4 if "qa" in k else 0.8), (1.0 if "qa" in k else 1.0))

    kept = []
    set_aside = []
    l, h = 0.5, 7
    
    for entry in data:
        samples = entry.get('samples', [])
    
        outside = [s for s in samples if s[1] < l or s[1] > h]
    
        if outside:
            set_aside.append({
                'prompt': entry['prompt'],
                'original': entry['original'],
                'samples': outside
            })
    
        inside = [s for s in samples if s[1] >= l and s[1] <= h]
        if inside:
            new_entry = deepcopy(entry)
            new_entry['samples'] = inside
            kept.append(new_entry)
    
    from model.model import load_tokenizer, load_aligned_model, load_base_model
    
    tokenizer = load_tokenizer()
    model = load_aligned_model()
    ref_model = load_base_model()
    
    model.train()
    ref_model.eval()
    
    import os
    import torch
    from datasets import Dataset
    from transformers import TrainingArguments, Trainer
    from torch.nn import functional as F
    from torch.optim import AdamW
    
    device = next(model.parameters()).device
    
    def _join_trace(trace):
        if isinstance(trace, (list, tuple)):
            return "\n".join(s.strip() for s in trace if s is not None)
        return str(trace)
    
    examples = []
    raw_scores = [float(sc) for e in kept for _, sc in e.get("samples", [])]
    if not raw_scores:
        raise ValueError("kept contains no samples")
    mn, mx = min(raw_scores), max(raw_scores)
    denom = max(1e-12, mx - mn)
    eos = tokenizer.eos_token or ""
    
    for e in kept:
        prompt = e["prompt"].strip()
        for trace, score in e.get("samples", []):
            weight = (float(score) - mn) / denom
            weight = 0.05 + 0.95 * weight
            inp = prompt + eos
            tgt = _join_trace(trace) + eos
            inp_ids = tokenizer.encode(inp, add_special_tokens=False)
            tgt_ids = tokenizer.encode(tgt, add_special_tokens=False)
            if len(inp_ids) + len(tgt_ids) > MAX_LENGTH:
                keep_tgt = MAX_LENGTH // 2
                keep_inp = MAX_LENGTH - keep_tgt
                inp_ids = inp_ids[-keep_inp:]
                tgt_ids = tgt_ids[:keep_tgt]
            input_ids = inp_ids + tgt_ids
            labels = [-100] * len(inp_ids) + tgt_ids
            examples.append({"input_ids": input_ids, "labels": labels, "weight": float(weight)})
    
    hf_ds = Dataset.from_list(examples)
    
    def data_collator(batch):
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        max_len = max(len(x["input_ids"]) for x in batch)
        input_ids = [x["input_ids"] + [pad_id] * (max_len - len(x["input_ids"])) for x in batch]
        labels = [x["labels"] + [-100] * (max_len - len(x["labels"])) for x in batch]
        attention_mask = [[1] * len(x["input_ids"]) + [0] * (max_len - len(x["input_ids"])) for x in batch]
        weights = [x["weight"] for x in batch]
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "weights": torch.tensor(weights, dtype=torch.float)
        }
    
    from torch.nn import functional as F
    
    class WeightedSFTTrainer(Trainer):
        def __init__(self, ref_model=None, kl_lambda=0.5, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.ref_model = ref_model
            self.kl_lambda = kl_lambda
            if self.ref_model is not None:
                self.ref_model.to(self.model.device)
                self.ref_model.eval()
                for p in self.ref_model.parameters():
                    p.requires_grad = False
    
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            weights = inputs.pop("weights", None)
            device = self.model.device
            tensor_inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        
            if weights is None:
                weights = torch.ones(tensor_inputs["labels"].size(0), dtype=torch.float, device=device)
            else:
                weights = weights.to(device).float()
        
            labels = tensor_inputs["labels"]
            outputs = model(**tensor_inputs)
            logits = outputs.logits  # (B, S, V)
        
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            mask = (shift_labels != -100).float()
        
            vocab = shift_logits.size(-1)
            loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction="none")
            flat_logits = shift_logits.view(-1, vocab)
            flat_labels = shift_labels.view(-1)
            token_losses = loss_fct(flat_logits, flat_labels).view(shift_labels.size(0), -1)
        
            token_loss_sum = (token_losses * mask).sum(dim=1)
            denom = mask.sum(dim=1).clamp(min=1.0)
            per_sample_ce = token_loss_sum / denom
            weighted_ce = (per_sample_ce * weights).sum() / max(1e-12, weights.sum())
            total_loss = weighted_ce
        
            if self.ref_model is not None and self.kl_lambda > 0:
                with torch.no_grad():
                    ref_logits = self.ref_model(
                        input_ids=tensor_inputs["input_ids"],
                        attention_mask=tensor_inputs.get("attention_mask", None)
                    ).logits
                ref_shift = ref_logits[..., :-1, :].contiguous()
                ref_logp = F.log_softmax(ref_shift, dim=-1)
                model_logp = F.log_softmax(shift_logits, dim=-1)
                ref_p = torch.exp(ref_logp)
                per_token_kl = (ref_p * (ref_logp - model_logp)).sum(dim=-1)    # (B, S-1)
                per_sample_kl = (per_token_kl * mask).sum(dim=1) / denom
                kl_weights = (1.0 - weights).clamp(min=0.0)
                weighted_kl = (per_sample_kl * kl_weights).sum() / max(1e-12, kl_weights.sum())
                total_loss = total_loss + self.kl_lambda * weighted_kl
        
            return (total_loss, outputs) if return_outputs else total_loss
    
    
    training_args = TrainingArguments(
        output_dir=out_subdir + "/training-output",
        per_device_train_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        learning_rate=LR,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        fp16=torch.cuda.is_available(),
        save_strategy="epoch",
        save_total_limit=3,
        remove_unused_columns=False,
        report_to="none",
    )
    
    trainer = WeightedSFTTrainer(
        model=model,
        args=training_args,
        train_dataset=hf_ds,
        data_collator=data_collator,
        tokenizer=tokenizer,
        ref_model=ref_model,
        kl_lambda=KL_LAMBDA
    )
    
    trainer.train()
    
    from model.model import save_aligned_model
    save_aligned_model(model)

    del tokenizer, model, ref_model, trainer

    clear_cuda()

def gen_work_queue():
    import newdatagen as datagen

    # Build deterministic sample_entries list from datagen.dataset_sources
    entries = []
    # Ensure deterministic order: sort dataset names
    for dataset_name in sorted(datagen.dataset_sources.keys()):
        dataset = datagen.dataset_sources[dataset_name]
        for entry in dataset:
            entries.append(datagen.format_entry(entry, dataset_name))

    # Save/load deterministically to SAMPLE_ENTRIES_PATH (so resume uses same ordering)
    sample_entries = save_sample_entries_if_missing(SAMPLE_ENTRIES_PATH, entries)

    # initialize work queue file (indices) if missing (resume respects existing queue)
    init_work_queue_if_missing(WORK_QUEUE_PATH, len(sample_entries))

# ---------- main ----------
if __name__ == "__main__":
    try:
        set_start_method("spawn")
    except RuntimeError:
        # already set earlier in this interpreter
        pass

    p = Process(target=gen_work_queue, args=())
    p.start()
    p.join()

    start_epoch = read_number(EPOCH_PATH)
    for gen_epoch in range(start_epoch, GEN_EPOCHS):
        print(f"=== GEN EPOCH {gen_epoch} ===")
        clear_cuda()

        original_count = len(safe_read_pickle(PICKLE_PATH))

        if original_count < TARGET_EXAMPLES:
            # spawn workers for this epoch
            procs = []
            gpu_map = None
            if GPU_IDS:
                gpu_map = [GPU_IDS[i % len(GPU_IDS)] for i in range(N_WORKERS)]

            print("[Main] Beginning processes with gpu map ", gpu_map)
            for wid in range(N_WORKERS):
                gid = gpu_map[wid] if gpu_map else None
                p = Process(target=worker_loop, args=(wid, DEBUG, gid))
                p.start()
                procs.append(p)

            print("[Main] Monitoring")
            start_time = time.time()

            try:
                while any(p.is_alive() for p in procs):
                    now = time.time()
                    elapsed = now - start_time
                    count = len(safe_read_pickle(PICKLE_PATH))

                    # compute rate (examples/sec)
                    rate = (count - original_count) / max(elapsed, 1e-6)
                    # estimate remaining time
                    remaining = max(TARGET_EXAMPLES - count, 0)
                    eta = remaining / max(rate, 1e-6)

                    print(f"[Monitor] epoch {gen_epoch}: {count}/{TARGET_EXAMPLES} examples generated "
                          f"({elapsed:.1f}s elapsed, ETA {eta:.1f}s)")

                    time.sleep(MONITOR_POLL_SECONDS)
            except KeyboardInterrupt:
                print("Monitor: keyboard interrupt — terminating workers")
                for p in procs:
                    try:
                        p.terminate()
                    except Exception:
                        pass
                for p in procs:
                    p.join()
                raise  # re-raise so the main run stops

            # wait for workers to finish (they should exit when TARGET_EXAMPLES reached)
            for p in procs:
                p.join()

            clear_cuda()

            for p in procs:
                if p.exitcode is not None and p.exitcode != 0:
                    print(f"[Main] worker pid={p.pid} exited with code {p.exitcode}")

        # final count for this epoch
        final = safe_read_pickle(PICKLE_PATH)
        print(f"[Main] epoch {gen_epoch} finished generation: {len(final)} examples")

        # run SFT on the epoch file (pass gen_epoch so run_sft can write status)
        # currently skipped to match your previous run behavior
        print("Datagen finished, skipping SFT")

        print(f"[Main] starting SFT for epoch {gen_epoch}")
        
        run_sft(gen_epoch, gpu_id=None)
        
        print(f"[Main] finished SFT for epoch {gen_epoch}")

        # Optionally archive the datagen pickle and advance epoch counter
        try:
            if os.path.exists(PICKLE_PATH):
                shutil.move(PICKLE_PATH, PICKLE_ARCHIVE_PATH % gen_epoch)
        except Exception as e:
            print("Rename failed:", e)
        increment_number(EPOCH_PATH)

    print("All gen epochs complete.")
