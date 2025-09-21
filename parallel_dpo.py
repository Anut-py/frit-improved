#!/usr/bin/env python3
import builtins
print = lambda *args, **kwargs: builtins.print(*args, **{**kwargs, "flush": True})

import os
import pickle
import time
import random
import fcntl
from multiprocessing import Process, set_start_method
from tqdm import tqdm
import json
import math
import torch, gc
from config import GPUS, PER_GPU, OUT_DIR, DPO_CONFIG, GEN_EPOCHS, TARGET_EXAMPLES, out_subdir

def clear_cuda():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

MONITOR_POLL_SECONDS = 5   # how often main prints datagen progress while workers run

PER_DATASET = math.ceil(TARGET_EXAMPLES / 6)

N_WORKERS = GPUS * PER_GPU
GPU_IDS = list(range(GPUS))
DEBUG = 0

LOCK_RETRY_DELAY = 0.05
LOCK_RETRY_ATTEMPTS = 5
PICKLE_PATH = OUT_DIR + "/datagen.pkl"
PICKLE_ARCHIVE_PATH = out_subdir + "/datagen%d.pkl"
EPOCH_PATH = OUT_DIR + "/current_epoch"
STATE_PATH = OUT_DIR + "/dpo_state.pt"
STATUS_PATH = OUT_DIR + "/dpo_status.json"

import tempfile
import shutil
import logging

LOG = logging.getLogger(__name__)

def safe_read_pickle(path):
    """Read pickle, but if file is corrupted try .bak before giving up."""
    try:
        with open(path, "rb") as f:
            fcntl.flock(f, fcntl.LOCK_SH)
            try:
                data = pickle.load(f)
            except EOFError:
                # primary file is corrupt/partial — try backup
                LOG.warning("%s appears truncated/empty; attempting to load backup", path)
                try:
                    fcntl.flock(f, fcntl.LOCK_UN)
                except Exception:
                    pass
                bak = path + ".bak"
                if os.path.exists(bak):
                    with open(bak, "rb") as fb:
                        fcntl.flock(fb, fcntl.LOCK_SH)
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
    except Exception as e:
        LOG.exception("Unexpected error reading %s: %s", path, e)
        # As a conservative fallback, try backup
        bak = path + ".bak"
        if os.path.exists(bak):
            try:
                with open(bak, "rb") as fb:
                    fcntl.flock(fb, fcntl.LOCK_SH)
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
                fcntl.flock(rf, fcntl.LOCK_SH)
                try:
                    try:
                        current = pickle.load(rf)
                    except EOFError:
                        # truncated primary; try backup (will be handled in safe_read_pickle)
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
    fd, tmppath = tempfile.mkstemp(dir=dirpath, prefix=os.path.basename(path) + ".tmp.")
    try:
        with os.fdopen(fd, "wb") as tf:
            # exclusive lock on temp isn't needed, but keep pattern consistent
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

        # Before replacing, optionally make a backup of the current file
        bak = path + ".bak"
        if os.path.exists(path):
            # atomically move current -> bak
            tmpbak = bak + f".tmp.{os.getpid()}"
            try:
                os.replace(path, tmpbak)
                os.replace(tmpbak, bak)
            except Exception:
                # If this fails, try a best-effort copy then continue
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

def read_number(path: str) -> int:
    """Read an integer from a file. Returns 0 if missing or invalid."""
    if not os.path.exists(path):
        return 0
    try:
        with open(path, "r") as f:
            fcntl.flock(f, fcntl.LOCK_SH)
            try:
                data = f.read().strip()
                return int(data) if data else 0
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)
    except Exception:
        return 0

def increment_number(path: str, delta: int = 1) -> int:
    """Atomically increment the number in the file by delta. Returns new value."""
    current = read_number(path)
    new_val = current + delta
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            f.write(str(new_val))
            f.flush()
            os.fsync(f.fileno())
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)
    return new_val

# ---------- worker loop ----------
def worker_loop(worker_id, n_per_dataset=PER_DATASET, debug=0, gpu_id=None):
    import builtins
    print = lambda *args, **kwargs: builtins.print(*args, **{**kwargs, "flush": True})

    if gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        print(f"[Worker {worker_id}] pinned to CUDA_VISIBLE_DEVICES={gpu_id}")

    import datagen

    # Build sample list once (shuffled)
    sample_entries = []
    for dataset_name, dataset in list(datagen.dataset_sources.items()):
        for entry in dataset.shuffle().select(range(n_per_dataset)):
            prompt, is_math = datagen.format_entry(entry, dataset_name)
            sample_entries.append((prompt, is_math))
    random.shuffle(sample_entries)

    print(f"[Worker {worker_id}] starting, {len(sample_entries)} candidates")

    for prompt, is_math in sample_entries:
        current = safe_read_pickle(PICKLE_PATH)
        if len(current) >= TARGET_EXAMPLES:
            print(f"[Worker {worker_id}] target reached ({len(current)}), exiting")
            return

        dpo_example = datagen.make_dpo_example(prompt, is_math, debug)
        if dpo_example is None:
            print(f"[Worker {worker_id}] result was None for prompt: {prompt}")
            continue

        try:
            new_len = safe_append_pickle(PICKLE_PATH, dpo_example)
            print(f"[Worker {worker_id}] appended example, total={new_len}")
            if new_len >= TARGET_EXAMPLES:
                print(f"[Worker {worker_id}] reached target after appending ({new_len})")
                return
        except Exception as e:
            print(f"[Worker {worker_id}] append failed: {e}")
            continue

    print(f"[Worker {worker_id}] finished sample list without reaching target")


# ---------- dpo runner (main process only) ----------
def run_dpo(gen_epoch, gpu_id=None):
    if gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        print(f"[DPO runner] pinned to CUDA_VISIBLE_DEVICES={gpu_id}")

    # import wandb
    from datasets import Dataset
    from trl import DPOConfig, DPOTrainer
    import torch

    torch.set_grad_enabled(True)

    from model.model import load_tokenizer, load_base_model, load_aligned_model, save_aligned_model

    tokenizer = load_tokenizer()
    ref_model = load_base_model()
    model = load_aligned_model()

    model.print_trainable_parameters()

    dpo_triples = safe_read_pickle(PICKLE_PATH)
    if len(dpo_triples) < TARGET_EXAMPLES:
        raise RuntimeError(f"DPO runner expected {TARGET_EXAMPLES} examples but found {len(dpo_triples)}")

    preference_dataset = Dataset.from_list([{"prompt": t["prompt"], "chosen": t["x_plus"], "rejected": t["x_minus"]} for t in dpo_triples])

    dpo_cfg = DPO_CONFIG

    def tokenize_dpo(example):
        chosen = tokenizer(example["chosen"], truncation=True, padding="max_length", max_length=512, return_tensors="pt")
        rejected = tokenizer(example["rejected"], truncation=True, padding="max_length", max_length=512, return_tensors="pt")
        return {
            "input_ids_chosen": chosen.input_ids.squeeze(0),
            "attention_mask_chosen": chosen.attention_mask.squeeze(0),
            "input_ids_rejected": rejected.input_ids.squeeze(0),
            "attention_mask_rejected": rejected.attention_mask.squeeze(0),
        }
    
    preference_dataset = preference_dataset.map(lambda x: tokenize_dpo(x))

    # run = wandb.init(project="frit", name="frit-dpo", config={"dpo_config": dpo_cfg})
    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        processing_class=tokenizer,
        args=DPOConfig(**dpo_cfg),
        train_dataset=preference_dataset
    )

    if os.path.exists(STATE_PATH):
        try:
            state = torch.load(STATE_PATH, map_location="cpu")
            if "optimizer" in state and getattr(trainer, "optimizer", None) is not None:
                trainer.optimizer.load_state_dict(state["optimizer"])
            if "scheduler" in state and getattr(trainer, "lr_scheduler", None) is not None:
                trainer.lr_scheduler.load_state_dict(state["scheduler"])
            saved_args = state.get("args")
            if saved_args:
                try:
                    LOG.info("Loaded saved trainer args (not overwriting active config)")
                except Exception:
                    pass
            print("[DPO] Resumed optimizer/scheduler state")
        except Exception:
            LOG.exception("Failed to load %s; starting from scratch", STATE_PATH)

    dpo_status = {"state": "running", "start_time": time.time(), "gen_epoch": gen_epoch}
    with open(STATUS_PATH, "w") as sf:
        json.dump(dpo_status, sf)
        sf.flush()
        os.fsync(sf.fileno())

    result = trainer.train()

    dpo_status = {
        "state": "finished",
        "end_time": time.time(),
        "gen_epoch": gen_epoch,
        "metrics": getattr(result, "metrics", None) or {}
    }
    with open(STATUS_PATH, "w") as sf:
        json.dump(dpo_status, sf)
    
    save_aligned_model(model)
    run.finish()

    # Save optimizer/scheduler + args (non-model state)
    save_dict = {}
    if getattr(trainer, "optimizer", None) is not None:
        try:
            save_dict["optimizer"] = trainer.optimizer.state_dict()
        except Exception:
            LOG.exception("Failed to serialize optimizer.state_dict()")
    if getattr(trainer, "lr_scheduler", None) is not None:
        try:
            save_dict["scheduler"] = trainer.lr_scheduler.state_dict()
        except Exception:
            LOG.exception("Failed to serialize lr_scheduler.state_dict()")
    
    # save trainer.args in a robust plain-dict form
    try:
        if hasattr(trainer, "args"):
            try:
                # prefer a simple serializable dict
                args_dict = vars(trainer.args)
            except Exception:
                # fallback: try any to_dict/to_sanitized methods
                args_dict = {}
                for name in dir(trainer.args):
                    if name.startswith("_"):
                        continue
                    try:
                        val = getattr(trainer.args, name)
                        # skip callables
                        if not callable(val):
                            args_dict[name] = val
                    except Exception:
                        pass
            save_dict["args"] = args_dict
    except Exception:
        LOG.exception("Failed to capture trainer.args")
    
    if save_dict:
        try:
            # atomic write
            tmpfd, tmpname = tempfile.mkstemp(dir=".", prefix=os.path.basename(STATE_PATH) + ".tmp.")
            with os.fdopen(tmpfd, "wb") as tf:
                torch.save(save_dict, tf)
                tf.flush()
                os.fsync(tf.fileno())
            os.replace(tmpname, STATE_PATH)
        except Exception:
            LOG.exception("Failed to save trainer state to %s", STATE_PATH)
            if os.path.exists(tmpname):
                try:
                    os.remove(tmpname)
                except Exception:
                    pass
        else:
            print("[DPO] Saved optimizer/scheduler state")

    del trainer
    del tokenizer
    del ref_model
    del model

# ---------- main ----------
if __name__ == "__main__":
    try:
        set_start_method("spawn")
    except RuntimeError:
        # already set earlier in this interpreter
        pass
    start_epoch = read_number(EPOCH_PATH)
    for gen_epoch in range(start_epoch, GEN_EPOCHS):
        print(f"=== GEN EPOCH {gen_epoch} ===")
        clear_cuda()
    
        # remove any leftover per-epoch files so generation starts fresh
        try:
            os.remove(STATUS_PATH)
        except FileNotFoundError:
            pass

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
                p = Process(target=worker_loop, args=(wid, PER_DATASET, DEBUG, gid))
                p.start()
                procs.append(p)
        
            print("[Main] Monitoring")
            # monitor progress while workers run
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
                raise  # re-raise so the main run stops (or remove `raise` if you want to continue)
        
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
    
        # run DPO on the epoch file (pass gen_epoch so run_dpo can write status)
    
        print(f"[Main] starting DPO for epoch {gen_epoch}")
        # pass gen_epoch to allow status writes
        run_dpo(gen_epoch, gpu_id=None)
        try:
            with open(STATUS_PATH, "r") as sf:
                s = json.load(sf)
                print("[Main] DPO metrics:", s.get("metrics"))
        except Exception:
            pass
        print(f"[Main] finished DPO for epoch {gen_epoch}")

        clear_cuda()
        
        try:
            if os.path.exists(PICKLE_PATH):
                shutil.move(PICKLE_PATH, PICKLE_ARCHIVE_PATH % gen_epoch)
        except Exception as e:
            print("Rename failed:", e)

        increment_number(EPOCH_PATH)
    
    print("All gen epochs complete.")
