#!/usr/bin/env python3
import builtins
print = lambda *args, **kwargs: builtins.print(*args, **{**kwargs, "flush": True})

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
from config import GPUS, PER_GPU, OUT_DIR, DPO_CONFIG, GEN_EPOCHS, TARGET_EXAMPLES, out_subdir

# Constants and paths
MONITOR_POLL_SECONDS = 5   # how often main prints datagen progress while workers run

PER_DATASET = math.ceil(TARGET_EXAMPLES / 6)

N_WORKERS = GPUS * PER_GPU
GPU_IDS = list(range(GPUS))
DEBUG = 2

LOCK_RETRY_DELAY = 0.05
LOCK_RETRY_ATTEMPTS = 5
PICKLE_PATH = os.path.join(OUT_DIR, "datagen.pkl")
PICKLE_ARCHIVE_PATH = os.path.join(out_subdir, "datagen%d.pkl")
EPOCH_PATH = os.path.join(OUT_DIR, "current_epoch")
STATE_PATH = os.path.join(OUT_DIR, "dpo_state.pt")
STATUS_PATH = os.path.join(OUT_DIR, "dpo_status.json")

SAMPLE_ENTRIES_PATH = os.path.join(OUT_DIR, "sample_entries.pkl")
WORK_QUEUE_PATH = os.path.join(OUT_DIR, "work_queue.pkl")

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
    # reverse so pop() yields indices in forward order
    indices.reverse()
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

        dpo_example = datagen.make_dpo_example(prompt, debug=debug)
        if dpo_example is None:
            print(f"[Worker {worker_id}] result was None for index {idx}")
            continue

        try:
            new_len = safe_append_pickle(PICKLE_PATH, dpo_example)
            print(f"[Worker {worker_id}] appended example for idx={idx}, total={new_len}")
            if new_len >= TARGET_EXAMPLES:
                print(f"[Worker {worker_id}] reached target after appending ({new_len})")
                return
        except Exception as e:
            print(f"[Worker {worker_id}] append failed for idx={idx}: {e}")
            # optionally requeue idx on failure; currently we skip
            continue

    # end while

# ---------- DPO runner (main-process only) ----------
def run_dpo(gen_epoch, gpu_id=None):
    import wandb
    from datasets import Dataset
    from trl import DPOConfig, DPOTrainer

    if gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        print(f"[DPO runner] pinned to CUDA_VISIBLE_DEVICES={gpu_id}")

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

    run = wandb.init(project="frit", name="frit-dpo", config={"dpo_config": dpo_cfg})
    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        processing_class=tokenizer,
        args=DPOConfig(**dpo_cfg),
        train_dataset=preference_dataset
    )

    # restore optimizer/scheduler if present
    if os.path.exists(STATE_PATH):
        try:
            state = torch.load(STATE_PATH, map_location="cpu", weights_only=False)
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
    try:
        with open(STATUS_PATH, "w") as sf:
            json.dump(dpo_status, sf)
            sf.flush()
            os.fsync(sf.fileno())
    except Exception:
        pass

    result = trainer.train()

    dpo_status = {
        "state": "finished",
        "end_time": time.time(),
        "gen_epoch": gen_epoch,
        "metrics": getattr(result, "metrics", None) or {}
    }
    try:
        with open(STATUS_PATH, "w") as sf:
            json.dump(dpo_status, sf)
    except Exception:
        pass

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
                args_dict = vars(trainer.args)
            except Exception:
                args_dict = {}
                for name in dir(trainer.args):
                    if name.startswith("_"):
                        continue
                    try:
                        val = getattr(trainer.args, name)
                        if not callable(val):
                            args_dict[name] = val
                    except Exception:
                        pass
            save_dict["args"] = args_dict
    except Exception:
        LOG.exception("Failed to capture trainer.args")

    if save_dict:
        try:
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

    # cleanup
    try:
        del trainer
        del tokenizer
        del ref_model
        del model
    except Exception:
        pass

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

        # remove any leftover per-epoch files so generation starts fresh
        try:
            os.remove(STATUS_PATH)
        except Exception:
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

        # run DPO on the epoch file (pass gen_epoch so run_dpo can write status)
        # currently skipped to match your previous run behavior
        print("Datagen finished, skipping DPO")
        # Uncomment to enable DPO stage:
        # print(f"[Main] starting DPO for epoch {gen_epoch}")
        # run_dpo(gen_epoch, gpu_id=None)
        # try:
        #     with open(STATUS_PATH, "r") as sf:
        #         s = json.load(sf)
        #         print("[Main] DPO metrics:", s.get("metrics"))
        # except Exception:
        #     pass
        # print(f"[Main] finished DPO for epoch {gen_epoch}")

        # Optionally archive the datagen pickle and advance epoch counter
        # try:
        #     if os.path.exists(PICKLE_PATH):
        #         shutil.move(PICKLE_PATH, PICKLE_ARCHIVE_PATH % gen_epoch)
        # except Exception as e:
        #     print("Rename failed:", e)
        # increment_number(EPOCH_PATH)

    print("All gen epochs complete.")
