from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils import PreTrainedTokenizer
from peft import PeftModel, LoraConfig, TaskType, get_peft_model
import torch
from pathlib import Path
import os
import shutil
from huggingface_hub import login
from config import LARGE, MISTRAL, CACHE_DIR, CHECKPOINTS_DIR

tokenizers = {}
models = {}
aligned_models = {}

def get_model_name(large: bool):
    if MISTRAL:
        return "mistralai/Mistral-Small-Instruct-2409" if large else "mistralai/Mistral-7B-Instruct-v0.3"
    else:
        return "Qwen/Qwen3-32B" if large else "Qwen/Qwen3-8B"

def get_model_name_aligned(large: bool):
    if MISTRAL:
        return "Mistral-22B-aligned" if large else "Mistral-7B-aligned"
    else:
        return "Qwen3-32B-aligned" if large else "Qwen3-8B-aligned"

def load_tokenizer(large: bool = None) -> PreTrainedTokenizer:
    if large is None:
        large = LARGE

    if large in tokenizers:
        return tokenizers[large]
    
    tokenizer = AutoTokenizer.from_pretrained(get_model_name(large), trust_remote_code=True, device_map="auto", cache_dir=CACHE_DIR)
    tokenizer.padding_side = 'left'

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizers[large] = tokenizer
    
    return tokenizer

# Load unaligned model
def load_base_model(large: bool = None) -> PreTrainedModel:
    if large is None:
        large = LARGE

    if large in models:
        return models[large]

    model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
        get_model_name(large),
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        cache_dir=CACHE_DIR
    )
    model.eval()
    models[large] = model
    
    return model

# Load aligned model weights from disk
def load_aligned_model(large: bool = None, trainable = True) -> PreTrainedModel:
    if large is None:
        large = LARGE

    if large in aligned_models:
        return aligned_models[large]
    
    model_dir = Path(f"{CHECKPOINTS_DIR}/{get_model_name_aligned(large)}")

    model = AutoModelForCausalLM.from_pretrained(
        get_model_name(large),
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        cache_dir=CACHE_DIR
    )

    if model_dir.exists():
        # Load from disk if aligned model exists
        model = PeftModel.from_pretrained(model, model_dir, is_trainable=trainable)
    else:
        lora_config = LoraConfig(
            r=64,
            lora_alpha=128,
            target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lora_config)
    if trainable:
        model.train()
    else:
        model.eval()
    
    aligned_models[large] = model
    
    return model

# Save aligned model weights to disk
def save_aligned_model(model: PreTrainedModel, large: bool = None) -> None:
    if large is None:
        large = LARGE

    model_dir = Path(f"{CHECKPOINTS_DIR}/{get_model_name_aligned(large)}")
    
    model_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(model_dir)

# Reset aligned model weights to base
def reset_aligned_model(large: bool = None) -> None:
    if large is None:
        large = LARGE

    model_dir = Path(f"{CHECKPOINTS_DIR}/{get_model_name_aligned(large)}")

    if os.path.exists(model_dir):
        shutil.rmtree(model_dir)

    save_aligned_model(load_aligned_model(large), large)
