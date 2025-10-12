LARGE = False # Set false for 7-8B, set true for 24-32B
MISTRAL = False # Set false for Qwen, set true for Mistral

CACHE_DIR = "/workspace/hf_cache" # Hugging Face cache directory
CHECKPOINTS_DIR = "/workspace/hf_checkpoints" # Directory to save fine-tuned models to

OUT_DIR = "/workspace/frit-out" # Directory for generated files to be written to; this directory must exist before you run the code
DATA_DIR = "/workspace/frit/data" # Directory where data files are stored (set this to /path/to/frit/data)

GPUS = 3 # Number of GPUs
PER_GPU = 2 # Number of workers per GPU (used only in parallel_dpo.py; adjust as needed for your GPU; we used 2 workers per GPU on RTX Pro 6000 Server GPUs)

data_subdir = DATA_DIR + ("/mistral" if MISTRAL else "/qwen")
out_subdir = OUT_DIR + ("/mistral" if MISTRAL else "/qwen")

# Config for FRIT

GEN_EPOCHS = 3
TARGET_EXAMPLES = 480

DPO_CONFIG = {
    "beta": 0.05,
    "learning_rate": 2e-6,
    "per_device_train_batch_size": 5,
    "num_train_epochs": 1,
    "logging_steps": 1,
    "max_length": 512,
    "label_names": ["labels"]
}
