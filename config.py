LARGE = False # Set false for 7-8B, set true for 24-32B
MISTRAL = True # Set false for Qwen, set true for Mistral

CACHE_DIR = "/workspace/hf_cache" # Hugging Face cache directory
CHECKPOINTS_DIR = "/workspace/hf_checkpoints" # Directory to save fine-tuned models to

OUT_DIR = "/workspace/frit-out" # Directory for generated files to be written to; this directory must exist before you run the code
DATA_DIR = "/workspace/frit/data" # Directory where data files are stored (set this to /path/to/frit/data)

GPUS = 1 # Number of GPUs
PER_GPU = 4 # Number of workers per GPU (used only in parallel_dpo.py; adjust as needed for your GPU; we used 2 workers per GPU on RTX Pro 6000 Server GPUs)

data_subdir = DATA_DIR + ("/mistral" if MISTRAL else "/qwen")
out_subdir = OUT_DIR + ("/mistral" if MISTRAL else "/qwen")

# Config for FRIT

GEN_EPOCHS = 1
TARGET_EXAMPLES = 5000
PRELIM_TEMP = 0.1 # 0.5
VARIOUS_TEMP = 0.2 # 1.6

BATCH_SIZE = 4
EPOCHS = 3
LR = 1e-4
GRAD_ACCUM_STEPS = 1
MAX_LENGTH = 512
KL_LAMBDA = 0.1
