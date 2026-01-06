LARGE = False # Set false for 7-8B, set true for 24-32B
MISTRAL = True # Set false for Qwen, set true for Mistral
idx = (0 if MISTRAL else 1) if not LARGE else (2 if MISTRAL else 3) # mistral small, qwen small, mistral large, qwen large

CACHE_DIR = "/lambda/nfs/Algoverse/hf_cache" # Hugging Face cache directory
CHECKPOINTS_DIR = "/lambda/nfs/Algoverse/hf_checkpoints" # Directory to save fine-tuned models to

OUT_DIR = "/lambda/nfs/Algoverse/frit-out" # Directory for generated files to be written to; this directory must exist before you run the code
DATA_DIR = "/lambda/nfs/Algoverse/frit/data" # Directory where data files are stored (set this to /path/to/frit/data)

GPUS = 1 # Number of GPUs
PER_GPU = 2 # Number of workers per GPU (used only in parallel_dpo.py; adjust as needed for your GPU; we used 2 workers per GPU on RTX Pro 6000 Server GPUs)

data_subdir = DATA_DIR + ("/mistral" if MISTRAL else "/qwen") + ("-large" if LARGE else "")
out_subdir = OUT_DIR + ("/mistral" if MISTRAL else "/qwen") + ("-large" if LARGE else "")

# Config for FRIT

GEN_EPOCHS = 1
# TARGET_EXAMPLES = 3000 if LARGE else 5000
TARGET_EXAMPLES = 2000
PRELIM_TEMP = [0.1, 0.5, 0.1, 0.5][idx]
VARIOUS_TEMP = [0.2, 1.6, 0.2, 1.6][idx]

BATCH_SIZE = 4
EPOCHS = 3
LR = [1e-5, 1e-4, 5e-7, 1e-4][idx]
GRAD_ACCUM_STEPS = 1
MAX_LENGTH = 2048
KL_LAMBDA = [0.4, 0.1, 0.2, 0.1][idx]
