# FRIT: Using Causal Importance to Improve Chain-of-Thought Faithfulness

You can find our paper at [https://arxiv.org/pdf/2509.13334](https://arxiv.org/pdf/2509.13334).

## Prerequisites

Python and pip are required to run the code. You must have Pytorch installed on your system.

## Setup

Clone this repository. Install the dependencies by running `pip -r /path/to/frit/requirements.txt`.

Adjust hyperparameters, directories (particularly `DATA_DIR`) and any other configuration in `config.py` prior to running.

If fine-tuning Mistral, you must go to the [Hugging Face page](https://huggingface.co/mistralai/Mistral-7B-v0.1), log in, and allow data collection as it is a gated model. Then, generate an access token in Hugging Face and store it in the environment variable `HF_TOKEN` prior to running the code.

## Fine-tuning a model

You can choose to run the full pipeline (more computational power/time) or use the pre-generated dataset (less computational power/time but less flexibility).

### Running FRIT (full pipeline)

Run the following on the command line:

```bash
cd /path/to/frit
python parallel_dpo.py
```

This will perform the full FRIT pipeline end-to-end. The final fine-tuned model will be saved to a directory inside `config.CHECKPOINTS_DIR`.

### Fine-tuning using pre-generated dataset

As an alternative, you may utilize the generated DPO triplets we have released. This takes far less computational power but, of course, you can't modify the parameters used to generate the triplets.

First, make sure you have set the hyperparameters in `config.py` to the correct values (refer to our paper for the hyperparameters we used). Then, run the following on the command line:

```bash
cd /path/to/frit
python dpo.py
```

The final fine-tuned model will be saved to a directory inside `config.CHECKPOINTS_DIR`.

## Evaluation

Once you have created a fine-tuned model, run `python evaluation.py [OPTIONS]` to evaluate it. See the `main` function in the source code for details on command-line options you may pass it.
