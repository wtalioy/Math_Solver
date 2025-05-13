from accelerate import Accelerator
accelerator = Accelerator()

from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import GRPOTrainer
from peft import get_peft_model, LoraConfig
from swanlab.integration.transformers import SwanLabCallback

from data_utils import load_custom_dataset
from config import training_args
from reward_funcs import format_reward, tag_count_reward, accuracy_reward, len_reward, cosine_scaled_reward
import os

max_seq_length = 1024
max_lora_rank = 32

trainset_path = "data/train_10k.json"
valset_path = "data/val.json"

if "MODEL_PATH" not in os.environ:
    os.environ["MODEL_PATH"] = "./models/Qwen3-0.6B"

def main():
    # Initialize the language model
    model = AutoModelForCausalLM.from_pretrained(
        os.environ["MODEL_PATH"],
        # device_map="auto",
        torch_dtype="auto",
    )

    # Initialize the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        os.environ["MODEL_PATH"],
        use_fast=True,
    )

    # Initialize peft config
    peft_config = LoraConfig(
        r=max_lora_rank,
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "down_proj", "up_proj"],
    )

    model = get_peft_model(
        model,
        peft_config,
    )

    # Load the dataset
    train_dataset = load_custom_dataset(trainset_path)
    val_dataset = load_custom_dataset(valset_path)

    # Initialize the SwanLab callback
    swanlab_callback = SwanLabCallback(
        project="Qwen3-0.6B-GRPO",
        # experiment_name="Qwen3-0.6B-GRPO",
        config={
            "model": os.environ["MODEL_PATH"],
            "dataset": "custom_dataset",
        }
    )

    # Initialize the GRPO trainer
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[
            format_reward,
            tag_count_reward,
            accuracy_reward,
            len_reward,
            cosine_scaled_reward,
        ],
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        callbacks=[swanlab_callback],
    )

    # Start training
    trainer.train()

    # Save the model
    trainer.save_model(training_args.output_dir)


if __name__ == "__main__":
    main()