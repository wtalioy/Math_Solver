from unsloth import FastLanguageModel, PatchFastRL
from trl import GRPOTrainer
from swanlab.integration.transformers import SwanLabCallback
from data_utils import load_custom_dataset
from config import training_args
from reward_funcs import format_reward, tag_count_reward, accuracy_reward, len_reward, cosine_scaled_reward
import os

PatchFastRL("GRPO", FastLanguageModel)

max_seq_length = 2048
max_lora_rank = 32

trainset_path = "data/train_10k.json"
valset_path = "data/val.json"

if "MODEL_PATH" not in os.environ:
    os.environ["MODEL_PATH"] = "./models/Qwen2.5-0.5B-Instruct"

def main():

    # Initialize the language model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=os.environ["MODEL_PATH"],
        max_seq_length=max_seq_length,
        max_lora_rank=max_lora_rank,
        load_in_4bit=True,
        # load_in_8bit=True,
        fast_inference=True,
        gpu_memory_utilization=0.9,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=max_lora_rank,
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "down_proj", "up_proj"],
    )

    # Load the dataset
    train_dataset = load_custom_dataset(trainset_path)
    val_dataset = load_custom_dataset(valset_path)

    # Initialize the SwanLab callback
    swanlab_callback = SwanLabCallback(
        project="Qwen2.5-0.5B-grpo",
        experiment_name="Qwen2.5-0.5B-grpo",
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