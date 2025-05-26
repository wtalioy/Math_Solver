from datasets import load_dataset
from trl import SFTTrainer
from transformers import AutoTokenizer, AutoModelForCausalLM
from swanlab.integration.transformers import SwanLabCallback
import os

from config import training_args, experiment_name

trainset_path = "data/limo.jsonl"

if "MODEL_PATH" not in os.environ:
    os.environ["MODEL_PATH"] = "./models/Qwen3-0.6B"

def main():
    # Load the dataset
    dataset = load_dataset("GAIR/LIMO", split="train")
    dataset = dataset.rename_columns({"question": "prompt", "solution": "completion"})

    # Load the model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        os.environ["MODEL_PATH"],
        # device_map="auto",
        torch_dtype="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(
        os.environ["MODEL_PATH"],
        use_fast=True,
    )

    # Initialize the SwanLab callback
    swanlab_config = {
        "workspace": "unknown_ft",
        "project": "Qwen3-0.6B-SFT",
        "experiment_name": experiment_name,
        "config": {
            "model": os.environ["MODEL_PATH"],
            "dataset": "limo",
        },
        "log_dir": "swanlogs",
    }

    swanlab_callback = SwanLabCallback(**swanlab_config)

    # Initialize the trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=None,
        processing_class=tokenizer,
        data_collator=None,  # You can define a custom data collator if needed
        callbacks=[swanlab_callback],
    )

    # Start training
    trainer.train()

    # Save the model
    trainer.save_model(training_args.output_dir)
    

if __name__ == "__main__":
    main()
    