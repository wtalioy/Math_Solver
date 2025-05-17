from trl import SFTConfig

experiment_name = "Qwen3-0.6B-SFT"

training_args = SFTConfig(
    packing=True,
    use_liger_kernel=True,
    eos_token="<|im_end|>",
    max_length=4096,
    bf16=True,
    per_device_train_batch_size=2,
    num_train_epochs=3,
    gradient_accumulation_steps=8,
    learning_rate=2e-5,
    weight_decay=0.01,
    save_steps=40,
    output_dir=f"outputs/{experiment_name}",
    logging_dir=f"logs/{experiment_name}",
    logging_steps=1,
    report_to="tensorboard",
)