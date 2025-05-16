from trl import GRPOConfig

experiment_name = "Qwen3-0.6B-GRPO-r16-official-sampling"

class RewardConfig:
    cosine_max_len = 1000
    cosine_min_value_wrong = -1.0
    cosine_max_value_wrong = -0.5
    cosine_min_value_correct = 0.5
    cosine_max_value_correct = 1.0

RewardConfig = RewardConfig()

training_args = GRPOConfig(
    temperature=0.6,
    top_p=0.95,
    top_k=20,
    min_p=0,
    presence_penalty=1.5,  # comment this line as current trl does not support presence_penalty

    use_vllm = True,

    learning_rate = 5e-5,
    adam_beta1 = 0.9,
    adam_beta2 = 0.99,
    weight_decay = 0.1,
    warmup_ratio = 0.1,
    lr_scheduler_type = "cosine",
    optim = "adamw_8bit",
    logging_steps = 1,
    # bf16 = is_bfloat16_supported(),
    # fp16 = not is_bfloat16_supported(),
    bf16=True,
    dataloader_num_workers=8,
    per_device_train_batch_size = 4,
    per_device_eval_batch_size = 4,
    gradient_accumulation_steps = 4,
    num_generations = 6,
    max_prompt_length = 256,
    max_completion_length = 1024,
    reward_weights=[
        1.0,  # format reward
        1.0,  # tag count reward
        1.0,  # accuracy reward
        1.0,  # length reward
        1.0,  # cosine reward
    ],

    max_steps = 360,
    save_steps = 40,
    max_grad_norm = 0.1,
    report_to = "tensorboard",
    logging_dir = f"logs/{experiment_name}",
    output_dir = f"outputs/{experiment_name}",
    log_completions=True,
)

SYSTEM_PROMPT = "You are a helpful Math assistant. Carefully think step by step and enclose your response within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>"
