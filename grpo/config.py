from transformers.utils import is_bfloat16_supported
from trl import GRPOConfig

class RewardConfig:
    cosine_max_len = 1000
    cosine_min_value_wrong = -1.0
    cosine_max_value_wrong = -0.5
    cosine_min_value_correct = 0.5
    cosine_max_value_correct = 1.0

RewardConfig = RewardConfig()

training_args = GRPOConfig(
    top_p=0.95,
    min_p=0,
    top_k=20,
    temperature=0.6,

    use_vllm = True,
    learning_rate = 5e-6,
    adam_beta1 = 0.9,
    adam_beta2 = 0.99,
    weight_decay = 0.1,
    warmup_ratio = 0.1,
    lr_scheduler_type = "cosine",
    optim = "adamw_8bit",
    logging_steps = 1,
    bf16 = is_bfloat16_supported(),
    fp16 = not is_bfloat16_supported(),
    per_device_train_batch_size = 4,
    per_device_eval_batch_size = 4,
    gradient_accumulation_steps = 1,
    num_generations = 8,
    max_prompt_length = 256,
    max_completion_length = 1024,
    reward_weights=[
        1.0,  # format reward
        1.0,  # tag count reward
        1.0,  # accuracy reward
        1.0,  # length reward
        1.0,  # cosine reward
    ],

    max_steps = 200,
    save_steps = 50,
    max_grad_norm = 0.1,
    report_to = "tensorboard",
    logging_dir = "logs", 
    output_dir = "outputs",
)

SYSTEM_PROMPT = """
你是资深的数学领域的专家。请按以下格式，一步步思考，回答小学数学1-6年级的校内题目，最终答案不要带单位:
<think>
...
</think>
<answer>
...
</answer>
"""
