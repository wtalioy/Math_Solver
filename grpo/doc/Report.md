# GRPO Report

## Shared Setup
- **Pretrained Model:** `Qwen3-0.6B(Instruct)`
- **Training Framework:** `Huggingface` + `Trl` + `Accelerate`
- **Training Method:** `LoRA`
- **Training Data Volume:** `10000`
- **Validation Data Volume:** `2000`

## Baseline performance
- **Validation Set:** `val.json`
    - **System Prompt**: "You are a helpful Math assistant. Carefully think step by step and enclose your response within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>"
    - **Accuracy:** `76.98%`

## Ablation Experiments

### Raw_5e-5_Latex

#### Setup
- `train_dataset`: `raw_10k.json`
- `max_lora_rank`: `32`
- `learning_rate`: `5e-5`
- `batch_size`: `28`
- `num_generations`: `7`
- `max_grad_norm`: `0.1`
- `reward_funcs`: `format_reward`, `tag_count_reward`, `accuracy_reward`, `cosine_scaled_reward`
- `system_prompt`: "You are a helpful math assistant. When the user ask a question, the assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e.,\n<think>\nreasoning process here\n</think>\n<answer> answer here wrapped in \\boxed{} </answer>"

#### Results
- Rewards increased with global steps, but the model's performance was not satisfactory. Accuracy reward fluctuated around 0.15, which we guess is due to the failure of the accuracy reward function to extract the correct answer from the model's completion.
    ![alt text](image.png)

- Tested on the validation set, the finetuned model failed to generate response following the format required by the system prompt. The answer between <answer> </answer> tags was not wrapped in `\\boxed{}`.
    ![alt text](image-1.png)
    ![alt text](image-2.png)

### Raw_5e-5_Plain_ng7_bs28

#### Setup
- `train_dataset`: `raw_10k.json`
- `max_lora_rank`: `32`
- `learning_rate`: `5e-5`
- `batch_size`: `28`
- `num_generations`: `7`
- `max_grad_norm`: `0.1`
- `reward_funcs`: `format_reward`, `tag_count_reward`, `accuracy_reward`, `cosine_scaled_reward`
- `system_prompt`: "You are a helpful Math assistant. Carefully think step by step and enclose your response within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>"
- *Extra modification*: refactor the logic of answer extraction which focuses on the response between <answer> </answer> tags.

#### Results
- Rewards increased slowly as expected. Accuracy reward improved significantly, fluctuating around 0.7
    ![alt text](image-4.png)
    ![alt text](image-5.png)
- On the validation set, the finetuned model generated response following the format required by the system prompt.
    ![alt text](image-3.png)
- However, the finetuned model only achieved an accuracy of `73.84%` on the validation set, which is lower than the baseline performance of `76.98%`. This indicates that the model's performance has not improved after finetuning.

### Raw_5e-5_Plain_ng6_bs24

#### Setup
- `train_dataset`: `raw_10k.json`
- `max_lora_rank`: `32`
- `learning_rate`: `5e-5`
- `batch_size`: `24`
- `num_generations`: `6`
- `max_grad_norm`: `0.1`
- `reward_funcs`: `format_reward`, `tag_count_reward`, `accuracy_reward`, `cosine_scaled_reward`
- `system_prompt`: "You are a helpful Math assistant. Carefully think step by step and enclose your response within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>"

#### Results
- Slight difference in the reward function, while `num_generations=6, batch_size=24` is even a little bit better than `num_generations=7, batch_size=28`.
    ![alt text](image-6.png)

### Raw_5e-5_Plain_Len_Reward
#### Setup
- `train_dataset`: `raw_10k.json`
- `max_lora_rank`: `32`
- `learning_rate`: `5e-5`
- `batch_size`: `24`
- `num_generations`: `6`
- `max_grad_norm`: `0.1`
- `reward_funcs`: `format_reward`, `tag_count_reward`, `accuracy_reward`, `cosine_scaled_reward`, `length_reward`
- `system_prompt`: "You are a helpful Math assistant. Carefully think step by step and enclose your response within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>"
- *Extra modification*: add length reward function provided by the Kimi 1.5 tech report, to discourage overthinking and promote token efficiency.

#### Results
- Total reward seems a little better than the previous one, with an accuracy of `74.64%` on the validation set.
    ![alt text](image-7.png)

### Raw_r16
#### Setup
- `train_dataset`: `raw_10k.json`
- `max_lora_rank`: `16`
- `learning_rate`: `5e-5`
- `batch_size`: `24`
- `num_generations`: `6`
- `max_grad_norm`: `0.1`
- `reward_funcs`: `format_reward`, `tag_count_reward`, `accuracy_reward`, `cosine_scaled_reward`, `length_reward`
- `system_prompt`: "You are a helpful Math assistant. Carefully think step by step and enclose your response within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>"

#### Results
- Significant improvement is seen in all reward functions, yet `74.79%` accuracy on the validation set is still lower than the baseline performance of `76.98%`.
    ![alt text](image-8.png)

### Raw_r8
#### Setup
- `train_dataset`: `raw_10k.json`
- `max_lora_rank`: `8`
- `learning_rate`: `5e-5`
- `batch_size`: `24`
- `num_generations`: `6`
- `max_grad_norm`: `0.1`
- `reward_funcs`: `format_reward`, `tag_count_reward`, `accuracy_reward`, `cosine_scaled_reward`, `length_reward`
- `system_prompt`: "You are a helpful Math assistant. Carefully think step by step and enclose your response within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>"

#### Results
- Training rewards resemble those of `max_lora_rank=16`, yet the accuracy on the validation set is `74.09%`.
    ![alt text](image-9.png)

### Raw_r16_official_sampling
- `train_dataset`: `raw_10k.json`
- `max_lora_rank`: `8`
- `learning_rate`: `5e-5`
- `batch_size`: `24`
- `num_generations`: `6`
- `max_grad_norm`: `0.1`
- `reward_funcs`: `format_reward`, `tag_count_reward`, `accuracy_reward`, `cosine_scaled_reward`, `length_reward`
- `system_prompt`: "You are a helpful Math assistant. Carefully think step by step and enclose your response within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>"
- *Extra modification*: use official sampling method to generate the response instead of greedy search.
    - `temperature`: `0.9`
    - `top_p`: `0.95`
    - `top_k`: `20`
    - `min_p`: `0`

#### Results