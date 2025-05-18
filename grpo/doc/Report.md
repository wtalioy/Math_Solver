# GRPO Report

## Shared Setup
- **Pretrained Model:** `Qwen3-0.6B(Instruct)`
- **Training Framework:** `Huggingface` + `Trl` + `Accelerate`
- **Training Method:** `LoRA`
- **Training Set Size:** `10000`

## Baseline performance
- **Validation Set Accuracy:** 
    - `val_2k.json`
        - "You are a helpful Math assistant. Carefully think step by step and enclose your response within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>": `74.04%`
    - `val_200.json`
        - "You are a helpful Math assistant. Carefully think step by step and enclose your response within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>": `74.00%`
        - "这是小学数学1-6年级的校内题目，无需进行分析，请直接输出数字答案，不带单位。": `75.00%`


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
- However, the finetuned model only achieved an accuracy of `73.84%` on the validation set, which is lower than the baseline performance of `74.04%`.

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
- Total reward seems a little better than the previous one.
    ![alt text](image-7.png)
- `val.json` accuracy: `74.64%`

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
- Significant improvement is seen in all reward functions.
    ![alt text](image-8.png)
- `val.json` accuracy: `74.79%`
- `val_200.json` accuracy: `81.00%`

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
- Training rewards resemble those of `max_lora_rank=16`.
    ![alt text](image-9.png)
- `val.json` accuracy: `74.09%`

### Raw_r16_official_sampling
#### Setup
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
- Training rewards showed no significant difference.
    ![alt text](image-10.png)
- `val.json` accuracy: `71.19%`

### Raw_r16_cosine_modified
#### Setup
- `train_dataset`: `raw_10k.json`
- `max_lora_rank`: `8`
- `learning_rate`: `5e-5`
- `batch_size`: `24`
- `num_generations`: `6`
- `max_grad_norm`: `0.1`
- `reward_funcs`: `format_reward`, `tag_count_reward`, `accuracy_reward`, `cosine_scaled_reward`, `length_reward`
- `system_prompt`: "You are a helpful Math assistant. Carefully think step by step and enclose your response within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>"
- *Extra modification*: modify the hyperparameters of cosine reward function
    - `cosine_max_len`: `1000`
    - `cosine_min_value_wrong`: `-10.0`
    - `cosine_max_value_wrong`: `0`
    - `cosine_min_value_correct`: `2.0`
    - `cosine_max_value_correct`: `1.0`

#### Results
- Performance at response format dropped a little bit.
    ![alt text](image-11.png)
- `val_200.json` accuracy: `80.50%`