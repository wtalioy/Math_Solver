# GRPO Report

## Shared Setup
- **Pretrained Model:** `Qwen3-0.6B(Instruct)`
- **Training Framework:** `Huggingface` + `Trl` + `Accelerate`
- **Training Method:** `LoRA`
- **Training Set Size:** `10000`

## Baseline performance
- **Validation Set Accuracy:** 
    - `val_2k.json`
        - "You are a helpful Math assistant. Carefully think step by step and enclose your response within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>": `73.39%`
    - `val_300.json`
        - "You are a helpful Math assistant. Carefully think step by step and enclose your response within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>": `74.33%`
        - "这是小学数学1-6年级的校内题目，无需进行分析，请直接输出数字答案，不带单位。": `74.67%`
        - "You are a helpful Math assistant. 这是小学数学1-6年级的校内题目，请直接输出数字答案，不带单位。": `75.00%`

## Problems
- The task of solving math problems is extremely sensitive to the randomness of model generation, as the model may take different inference paths, leading to different results and, importantly, high variation in accuracy on validation sets.
- To address this, one effective way is greedy search (or nearly greedy search), i.e., set `temperature` to `0`.

## Ablation Experiments

### Raw_r32P_Latex

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

### Raw_r32_Plain_ng7_bs28

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

### Raw_r32_Plain_ng6_bs24

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

### Raw_r32_Plain_Len_Reward
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
- `val_300.json` accuracy: `75.33%`

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
- `val_300.json` accuracy: `72.33%`

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
- `val_300.json` accuracy: `72.33%`

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
- *Extra modification*: use official sampling parameters during training.
    - `temperature`: `0.9`
    - `top_p`: `0.95`
    - `top_k`: `20`
    - `min_p`: `0`

#### Results
- Training rewards showed no significant difference.
    ![alt text](image-10.png)
- `val_300.json` accuracy: `69.67%`

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
    - `cosine_min_value_correct`: `1.0`
    - `cosine_max_value_correct`: `2.0`

#### Results
- Performance at response format dropped a little bit.
    ![alt text](image-11.png)
- `val_300.json` accuracy: `71.00%`

### Raw_r32_repetition_penalty
#### Setup
- `train_dataset`: `raw_10k.json`
- `max_lora_rank`: `32`
- `learning_rate`: `5e-5`
- `batch_size`: `24`
- `num_generations`: `6`
- `max_grad_norm`: `0.1`
- `reward_funcs`: `format_reward`, `tag_count_reward`, `accuracy_reward`, `cosine_scaled_reward`, `length_reward`, `repetition_penalty_reward`
- `system_prompt`: "You are a helpful Math assistant. Carefully think step by step and enclose your response within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>"
- *Extra modification*: add repetition penalty reward function to penalize the model for generating repeated tokens.
    - `ngram_size`: `40`
    - `max_penalty`: `-0.05`

#### Results
- Better training rewards than that without repetition penalty
    ![alt text](image-12.png)
- `val_300.json` accuracy: `68.33%`

### Raw_r32_new_prompt
#### Setup
- `train_dataset`: `raw_10k.json`
- `max_lora_rank`: `32`
- `learning_rate`: `5e-5`
- `batch_size`: `24`
- `num_generations`: `6`
- `max_grad_norm`: `0.1`
- `reward_funcs`: `accuracy_reward`, `cosine_scaled_reward`, `length_reward`
- `system_prompt`: "You are a helpful Math assistant. 这是小学数学1-6年级的校内题目，请直接输出数字答案，不带单位。"

#### Results
- No big difference in training rewards.
    ![alt text](image-13.png)
- `val_300.json` accuracy: `69.33%`