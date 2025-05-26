import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import re
from tqdm import tqdm

MODEL_PATH = "Qwen/Qwen2.5-0.5B-Instruct"
PEFT_PATH = "models/qwen2.5-0.5b-finetuned/checkpoint-1250/"
TEST_PATH = "data/test.json"
SUBMIT_PATH = "submit.csv"

# BATCH_SIZE = 32 # 可以根据您的GPU显存调整批次大小
BATCH_SIZE = 8  # 临时用于快速测试答案提取逻辑


def extract_final_answer(text):
    # 定义提取数字/分数的正则表达式
    # 这个 regex 会匹配整数，小数 (如 3.14)，或分数 (如 1/2)
    number_regex = r"\d+\.\d+|\d+/\d+|\d+"

    extracted_text = None

    # 1. 提取 \boxed{...} 中的内容
    match_boxed = re.search(r"\\boxed\{(.*?)\}", text)
    if match_boxed:
        extracted_text = match_boxed.group(1).strip()

    # 2. 如果没有 \boxed{}，则提取"最终答案"后面的内容
    if extracted_text is None:
        match_final_answer = re.search(r"最终答案[：: ]*(.*?)(?:[。，,\n]|$)", text)
        if match_final_answer:
            extracted_text = match_final_answer.group(1).strip()

    # 3. 如果以上都找不到，则使用整个文本
    if extracted_text is None:
        extracted_text = text

    # 从提取到的文本中查找所有数字/分数
    matches = re.findall(number_regex, extracted_text)

    # 返回最后一个找到的数字/分数，如果没有找到则返回原始文本（作为兜底）
    if matches:
        return matches[-1].strip()
    else:
        # 如果在提取的片段中没有找到数字，尝试在整个原始文本中查找最后的数字
        all_numbers_in_text = re.findall(number_regex, text)
        if all_numbers_in_text:
            return all_numbers_in_text[-1].strip()
        else:
            return text.strip()  # 实在找不到数字，返回清理后的原始文本

# 修改 predict 函数以支持批量处理


def predict_batch(batch_messages, model, tokenizer):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # 将批次消息列表应用 chat template 并 token化
    texts = [tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True) for messages in batch_messages]

    # 批量 tokenization，需要 padding
    model_inputs = tokenizer(texts, return_tensors="pt",
                             padding=True, truncation=True).to(device)

    # 批量生成
    generated_ids = model.generate(
        **model_inputs,  # 传递 input_ids 和 attention_mask 等
        max_new_tokens=1024,
        pad_token_id=tokenizer.eos_token_id  # 设置 pad_token_id 以正确处理 padding
    )

    # 解码生成结果
    # 注意：这里需要根据输入的实际长度来切分生成的输出，以去除输入部分的 token
    decoded_responses = []
    # 对于 batched generate 输出，generated_ids 的形状是 (batch_size, sequence_length)
    # 其中 sequence_length = input_length + generated_length
    # 我们需要根据每个样本原始输入的长度来提取生成的部分
    input_lengths = model_inputs.input_ids.shape[1]  # 批量处理后所有输入的长度（包含了padding）

    for i in range(generated_ids.shape[0]):
        # 提取当前样本的生成部分
        # generated_ids[:, input_lengths:] 适用于右侧 padding 的情况
        generated_tokens = generated_ids[i, input_lengths:]
        response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        decoded_responses.append(response)

    return decoded_responses


def main():
    print("加载模型和分词器...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH, use_fast=False, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, device_map="auto", torch_dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(model, PEFT_PATH)
    model.eval()

    print("加载测试数据...")
    with open(TEST_PATH, 'r', encoding='utf-8') as file:
        test_data = json.load(file)

    submit_results = []  # 存储所有结果
    batch_data = []

    print(f"总共 {len(test_data)} 条测试数据，使用批次大小 {BATCH_SIZE}")

    for i, row in enumerate(tqdm(test_data)):
        instruction = row['instruction']
        input_value = row['question']
        id = row['id']
        messages = [
            {"role": "system", "content": f"{instruction}"},
            {"role": "user", "content": f"{input_value}"}
        ]
        batch_data.append({'id': id, 'messages': messages})  # 将数据添加到批次列表

        # 如果批次满了或者处理到最后一条数据
        if len(batch_data) == BATCH_SIZE or i == len(test_data) - 1:
            # 处理当前批次
            batch_ids = [item['id'] for item in batch_data]
            batch_messages = [item['messages'] for item in batch_data]

            batch_responses = predict_batch(batch_messages, model, tokenizer)

            # 将批次结果添加到总结果列表
            for j, response in enumerate(batch_responses):
                answer = extract_final_answer(response)
                # 可以取消注释用于快速验证提取逻辑
                # print(f"ID: {batch_ids[j]}, Response:\n{response}\nExtracted Answer: {answer}\n---") # 修改打印语句以显示完整的 Response
                answer = answer.replace('\n', ' ').replace(',', '')
                submit_results.append(f"{batch_ids[j]},{answer}")

            batch_data = []  # 清空批次列表以便处理下一个批次

        # if i >= BATCH_SIZE * 2 - 1: # 临时用于处理少量数据后提前退出
        #     print("已处理少量数据，提前退出以验证提取逻辑...")
        #     break

    print(f"预测结果已收集 {len(submit_results)} 条")
    print(f"预测结果将保存到 {SUBMIT_PATH}")
    with open(SUBMIT_PATH, 'w', encoding='utf-8') as file:
        # 写入CSV头部
        file.write("id,answer\n")
        for result_line in submit_results:
            file.write(result_line + '\n')

    print("预测完成。")


if __name__ == "__main__":
    main()
