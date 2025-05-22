import json
import torch
from tqdm import tqdm
from modelscope import AutoTokenizer
from transformers import AutoModelForCausalLM
from peft import PeftModel
import re


def predict(messages_batch, model, tokenizer, device):
    # 将一个批次的 messages 转换为模型输入
    texts = [tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    ) for messages in messages_batch]

    # 对文本进行编码和填充
    model_inputs = tokenizer(
        texts, return_tensors="pt", padding=True).to(device)

    # 生成响应
    generated_ids = model.generate(
        model_inputs.input_ids,
        attention_mask=model_inputs.attention_mask,
        max_new_tokens=512  # 保持与之前一致，或根据需要调整
    )

    # 解码生成的响应
    # 需要处理批次输出，并移除输入部分的 token
    responses = []
    for i in range(generated_ids.shape[0]):
        # 找到输入部分的长度
        input_len = model_inputs.input_ids[i].shape[0]
        # 解码只包含生成部分的 token
        response = tokenizer.decode(
            generated_ids[i, input_len:], skip_special_tokens=True)
        responses.append(response)

    return responses


def extract_answer(response):
    # 1. 尝试查找特定的结束语后的数字
    # 查找 "最终答案：", "最终结果：", "答案是：", "结果是：" 后紧跟的数字
    specific_phrase_match = re.search(
        r"(?:最终答案|最终结果|答案是|结果是)[:：\s]*(-?\d+\.?\d*)", response)
    if specific_phrase_match:
        return specific_phrase_match.group(1).strip()

    # 2. 如果未找到，尝试提取文本中最后出现的数字
    # 使用正则表达式匹配最后出现的数字 (整数或小数，可能带负号)，忽略其后非数字/小数点/负号的字符直到字符串结束
    last_number_match = re.search(r"(-?\d+\.?\d*)[^0-9.-]*$", response)
    if last_number_match:
        return last_number_match.group(1).strip()

    # 3. Fallback: 如果以上都找不到，返回空字符串
    return ""


test_json_new_path = "test.json"

with open(test_json_new_path, 'r', encoding='utf-8') as file:
    test_data = json.load(file)

tokenizer = AutoTokenizer.from_pretrained(
    "./Qwen/Qwen2___5-0___5B-Instruct/", use_fast=False, trust_remote_code=True, padding_side='left')
model = AutoModelForCausalLM.from_pretrained(
    "./Qwen/Qwen2___5-0___5B-Instruct/", device_map="auto", torch_dtype=torch.bfloat16)
device = "cuda"  # 定义设备
model.to(device)  # 将模型移动到指定设备

# 定义批次大小
batch_size = 16  # 可以根据您的硬件调整

with open("submit.csv", 'w', encoding='utf-8') as submit_file, \
        open("full_responses.txt", 'w', encoding='utf-8') as response_file:
    # 移除 [:20] 以处理所有数据
    # 使用 tqdm 包装批次循环
    for i in tqdm(range(0, len(test_data), batch_size)):
        # 获取当前批次的数据
        batch_data = test_data[i:i + batch_size]

        # 构建当前批次的 messages
        messages_batch = []
        original_indices = []  # 存储原始数据索引，用于写回 submit.csv 和 full_responses.txt
        for row in batch_data:
            instruction = row['instruction']
            input_value = row['question']
            id = row['id']
            original_indices.append((id, row))  # 存储 id 和原始行数据

            messages = [
                {"role": "system", "content": f"{instruction}"},
                # Few-shot example 1
                {"role": "user", "content": "计算 5 + 3 × 2"},
                {"role": "assistant", "content": "详细思考过程：\n1. 根据数学运算顺序，先计算乘法。\n2. 3 × 2 = 6。\n3. 然后计算加法。\n4. 5 + 6 = 11。\n最终答案：<answer>11</answer>"},
                # Few-shot example 2
                {"role": "user", "content": "小明有 10 个苹果，吃了 4 个，还剩几个？"},
                {"role": "assistant", "content": "详细思考过程：\n1. 小明原来有 10 个苹果。\n2. 他吃了 4 个。\n3. 剩下的苹果数量是原来的减去吃的。\n4. 10 - 4 = 6。\n最终答案：<answer>6</answer>"},
                # The actual problem
                {"role": "user", "content": f"{input_value} 请针对每一道题目，都一步一步地详细思考，并给出完整的解答过程和最终答案。"}
            ]
            messages_batch.append(messages)

        # 对当前批次进行预测
        responses_batch = predict(messages_batch, model, tokenizer, device)

        # 处理批次中的每个响应
        for j, response in enumerate(responses_batch):
            original_id, original_row = original_indices[j]

            # 提取答案
            extracted_answer = extract_answer(response)

            # 将提取的答案写入 submit 文件
            submit_file.write(f"{original_id},{extracted_answer}\n")

            # 将完整响应写入 response 文件
            response_file.write(f"--- ID: {original_id} ---\n{response}\n\n")
