import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import re
from tqdm import tqdm

MODEL_PATH = "Qwen/Qwen3-0.6B"
PEFT_PATH = "../models/qwen3-0.6b-finetuned/"
TEST_PATH = "../data/test.json"
SUBMIT_PATH = "../submit.csv"


def extract_final_answer(text):
    # 1. 提取 \boxed{...} 或 \boxed{...}
    match = re.search(r"\\boxed\{([^}]+)\}", text)
    if match:
        return match.group(1).strip()
    # 2. 提取"最终答案"后面的内容
    match = re.search(r"最终答案[：: ]*([\d./]+)", text)
    if match:
        return match.group(1).strip()
    # 3. 提取所有数字/分数，取最后一个
    matches = re.findall(r"\d+\.\d+|\d+/\d+|\d+", text)
    if matches:
        return matches[-1].strip()
    # 4. 兜底：返回原文
    return text.strip()


def predict(messages, model, tokenizer):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(device)
    generated_ids = model.generate(
        model_inputs.input_ids,
        max_new_tokens=512
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    response = tokenizer.batch_decode(
        generated_ids, skip_special_tokens=True)[0]
    return response


def main():
    print("加载模型和分词器...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH, use_fast=False, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, device_map="auto", torch_dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(model, model_id=PEFT_PATH)
    model.eval()

    print("加载测试数据...")
    with open(TEST_PATH, 'r', encoding='utf-8') as file:
        test_data = json.load(file)

    with open(SUBMIT_PATH, 'w', encoding='utf-8') as file:
        for row in tqdm(test_data):
            instruction = row['instruction']
            input_value = row['question']
            id = row['id']
            messages = [
                {"role": "system", "content": f"{instruction}"},
                {"role": "user", "content": f"{input_value}"}
            ]
            response = predict(messages, model, tokenizer)
            answer = extract_final_answer(response)
            answer = answer.replace('\n', ' ').replace(',', '')
            file.write(f"{id},{answer}\n")
    print(f"预测结果已保存到 {SUBMIT_PATH}")


if __name__ == "__main__":
    main()
