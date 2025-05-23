import json
import random
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from tqdm import tqdm

# 加载 Qwen 模型
model_path = "./Qwen/Qwen2.5-0.5B-Instruct/"
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype=torch.bfloat16)
# model.enable_input_require_grads()

# 简单提示词生成错误反例
def generate_qwen_cot(question: str) -> str:
    prompt = f"""请逐步思考并回答以下问题：
{question}
以最终答案：<数字>结尾。"""
    input_ids = tokenizer(prompt, return_tensors="pt").to(model.device)
    output = model.generate( #注意这里的参数，我特地调高了temperature和top_p，让输出更乱七八糟
        **input_ids,
        max_new_tokens=200,
        temperature=0.8,
        do_sample=True,
        top_p=0.95,
        eos_token_id=tokenizer.eos_token_id,
        repetition_penalty=1.1 #加上这个才不会重复
    )
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response.replace(prompt, '').strip()

# 加载原始数据
with open("train_10k_ds.json", "r", encoding="utf-8") as f:
    data = json.load(f)
    
data = data[:3800] #生成太多太花时间

results = []
for item in tqdm(data, desc="生成反例 COT"):
    try:
        question = item["question"]
        pos_cot = item["answer"]
        neg_cot = generate_qwen_cot(question)

        results.append({
            "id": item["id"],
            "question": question,
            "cot_pos": pos_cot,
            "cot_neg": neg_cot
        })
    except Exception as e:
        print(f"[WARN] 第 {item['id']} 条生成失败：{e}")
        results.append({
            "id": item["id"],
            "question": item["question"],
            "cot_pos": item["answer"],
            "cot_neg": f"步骤1：示例。\n最终答案：{random.randint(1,999)}"
        })

# 保存结果
with open("train_10k_with_neg_cot.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print("✅ 已保存到 train_10k_with_neg_cot.json")
