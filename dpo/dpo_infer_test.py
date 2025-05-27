import json
import torch
from tqdm import tqdm
import time
import re
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from math_verify.parser import extract_target_from_pred, get_extraction_regexes, ExprExtractionConfig, LatexExtractionConfig

def extract_answer(content):
    """
    从模型回答中提取最终答案
    使用math_verify库提取数学表达式
    优先使用<answer>标签，如果没有则使用整个内容
    """
    try:        # 先使用<answer>标签提取内容
        match = re.search(r"<answer>(.*?)</answer>", content, re.DOTALL)
        if match:
            answer_text = match.group(1).strip()
        else:
            answer_text = content
            
        # 使用一系列正则表达式提取数学表达式和数字
        patterns = [
            r"\\boxed\{\\[d]?frac\{([^}]+)\}\{([^}]+)\}\}",
            r"\\boxed{([^}]+)}"
        ]
        
        # 尝试所有模式
        for pattern in patterns:
            match = re.search(pattern, answer_text)
            if match:
                # 对于分数格式特殊处理
                if "frac" in pattern:
                    numerator = match.group(1)
                    denominator = match.group(2)
                    answer_text = f"{numerator}/{denominator}"
                    break
                else:
                    answer_text = match.group(1).strip()
                    break
        
        
        extraction_config = [ExprExtractionConfig()]
        target_res = get_extraction_regexes(extraction_config)
        
        # 直接调用extract_target_from_pred函数，避开parse函数中的超时机制
        answer_parsed = extract_target_from_pred(
            answer_text,
            target_res=target_res,
            extraction_mode="first_match"
        )
        
        # 如果成功解析出数学表达式，返回第一个表达式
        if len(answer_parsed) > 0:
            result = answer_parsed[0]
            # 如果是浮点数，格式化为保留2位小数
            try:
                float_result = float(str(result))
                if float_result == int(float_result):
                    # 如果是整数，去掉小数部分
                    return str(int(float_result))
                else:
                    # 如果是小数，保留2位
                    return f"{float_result:.2f}"
            except (ValueError, TypeError):
                # 如果不是可转换为浮点数的类型，直接返回原字符串
                return str(result)
        
        # 如果上述模式都不匹配，尝试提取任何数字（作为最后手段）
        numbers = re.findall(r"\d+(?:\.\d+)?", answer_text)
        if numbers:
            return numbers[-1]  # 返回最后找到的数字
    
    except Exception as e:
        print(f"提取答案时出错: {e}")
    
    # 如果无法解析，返回原始内容
    return content

def predict(message, model, tokenizer):
    """单个样本处理函数"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 应用聊天模板
    text = tokenizer.apply_chat_template(
        message,
        tokenize=False,
        add_generation_prompt=True
    )
    
    # 转为模型输入
    model_inputs = tokenizer([text], return_tensors="pt").to(device)
    
    # 生成回答
    with torch.no_grad():  # 关闭梯度计算以节省内存
        generated_ids = model.generate(
            model_inputs.input_ids,
            max_new_tokens=2048
        )
    
    # 提取新生成的token
    new_tokens = generated_ids[0][len(model_inputs.input_ids[0]):]

    # 解码为文本
    response = tokenizer.decode(new_tokens, skip_special_tokens=True)
    
    return response

# 将主程序代码移到这个条件块中
if __name__ == '__main__':
    # 加载测试数据
    test_json_new_path = "train_100.json"
    with open(test_json_new_path, 'r', encoding='utf-8') as file:
        test_data = json.load(file)

    # 加载模型和tokenizer
    tokenizer = AutoTokenizer.from_pretrained("./Qwen\Qwen3-0___6B/", use_fast=False, trust_remote_code=True)
    tokenizer.padding_side = 'left'  # 确保使用左侧填充
    model = AutoModelForCausalLM.from_pretrained("./Qwen\Qwen3-0___6B/", device_map="cuda", torch_dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(model, model_id="./output/Qwen_DPO_final/")

    # 准备结果文件
    results = []
    total_samples = len(test_data)
    start_time = time.time()

    # 逐个处理样本
    with open("submit.csv", 'w', encoding='utf-8') as file:
        count = 0
        for i, row in tqdm(enumerate(test_data)):
            # 准备输入
            instruction = row['instruction']
            input_value = row['question']
            id = row['id']

            answer = row['answer']
            
            messages = [
                {"role": "system", "content": f"{instruction}"},
                {"role": "user", "content": f"{input_value}"}
            ]
            
            # 预测样本
            response = predict(messages, model, tokenizer)
            print()
            print(f"样本 {i + 1}/{total_samples} - ID: {id} - 预测: {response}")
            # 提取答案
            extracted_answer = extract_answer(response)
            extracted_answer = extracted_answer.replace('\n', ' ')
            # 提取训练集答案
            answer = extract_answer(answer)
            answer = answer.replace('\n', ' ')

            # 尝试将答案转换为数字进行比较
            try:
                extracted_num = float(extracted_answer)
                answer_num = float(answer)
                # 比较数字是否相等（允许小的浮点误差）
                if abs(extracted_num - answer_num) < 1e-6:
                    count += 1
                    match_status = "✓"
                else:
                    match_status = "✗"
            except ValueError:
                # 如果无法转换为数字，则使用字符串比较
                if extracted_answer == answer:
                    count += 1
                    match_status = "✓"
                else:
                    match_status = "✗"
            
            print(f"样本 {i + 1}/{total_samples} - ID: {id} - 预测: {extracted_answer} - 正确: {answer} - 正确率: {count / (i + 1) * 100:.2f}% - {match_status}")

            # 写入CSV
            file.write(f"{id},{extracted_answer}\n")
        file.write(f"{count / (i + 1) * 100:.2f}%\n")
    print("处理完成，结果已保存到 submit.csv")