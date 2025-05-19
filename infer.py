import json
import torch
import re
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import os
import numpy as np
from multiprocessing import freeze_support


def predict_batch(batch_messages, model, tokenizer, batch_size=32):
    """批量处理样本，加速推理，使用更优的生成参数"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    all_responses = []

    # 按批次处理
    for i in range(0, len(batch_messages), batch_size):
        current_batch = batch_messages[i:i+batch_size]
        texts = []

        # 为每个样本准备输入
        for msg in current_batch:
            text = tokenizer.apply_chat_template(
                msg,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True  # 启用Qwen3的思考模式
            )
            texts.append(text)

        # 批量处理输入，确保设置attention_mask
        model_inputs = tokenizer(
            texts, return_tensors="pt", padding=True).to(device)

        # 确保attention_mask已经设置
        if 'attention_mask' not in model_inputs:
            print("手动创建attention_mask")
            # 创建attention_mask：非填充位置为1，填充位置为0
            attention_mask = torch.ne(
                model_inputs.input_ids, tokenizer.pad_token_id).long()
            model_inputs['attention_mask'] = attention_mask

        # 使用优化的生成参数
        generated_ids = model.generate(
            model_inputs.input_ids,
            attention_mask=model_inputs.attention_mask,  # 显式传递attention_mask
            max_new_tokens=384,  # 增加生成长度以获得更完整的推理过程
            num_beams=2,  # 使用beam search提高质量
            do_sample=True,  # 启用采样以提高创造性思考
            temperature=0.6,  # Qwen3文档推荐的思考模式temperature
            top_p=0.95,  # Qwen3文档推荐的思考模式top_p
            top_k=20,  # 控制词汇多样性
            repetition_penalty=1.1,  # 减少重复
            pad_token_id=tokenizer.pad_token_id,
        )

        # 处理生成结果
        for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids):
            generated_id = output_ids[len(input_ids):]

            # 提取思考内容和回答
            try:
                # 尝试提取思考内容（如果使用<think>标签）
                response_text = tokenizer.decode(
                    generated_id, skip_special_tokens=True)

                # 检查是否包含思考内容标记
                think_match = re.search(
                    r'<think>(.*?)</think>(.*)', response_text, re.DOTALL)
                if think_match:
                    # 有显式思考内容，提取回答部分
                    response = think_match.group(2).strip()
                else:
                    # 没有显式思考内容，使用整个回答
                    response = response_text

                all_responses.append(response)
            except Exception as e:
                print(f"解析生成内容时出错: {e}")
                # 出错时使用原始方法
                response = tokenizer.decode(
                    generated_id, skip_special_tokens=True)
                all_responses.append(response)

    return all_responses


def extract_answer(cot_response):
    """增强的答案提取逻辑，更好地处理各种回答形式"""
    # 1. 尝试找结构化答案格式
    # 扩展的答案模式匹配，支持更多表达方式
    answer_patterns = [
        # 中文标准答案格式
        r"最终答案是[:：]\s*([0-9]+\.?[0-9]*)",
        r"答案是[:：]\s*([0-9]+\.?[0-9]*)",
        r"答案[:：]\s*([0-9]+\.?[0-9]*)",
        r"答案为[:：]\s*([0-9]+\.?[0-9]*)",
        r"结果是[:：]\s*([0-9]+\.?[0-9]*)",
        r"计算结果为[:：]\s*([0-9]+\.?[0-9]*)",
        r"计算得[:：]\s*([0-9]+\.?[0-9]*)",
        r"等于[:：]\s*([0-9]+\.?[0-9]*)",
        r"得出[:：]\s*([0-9]+\.?[0-9]*)",
        # 格式化输出匹配
        r"\\boxed{([0-9]+\.?[0-9]*)}",  # LaTeX风格答案
        r"\*\*答案[：:]\s*([0-9]+\.?[0-9]*)\*\*",  # Markdown加粗风格
        # 数字+单位格式匹配
        r"最终结果是\s*([0-9]+\.?[0-9]*)[\s]*(个|只|本|张|元|千克|克|米|厘米|平方米|立方米|吨)",
        r"答案是\s*([0-9]+\.?[0-9]*)[\s]*(个|只|本|张|元|千克|克|米|厘米|平方米|立方米|吨)",
    ]

    # 首先尝试精确匹配
    for pattern in answer_patterns:
        match = re.search(pattern, cot_response)
        if match:
            return match.group(1)  # 返回第一个捕获组，即答案数字

    # 2. 如果没有明确的答案格式，尝试智能提取答案
    # 提取所有数字
    numbers = re.findall(r"([0-9]+\.?[0-9]*)", cot_response)

    if not numbers:
        # 3. 没有找到任何数字，返回原始响应的末尾部分
        return cot_response[-30:].strip()

    # 优先考虑句子末尾的数字
    sentences = re.split(r'[。.!！?？\n]', cot_response)
    for sentence in reversed(sentences):  # 从后向前检查句子
        if sentence.strip():  # 非空句子
            nums_in_sentence = re.findall(r"([0-9]+\.?[0-9]*)", sentence)
            if nums_in_sentence:
                return nums_in_sentence[-1]  # 返回句子中最后一个数字

    # 如果上述方法都失败，返回文本中最后一个数字
    return numbers[-1]


def enable_print_for_debugging():
    """在调试时开启详细打印"""
    # 全局变量控制是否打印详细信息
    global VERBOSE
    VERBOSE = os.environ.get("VERBOSE", "false").lower() == "true"
    if VERBOSE:
        print("已开启详细调试模式")
    return VERBOSE


# 初始化调试打印
VERBOSE = enable_print_for_debugging()

# 加载测试数据
test_json_new_path = "test.json"
print(f"加载测试数据: {test_json_new_path}")

with open(test_json_new_path, 'r', encoding='utf-8') as file:
    test_data = json.load(file)

# 模型和检查点路径
model_id = "Qwen/Qwen3-0.6B"
# 修改为使用存在的最新检查点或final目录
cot_checkpoint_path = "./output/Qwen3_COT/final/"

# 加载基础模型
print(f"加载基础模型: {model_id}")
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
# 设置左侧填充以避免警告
tokenizer.padding_side = 'left'
# 确保pad_token和eos_token不同
if tokenizer.pad_token_id == tokenizer.eos_token_id:
    print("设置特殊pad_token以区分于eos_token")
    tokenizer.pad_token = '[PAD]'
print("已设置tokenizer为左侧填充模式")

print(f"加载模型参数...")
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True)

# 加载训练好的COT checkpoint
if os.path.exists(cot_checkpoint_path):
    print(f"加载COT checkpoint: {cot_checkpoint_path}")
    model = PeftModel.from_pretrained(model, model_id=cot_checkpoint_path)
else:
    print(f"警告：找不到COT checkpoint: {cot_checkpoint_path}")
    print("请先运行训练脚本 qwen_ft.py 生成COT模型")

# 设置推理批大小
GPU_VRAM = int(torch.cuda.get_device_properties(0).total_memory / (1024**3))
# 根据显存大小调整批量大小
if GPU_VRAM >= 24:
    BATCH_SIZE = 32
elif GPU_VRAM >= 16:
    BATCH_SIZE = 16
elif GPU_VRAM >= 8:
    BATCH_SIZE = 8
else:
    BATCH_SIZE = 4

print(f"检测到GPU显存: {GPU_VRAM}GB, 设置批大小为{BATCH_SIZE}")
batch_messages = []
batch_ids = []

# 批量准备数据
for row in test_data:
    instruction = row['instruction']
    input_value = row['question']
    id = row['id']

    # 保存ID以便后续写入结果
    batch_ids.append(id)

    # 优化系统提示，添加更详细的思维链指导
    system_prompt = f"{instruction} 请先详细分析题目，展示完整的思考过程和计算步骤，然后给出准确的最终答案。"
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"{input_value}"}
    ]

    batch_messages.append(messages)

# 分批次处理
print(f"开始批量推理，共{len(batch_messages)}个样本，批大小={BATCH_SIZE}")
total_batches = (len(batch_messages) + BATCH_SIZE - 1) // BATCH_SIZE
with tqdm(total=len(batch_messages)) as pbar:
    with open("submit.csv", 'w', encoding='utf-8') as file:
        # 每次处理BATCH_SIZE个样本
        for i in range(0, len(batch_messages), BATCH_SIZE):
            current_batch_msgs = batch_messages[i:i+BATCH_SIZE]
            current_batch_ids = batch_ids[i:i+BATCH_SIZE]

            # 批量预测
            batch_responses = predict_batch(
                current_batch_msgs, model, tokenizer, BATCH_SIZE)

            # 处理每个回答并写入文件
            for id, cot_response in zip(current_batch_ids, batch_responses):
                # 从思维链回答中提取最终答案
                final_answer = extract_answer(cot_response)

                # 调试输出
                if VERBOSE and i % 10 == 0:
                    print(f"\nID: {id}")
                    print(f"回答: {cot_response[:200]}...")
                    print(f"提取答案: {final_answer}")

                # 将最终答案写入提交文件
                file.write(f"{id},{final_answer}\n")

            # 更新进度条
            pbar.update(len(current_batch_msgs))

print("推理完成，结果已保存到submit.csv")
