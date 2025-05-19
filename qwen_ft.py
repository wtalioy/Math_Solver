import json
import torch
from modelscope import snapshot_download, AutoTokenizer
from swanlab.integration.huggingface import SwanLabCallback
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForSeq2Seq
from transformers.trainer_utils import set_seed
import swanlab
import os
import random
from multiprocessing import freeze_support


def create_math_cot_template(problem, answer):
    """
    创建更加详细的数学思维链模板，提供更具体的步骤和解题思路
    """
    templates = [
        f"""让我一步一步思考：
1. 我需要理解问题：{problem}
2. 分析已知条件：
   - 提取关键信息和数值
   - 确定需要计算的目标
3. 确定解题方法和公式
4. 执行计算步骤：
   - 根据公式进行计算
   - 注意单位换算和小数点
5. 检查计算结果是否合理
最终答案是：{answer}""",

        f"""我来详细解答这个问题：
首先，我们面对的问题是：{problem}
分析：
- 提取题目中的关键数值和条件
- 明确题目要求我们计算什么
解题思路：
1. 列出适当的数学公式或方程
2. 将已知数值代入公式
3. 进行计算并得出结果
计算过程：
- 按照上述思路进行详细计算
- 注意计算过程中的单位和精度
检验答案的合理性
答案是：{answer}""",

        f"""解题过程：
【问题分析】
题目：{problem}
需要计算的是什么？题目的核心是什么类型的问题？
【列出已知条件】
从题目中提取出所有已知的数值和条件
【确定方程或公式】
根据问题类型选择合适的数学方法
【求解步骤】
1. 进行第一步计算...
2. 进行第二步计算...
3. 进行最终计算...
【最终结果】
经过以上步骤的计算，得出最终答案为：{answer}"""
    ]

    # 随机选择一个模板，增加训练数据的多样性
    return random.choice(templates)


def process_func(example):
    """
    将数据集进行预处理，添加COT（思维链）提示，使用优化的思维链模板
    """
    MAX_LENGTH = 512  # 增加最大长度以容纳更详细的思维过程
    input_ids, attention_mask, labels = [], [], []

    # 构建系统指令，提供更明确的COT引导
    system_prompt = f"{example['instruction']} 在回答前，请详细分析问题，展示完整的思考过程和计算步骤，然后给出最终答案。"

    # 添加思维链提示，让模型先展示解题步骤再给出答案
    instruction = tokenizer(
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{example['question']}<|im_end|>\n<|im_start|>assistant\n",
        add_special_tokens=False,
    )

    # 使用优化的思维链模板
    cot_answer = create_math_cot_template(
        example['question'], example['answer'])

    response = tokenizer(cot_answer, add_special_tokens=False)
    input_ids = instruction["input_ids"] + \
        response["input_ids"] + [tokenizer.pad_token_id]
    attention_mask = (
        instruction["attention_mask"] + response["attention_mask"] + [1]
    )
    labels = [-100] * len(instruction["input_ids"]) + \
        response["input_ids"] + [tokenizer.pad_token_id]

    if len(input_ids) > MAX_LENGTH:
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]

    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


if __name__ == '__main__':
    # 解决Windows多进程问题
    freeze_support()

    # 设置随机种子以确保结果可复现
    set_seed(42)

    print("使用Qwen3-0.6B模型...")
    # 直接使用模型ID，不进行本地路径检查
    model_id = "Qwen/Qwen3-0.6B"

    # 使用transformers API加载模型和tokenizer
    print(f"加载tokenizer: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    print(f"加载模型: {model_id}")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True)

    model.enable_input_require_grads()  # 开启梯度检查点

    # 加载训练数据
    train_json_new_path = "train.json"
    print(f"加载训练数据: {train_json_new_path}")

    with open(train_json_new_path, 'r', encoding='utf-8') as file:
        train_data = json.load(file)

    print(f"处理训练数据: {len(train_data)}条样本")
    train_dataset = []
    for d in train_data:
        train_dataset.append(process_func(d))

    # 配置LoRA参数
    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj",
                        "o_proj", "gate_proj", "up_proj", "down_proj"],
        inference_mode=False,  # 训练模式
        r=16,  # 增加Lora秩以提高模型容量
        lora_alpha=32,  # Lora alpha参数
        lora_dropout=0.05,  # 降低dropout以减少过拟合
    )

    model = get_peft_model(model, config)

    # 优化训练参数
    args = TrainingArguments(
        output_dir="./output/Qwen3_COT",  # 输出目录
        per_device_train_batch_size=8,  # 增加批大小以加快训练
        gradient_accumulation_steps=2,  # 减少梯度累积步数以适应更大的批大小
        logging_steps=5,  # 更频繁的日志记录
        num_train_epochs=3,  # 减少训练轮次，重点在于质量而非数量
        save_steps=500,  # 更频繁地保存检查点
        learning_rate=2e-4,  # 稍微提高学习率
        lr_scheduler_type="cosine",  # 使用余弦学习率调度
        warmup_ratio=0.03,  # 添加预热阶段
        save_on_each_node=True,
        gradient_checkpointing=True,
        report_to="none",
        # 启用混合精度训练以加速
        fp16=True,  # 启用混合精度训练
        fp16_opt_level="O1",  # 设置混合精度优化级别
        dataloader_num_workers=0,  # Windows下设为0避免多进程问题
        dataloader_pin_memory=True,  # 使用内存固定以加速数据传输
    )

    # 配置SwanLab回调
    swanlab_callback = SwanLabCallback(
        project="Qwen3-0.6B-fintune-COT-Optimized",  # 更新项目名称
        experiment_name="Qwen3-0.6B-COT-Improved",  # 标记为改进版本
        config={
            "model": "Qwen/Qwen3-0.6B",
            "dataset": "math_problems",
            "method": "Enhanced-COT",  # 表明这是增强版COT
            "optimization": "mixed_precision",  # 记录使用了混合精度训练
        }
    )

    # 配置Trainer
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer=tokenizer, padding=True),
        callbacks=[swanlab_callback],
    )

    print("开始训练...")
    trainer.train()

    print("训练完成，保存最终模型...")
    trainer.save_model("./output/Qwen3_COT/final")

    print("生成完成")
    swanlab.finish()
