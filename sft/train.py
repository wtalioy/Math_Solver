import os
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForSeq2Seq
import json
import torch
print("CUDA available:", torch.cuda.is_available())
print("CUDA device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("Current device:", torch.cuda.current_device())
    print("Device name:", torch.cuda.get_device_name(0))
else:
    print("未检测到可用的CUDA设备！")


# 配置
MODEL_PATH = "Qwen/Qwen2.5-0.5B-Instruct"
TRAIN_PATH = "../data/train_10k_ds.json"
OUTPUT_DIR = "../models/qwen2.5-0.5b-finetuned/"

# 加载tokenizer和模型
print("加载模型和分词器...")
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH, use_fast=False, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, device_map="cuda", torch_dtype=torch.float16)
model.enable_input_require_grads()

# 数据预处理
MAX_LENGTH = 384


def process_func(example):
    instruction = tokenizer(
        f"<|im_start|>system\n{example['instruction']}<|im_end|>\n<|im_start|>user\n{example['question']}<|im_end|>\n<|im_start|>assistant\n",
        add_special_tokens=False,
    )
    response = tokenizer(f"{example['answer']}", add_special_tokens=False)
    input_ids = instruction["input_ids"] + \
        response["input_ids"] + [tokenizer.pad_token_id]
    attention_mask = instruction["attention_mask"] + \
        response["attention_mask"] + [1]
    labels = [-100] * len(instruction["input_ids"]) + \
        response["input_ids"] + [tokenizer.pad_token_id]
    if len(input_ids) > MAX_LENGTH:
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


print("加载训练数据...")
with open(TRAIN_PATH, 'r', encoding='utf-8') as file:
    train_data = json.load(file)
print(f"原始训练数据量: {len(train_data)}")
print(f"用于本次调试的训练数据量: {len(train_data)}")
print("开始预处理数据...")
train_dataset = [process_func(d) for d in train_data]
print("数据预处理完成！")

# LoRA配置
config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "k_proj", "v_proj",
                    "o_proj", "gate_proj", "up_proj", "down_proj"],
    inference_mode=False,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
)
model = get_peft_model(model, config)

# 训练参数
args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    logging_steps=10,
    num_train_epochs=1,
    save_steps=500,
    learning_rate=1e-4,
    save_on_each_node=True,
    gradient_checkpointing=True,
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
)

print("开始训练...")
trainer.train()

model.save_pretrained(OUTPUT_DIR)
print(f"模型已保存到 {OUTPUT_DIR}")
