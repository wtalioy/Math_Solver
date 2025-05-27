import json
import torch
from modelscope import snapshot_download
from swanlab.integration.huggingface import SwanLabCallback
from peft import LoraConfig, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOTrainer, DPOConfig
import swanlab
from datasets import Dataset

model_dir = snapshot_download("Qwen/Qwen3-0.6B", cache_dir="./", revision="master")
print(f"模型下载完成，存储在: {model_dir}")

# Transformers加载模型权重
tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="cuda", torch_dtype=torch.bfloat16, trust_remote_code=True)
model.enable_input_require_grads()  # 开启梯度检查点时，要执行该方法

# 直接加载已经处理好的DPO数据
dpo_json_path = "dpo_train_data.json"

# 加载数据集
with open(dpo_json_path, 'r', encoding='utf-8') as file:
    dpo_data = json.load(file)

print(f"DPO训练数据集大小: {len(dpo_data)}")

# 创建DPO格式的Dataset对象 - 直接使用已格式化好的数据
train_dataset = Dataset.from_dict({
    "prompt": [item.get("prompt", "") for item in dpo_data],
    "chosen": [item.get("chosen", "") for item in dpo_data],
    "rejected": [item.get("rejected", "") for item in dpo_data]
})

print(f"训练数据集大小: {len(train_dataset)}")

# 设置LoRA配置
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    inference_mode=False,  # 训练模式
    r=8,  # Lora 秩
    lora_alpha=32,  # Lora alaph
    lora_dropout=0.1,  # Dropout 比例
)

# 创建DPO训练配置
dpo_config = DPOConfig(
    output_dir="./output/Qwen_DPO",
    per_device_train_batch_size=4, # 2
    gradient_accumulation_steps=4,
    learning_rate=4e-6,  # 5e-6
    num_train_epochs=3,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    logging_steps=10,
    save_steps=1500,
    eval_steps=500,
    report_to="none",  # 关闭默认的报告工具，使用SwanLab
)

# 创建SwanLab回调
swanlab_callback = SwanLabCallback(
    project="Qwen2.5-0.5B-DPO",
    experiment_name="Qwen-DPO-Math-Solver",
    config={
        "model": "Qwen/Qwen2.5-0.5B-Instruct",
        "dataset": "math_solver_dpo",
        "training_type": "DPO",
        "beta": 0.1,
    }
)

# 创建DPO训练器
trainer = DPOTrainer(
    model=model,  # 明确指定参数名称
    ref_model=None,  # ref_model设为None，使用peft_config时不需要
    args=dpo_config,
    train_dataset=train_dataset,
    eval_dataset=None,  # 不使用验证集
    processing_class=tokenizer,  # 使用processing_class而不是tokenizer
    peft_config=peft_config,
    callbacks=[swanlab_callback]  # 添加SwanLab回调
)
'''
# 检查CUDA可用性
print(f"PyTorch 是否可以使用 CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"当前使用的GPU设备: {torch.cuda.get_device_name(0)}")
    print(f"GPU数量: {torch.cuda.device_count()}")
    print(f"当前GPU内存使用情况: {torch.cuda.memory_allocated(0)/1024**2:.2f} MB")
'''
# 开始训练
trainer.train()

# 保存模型
trainer.save_model("./output/Qwen_DPO_final")

swanlab.finish()
