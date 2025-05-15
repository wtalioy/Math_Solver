import json
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from config import SYSTEM_PROMPT
from math_verify import ExprExtractionConfig, parse, verify
import re
from loguru import logger

model_id_or_path = "models/Qwen3-0.6B/"
val_json_path = "data/val.json"
checkpoint_path = "./outputs/Qwen3-0.6B-GRPO-latex/"  # Path to the checkpoint if loRA

def validate_answer(content, sol):
    gold_parsed = parse(
        sol,
        extraction_mode="first_match",
    )
    if len(gold_parsed) != 0:
        match = re.search(r"<answer>(.*?)</answer>", content, re.DOTALL)
        if match:
            answer_text = match.group(1).strip()
        else:
            answer_text = content
        answer_parsed = parse(
            answer_text,
            extraction_config=[ExprExtractionConfig()],
            extraction_mode="first_match",
        )
        try:
            logger.info(f"Answer parsed: {answer_parsed}")
            logger.info(f"Ground truth parsed: {gold_parsed}")
            is_correct = verify(gold_parsed, answer_parsed)
        except Exception as e:
            logger.warning(f"verify failed: {e}, answer: {answer_parsed}, ground_truth: {gold_parsed}")
    else:
        # If the gold solution is not parseable, we assign `None` to skip this example
        is_correct = False
        logger.warning(f"Failed to parse: {sol}")

    return is_correct

def predict(messages, model, tokenizer):
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        model_inputs.input_ids,
        max_new_tokens=2048,
    )

    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 

    response = tokenizer.decode(output_ids, skip_special_tokens=True).strip("\n")
     
    return response


with open(val_json_path, 'r', encoding='utf-8') as file:
    val_data = json.load(file)

tokenizer = AutoTokenizer.from_pretrained(model_id_or_path, use_fast=False, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_id_or_path, device_map="auto", torch_dtype=torch.bfloat16)
model = PeftModel.from_pretrained(model, model_id=checkpoint_path)  # Load the LoRA model (comment this line if not using LoRA)

correct_count = 0
total_count = len(val_data)
for idx, row in enumerate(tqdm(val_data)):
    input_value = row['question']
    id = row['id']
    
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"{input_value}"}
    ]
    logger.info(f"ID: {id}")
    logger.info(f"Question: {input_value}")

    response = predict(messages, model, tokenizer)
    logger.info(f"Response: {response}")
    
    if validate_answer(response, row['answer']):
        correct_count += 1
        logger.info("Correct")
    else:
        logger.info("Incorrect")
        

print(f"Accuracy: {correct_count / total_count * 100:.2f}%")
