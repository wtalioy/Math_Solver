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
batch_size = 8

tokenizer = AutoTokenizer.from_pretrained(model_id_or_path, use_fast=False, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_id_or_path, device_map="auto", torch_dtype=torch.bfloat16)
model = PeftModel.from_pretrained(model, model_id=checkpoint_path)  # Load the LoRA model (comment this line if not using LoRA)


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
        temperature=0,
    )

    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 

    response = tokenizer.decode(output_ids, skip_special_tokens=True).strip("\n")
    return response


def predict_batch(messages_list, model, tokenizer, batch_size=8, max_new_tokens=2048):
    responses = []
    for i in range(0, len(messages_list), batch_size):
        batch = messages_list[i:i+batch_size]
        texts = [tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        ) for messages in batch]
        model_inputs = tokenizer(texts, return_tensors="pt", padding=True).to(model.device)
        generated_ids = model.generate(
            model_inputs.input_ids,
            attention_mask=model_inputs.attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=0,
        )
        for j, input_ids in enumerate(model_inputs.input_ids):
            output_ids = generated_ids[j][len(input_ids):].tolist()
            response = tokenizer.decode(output_ids, skip_special_tokens=True).strip("\n")
            responses.append(response)
    return responses


with open(val_json_path, 'r', encoding='utf-8') as file:
    val_data = json.load(file)

correct_count = 0
total_count = len(val_data)

all_messages = [
    [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": row['question']}
    ]
    for row in val_data
]

results = []
for i in tqdm(range(0, len(all_messages), batch_size), desc="Batch Inference Progress"):
    batch = all_messages[i:i+batch_size]
    batch_rows = val_data[i:i+batch_size]
    batch_responses = predict_batch(batch, model, tokenizer, batch_size=len(batch), max_new_tokens=2048)
    for row, response in zip(batch_rows, batch_responses):
        id = row['id']
        input_value = row['question']
        logger.info(f"ID: {id}")
        logger.info(f"Question: {input_value}")
        logger.info(f"Response: {response}")
        is_correct = validate_answer(response, row['answer'])
        if is_correct:
            correct_count += 1
            logger.info("Correct")
        else:
            logger.info("Incorrect")
        results.append({
            "id": id,
            "question": input_value,
            "answer": row['answer'],
            "model_response": response,
            "is_correct": is_correct
        })

print(f"Accuracy: {correct_count / total_count * 100:.2f}%")

with open("val_with_model_response.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
