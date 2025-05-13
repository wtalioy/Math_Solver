import json
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from config import SYSTEM_PROMPT
from math_verify import LatexExtractionConfig, parse, verify
from latex2sympy2_extended import NormalizationConfig
from loguru import logger

val_json_path = "data/val.json"
checkpoint_path = "./outputs/checkpoint-200/"

def validate_answer(content, sol):
    gold_parsed = parse(
        sol,
        extraction_mode="first_match",
    )
    if len(gold_parsed) != 0:
        answer_parsed = parse(
            content,
            extraction_config=[
                LatexExtractionConfig(
                    normalization_config=NormalizationConfig(
                        nits=False,
                        malformed_operators=False,
                        basic_latex=True,
                        equations=True,
                        boxed="all",
                        units=True,
                    ),
                    boxed_match_priority=0,
                    try_extract_without_anchor=False,
                )
            ],
            extraction_mode="first_match",
        )
        try:
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
        max_new_tokens=1024,
    )

    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 

    response = tokenizer.decode(output_ids, skip_special_tokens=True).strip("\n")
     
    return response


with open(val_json_path, 'r', encoding='utf-8') as file:
    val_data = json.load(file)

tokenizer = AutoTokenizer.from_pretrained("models/Qwen3-0.6B/", use_fast=False, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("models/Qwen3-0.6B/", device_map="auto", torch_dtype=torch.bfloat16)
model = PeftModel.from_pretrained(model, model_id=checkpoint_path)

correct_count = 0
total_count = len(val_data)
for idx, row in enumerate(tqdm(val_data)):
    input_value = row['question']
    id = row['id']
    
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"{input_value}"}
    ]
    response = predict(messages, model, tokenizer)
    logger.info(f"ID: {id}")
    logger.info(f"Question: {input_value}")
    logger.info(f"Response: {response}")
    
    if validate_answer(response, row['answer']):
        correct_count += 1

print(f"Accuracy: {correct_count / total_count * 100:.2f}%")