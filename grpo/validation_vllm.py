from openai import OpenAI
from config import SYSTEM_PROMPT
import json
import re
from loguru import logger
from tqdm.contrib import tzip
from math_verify import ExprExtractionConfig, parse, verify

val_json_path = "data/val.json"

client = OpenAI(
    api_key="EMPTY",
    base_url="http://localhost:8000/v1",
)

models = client.models.list()
model_name = models.data[0].id

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
for row, message in tzip(val_data, all_messages):
    id = row['id']
    input_value = row['question']
    logger.info(f"ID: {id}")
    logger.info(f"Question: {input_value}")
    response = client.chat.completions.create(
        model=model_name,
        max_tokens=2048,
        messages=message,
    ).choices[0].message.content
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