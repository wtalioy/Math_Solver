from openai import OpenAI
from config import SYSTEM_PROMPT
import json
import re
import requests
import argparse
from loguru import logger
from tqdm.contrib import tzip
from math_verify import parse as mp, verify

def validate_answer(content, sol):
    gold_parsed = mp(
        sol,
        extraction_mode="first_match",
    )
    if len(gold_parsed) != 0:
        match = re.search(r"<answer>(.*?)</answer>", content, re.DOTALL)
        if match:
            answer_text = match.group(1).strip()
        else:
            answer_text = content
        answer_parsed = mp(
            answer_text,
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

def main(args):
    client = OpenAI(
        api_key="EMPTY",
        base_url="http://localhost:8000/v1",
    )
    models = client.models.list().data
    if args.lora is not None:
        if len(models) > 1:
            requests.post("http://localhost:8000/v1/unload_lora_adapter", json={
                "lora_name": "grpo"
            })
            logger.info("Successfully unload previous lora adpater.")
        else:
            logger.info("No previous lora adapter found.")
        requests.post("http://localhost:8000/v1/load_lora_adapter", json={
            "lora_name": "grpo",
            "lora_path": args.lora
        })
        logger.info(f"Successfully load current lora adapter at path {args.lora}")
        logger.info("Validating lora model ...")
        model_name = "grpo"
        output_name = args.lora.split('/')[-1]
    else:
        logger.info("Validating base model ...")
        model_name = models[0].id
        output_name = model_name


    with open(args.set, 'r', encoding='utf-8') as file:
        val_data = json.load(file)
        
    correct_count = 0
    total_count = len(val_data)

    all_messages = [
        [
            {"role": "system", "content": SYSTEM_PROMPT[args.pr]},
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
            temperature=0,
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

    with open(f"{output_name}_response.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    parse = argparse.ArgumentParser(description='Validation Script')
    parse.add_argument('--lora', type=str, default=None, help='path to the lora adapter')
    parse.add_argument('--set', type=str, default="data/val_300.json", help='path to the validation set')
    parse.add_argument('--pr', type=int, default=0, help="prompt id")
    args = parse.parse_args()
    main(args)