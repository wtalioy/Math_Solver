import json
from datasets import Dataset
from config import SYSTEM_PROMPT

def load_custom_dataset(file_path):
    """
    Load a custom dataset from a JSON file.

    Args:
        file_path (str): Path to the JSON file.

    Returns:
        list: List of dictionaries representing the dataset.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    processed_data = []
    for item in data:
        if isinstance(item, dict):
            question = item.get("question", "")
            answer = item.get("answer", "")
            
            if question and answer:
                prompt = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": question},
                ]
                processed_data.append({
                    "prompt": prompt,
                    "solution": answer
                })
    
    dataset = Dataset.from_list(processed_data)
    return dataset