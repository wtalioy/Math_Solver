import json
from datasets import Dataset

SYSTEM_PROMPT = """
请使用中文按以下格式，一步步思考，回答小学数学1-6年级的校内题目，答案不要带单位:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""

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
                    "answer": answer
                })
    
    dataset = Dataset.from_list(processed_data)
    return dataset