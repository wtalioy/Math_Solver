'''
Script to clean the format of 'question' fields in JSON data.
Specifically, it converts questions that are lists of strings into a single string.
'''
import json
import os

def clean_question_field_in_file(file_path):
    '''
    Reads a JSON file, cleans the 'question' field, and writes it back.

    Args:
        file_path (str): Path to the JSON file to clean.
    '''
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from {file_path}: {e}")
        return
    except IOError as e:
        print(f"Error reading file {file_path}: {e}")
        return

    if not isinstance(data, list):
        print(f"Error: Expected a list of items in {file_path}, but got {type(data)}.")
        return

    modified_count = 0
    processed_count = 0

    for item in data:
        processed_count += 1
        if isinstance(item, dict) and "question" in item and isinstance(item["question"], list):
            # Join the list of strings into a single string.
            # This will handle cases like [""abc", "def", "ghi""] correctly.
            item["question"] = "".join(item["question"])
            item["question"] = item["question"][1:-1]  # Remove the first two and last two characters.
            modified_count += 1
    
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"Cleaned {file_path}:")
        print(f"  Processed {processed_count} items.")
        print(f"  Modified {modified_count} 'question' fields.")
    except IOError as e:
        print(f"Error writing cleaned data to {file_path}: {e}")

if __name__ == '__main__':
    base_dir = "data"
    train_file = os.path.join(base_dir, "train_10k.json")
    validation_file = os.path.join(base_dir, "val.json")

    print("Starting cleaning process...")
    clean_question_field_in_file(train_file)
    clean_question_field_in_file(validation_file)
    print("Cleaning process finished.")
