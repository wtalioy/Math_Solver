import json
import os

def process_data_for_dpo(input_file, output_file):
    """
    Process the input JSON file to create a dataset suitable for DPOTrainer.
    
    Args:
        input_file (str): Path to the input JSON file with cot_pos and cot_neg fields.
        output_file (str): Path to save the processed DPO-formatted JSON file.
    """
    # Read the input file
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Process the data
    dpo_data = []
    for item in data:
        # Skip items with missing fields
        if not all(key in item for key in ["question", "cot_pos", "cot_neg"]):
            continue        # Create entry in DPO format with Qwen format
        formatted_prompt = f"<|im_start|>user\n{item['question']}<|im_end|>\n<|im_start|>assistant\n"
        dpo_entry = {
            "prompt": formatted_prompt,
            "chosen": item["cot_pos"],
            "rejected": item["cot_neg"]
        }
        dpo_data.append(dpo_entry)
    
    # Save the processed data
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(dpo_data, f, ensure_ascii=False, indent=2)
    
    return len(dpo_data)

if __name__ == "__main__":
    input_file = "train_10k_with_neg_cot.json"
    output_file = "dpo_train_data.json"
    
    # Process the data
    count = process_data_for_dpo(input_file, output_file)
    print(f"Successfully processed {count} examples for DPO training.")
    print(f"Output saved to {output_file}")
