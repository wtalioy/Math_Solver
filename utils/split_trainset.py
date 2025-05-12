'''
Script to split Math Solver JSON data into training and validation sets.
The original format of the items is preserved.
'''
import json
import os

def write_json_subset(data_subset, output_file_path, set_name):
    '''
    Writes a subset of data to a JSON file.

    Args:
        data_subset (list): The subset of data items to write.
        output_file_path (str): Path to the output JSON file.
        set_name (str): Name of the dataset (e.g., "training", "validation") for logging.
    '''
    # Ensure the output directory exists
    output_dir = os.path.dirname(output_file_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    try:
        with open(output_file_path, 'w', encoding='utf-8') as outfile:
            json.dump(data_subset, outfile, ensure_ascii=False, indent=2) # indent for readability
        
        print(f"Successfully wrote {len(data_subset)} items to {set_name} set: {output_file_path}")

    except IOError as e:
        print(f"Error writing file {output_file_path}: {e}")


def split_json_data(input_file_path, train_output_path, val_output_path, train_split_size=10000):
    '''
    Loads data from a JSON file, splits it into training and validation sets,
    and writes them to separate JSON files, preserving the original item format.

    Args:
        input_file_path (str): Path to the input JSON file (list of items).
        train_output_path (str): Path for the training set output JSON file.
        val_output_path (str): Path for the validation set output JSON file.
        train_split_size (int): Number of items to include in the training set.
    '''
    try:
        with open(input_file_path, 'r', encoding='utf-8') as infile:
            try:
                all_data = json.load(infile)
            except json.JSONDecodeError as e:
                print(f"Error: Could not decode JSON from {input_file_path}. Details: {e}")
                return

            if not isinstance(all_data, list):
                print(f"Error: Input JSON content from {input_file_path} is not a list of items.")
                return

            if len(all_data) < train_split_size:
                print(f"Warning: Total items ({len(all_data)}) is less than train_split_size ({train_split_size}).")
                print(f"All items will be used for training, and the validation set will be empty.")
                training_data = all_data
                validation_data = []
            else:
                training_data = all_data[:train_split_size]
                validation_data = all_data[train_split_size:]
            
            print(f"Total items loaded: {len(all_data)}")
            print(f"Training set size: {len(training_data)}")
            print(f"Validation set size: {len(validation_data)}")

            write_json_subset(training_data, train_output_path, "training")
            if validation_data:
                write_json_subset(validation_data, val_output_path, "validation")
            else:
                print("Validation set is empty, skipping write for validation set.")

    except FileNotFoundError:
        print(f"Error: Input file not found at {input_file_path}")
    except IOError as e:
        print(f"Error reading input file {input_file_path}: {e}")

if __name__ == '__main__':
    base_dir = "data"
    original_train_file = os.path.join(base_dir, "train.json")
    
    # New output filenames for split data in original format
    output_train_file = os.path.join(base_dir, "train_10k.json")
    output_val_file = os.path.join(base_dir, "val.json")

    split_json_data(original_train_file, output_train_file, output_val_file, train_split_size=10000)
