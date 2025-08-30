import yaml
import os
import json
import hashlib

from result_evaluation.API import evaluate_api_test_case
from result_evaluation.SQL import evaluate_sql_test_case

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
cfg_path = os.path.join(root_dir, 'config.yaml')
with open(cfg_path, 'r') as f:
    cfg = yaml.safe_load(f)


def get_processed_test_cases_array():
    test_case_dir = os.path.join(
        root_dir, 'test_output',
        cfg.get('evaluation', {}).get('test_case_folder', '')
    )
    print("Test case dir:", test_case_dir)

    if not os.path.isdir(test_case_dir):
        raise FileNotFoundError(f"Test case folder not found: {test_case_dir}")

    processed_test_cases_array = []

    for entry in os.listdir(test_case_dir):
        file_path = os.path.join(test_case_dir, entry)

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                json_content = json.load(f)
                processed_test_cases_array.append(json_content)
                print(f"Loaded: {entry}")
        except json.JSONDecodeError as e:
            print(f"Skipping {entry}: invalid JSON ({e})")
        except Exception as e:
            print(f"Skipping {entry}: {e}")

    print("Total test cases loaded:", len(processed_test_cases_array))
    return processed_test_cases_array


def save_evaluated_test_case(test_case):
    """
    Save the evaluated test case in the corresponding subfolder.
    """
    evaluation_results_dir = os.path.join(root_dir, 'evaluation_results', cfg.get('evaluation', {}).get('test_case_folder', ''))

    # Create the subfolder if it doesn't exist
    os.makedirs(evaluation_results_dir, exist_ok=True)
    hash_value = hashlib.sha256(str(test_case).encode("utf-8")).hexdigest()

    # Define the path for the evaluated test case JSON
    output_file_path = os.path.join(evaluation_results_dir, f"{hash_value}.json")

    # Save the evaluated test case as a JSON file
    with open(output_file_path, 'w', encoding='utf-8') as f:
        json.dump(test_case, f, ensure_ascii=False, indent=4)
    print(f"Saved evaluated test case: {output_file_path}")

def evaluate_test_cases():
    print("I am here")
    processed_test_cases_array = get_processed_test_cases_array()
    test_cases_incl_evaluation_result = []
    for test_case in processed_test_cases_array:
        if test_case['type'] == 'API':
            test_case = evaluate_api_test_case(test_case)
        elif test_case['type'] == 'SQL':
            print("Evaluating SQL test case...")
            test_case = evaluate_sql_test_case(test_case)
        else:
            raise ValueError(f"Unknown test case type: {test_case['type']}")
        
        test_cases_incl_evaluation_result.append(test_case)

        save_evaluated_test_case(test_case)


    