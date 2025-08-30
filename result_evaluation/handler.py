import yaml
import os
import json


root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
cfg_path = os.path.join(root_dir, 'config.yaml')
with open(cfg_path, 'r') as f:
    cfg = yaml.safe_load(f)


def get_processed_test_cases_array():
    test_case_dir = os.path.join(root_dir, 'results',  cfg.get('evaluation', {}).get('test_case_folder'))

    processed_test_cases_array = []

    for file in test_case_dir:
        file_path = os.path.join(test_case_dir, file)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                json_content = json.load(f)
                processed_test_cases_array.append(json_content)

        except Exception as e:
            # Optionally log or handle error
            pass
    return processed_test_cases_array



def evaluate_test_cases():
    return None