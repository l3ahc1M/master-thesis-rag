import os
import json
import yaml
import pandas as pd

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
cfg_path = os.path.join(root_dir, 'config.yaml')
with open(cfg_path, 'r') as f:
    cfg = yaml.safe_load(f)


def process_test_case(file_path):
    # Open and read the JSON file
    with open(file_path, 'r') as file:
        data = json.load(file)
    
    # Extract required data
    summary = {
        'file': file_path,
        'framework': data.get('framework'),
        'model': data.get('model'),
        'llm_provider': data.get('llm_provider'),
        'knowledge_basis': data.get('knowledge_basis'),
        'exact_match': data.get('exact_match'),
    }
    
    # Initialize the component matching data with default values as False
    component_matches = {
        "method": False,
        "endpoint": False,
        "body": False,
        "select": False,
        "from": False,
        "join": False,
        "on": False,
        "where": False,
        "group_by": False,
        "having": False,
        "order_by": False,
        "limit": False,
        "[END]": False,
    }
    
    for result in data.get('component_matching_results', []):
        component = result.get('component')
        has_component = result.get('has_component')
        match = result.get('match')
        
        if has_component:
            component_matches[component] = match
    
    summary.update(component_matches)
    
    return summary

def generate_summary_dataframe():

    relevant_folders = cfg.get('evaluation', {}).get('folders_for_summary_generation')
    all_summaries = []
    for folder in relevant_folders:
        print("Folder: ", folder)
        directory_path = os.path.join(root_dir, 'evaluation_results', folder)

        # List to store all summaries

        
        # Iterate through all files in the folder
        for subdir, _, files in os.walk(directory_path):
            for file in files:
                if file.endswith('.json'):  # Only process JSON files
                    file_path = os.path.join(subdir, file)
                    summary = process_test_case(file_path)
                    all_summaries.append(summary)
            print("All summaries: ", all_summaries)
    # Convert the summaries into a DataFrame for easy analysis
    df = pd.DataFrame(all_summaries) #type: ignore
    
    return df




