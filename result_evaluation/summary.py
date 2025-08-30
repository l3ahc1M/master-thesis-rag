import os
import json
import pandas as pd

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

def generate_summary_dataframe(directory_path):
    # List to store all summaries
    all_summaries = []
    
    # Iterate through all files in the folder
    for subdir, _, files in os.walk(directory_path):
        for file in files:
            if file.endswith('.json'):  # Only process JSON files
                file_path = os.path.join(subdir, file)
                summary = process_test_case(file_path)
                all_summaries.append(summary)
    
    # Convert the summaries into a DataFrame for easy analysis
    df = pd.DataFrame(all_summaries)
    
    return df




