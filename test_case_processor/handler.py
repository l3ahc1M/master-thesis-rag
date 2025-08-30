import os
import json
import yaml
import datetime


root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
cfg_path = os.path.join(root_dir, 'config.yaml')
with open(cfg_path, 'r') as f:
    cfg = yaml.safe_load(f)

def harmonize_output(output):
    if isinstance(output, str):
        # Remove line breaks and extra spaces
        return ' '.join(output.split())
    return output

def load_test_cases():
    test_cases = []
    for category in ["API", "SQL"]:
        test_cases_dir = os.path.join(root_dir, cfg.get('test_cases', {}).get('source_folder') , category)
        if not os.path.isdir(test_cases_dir):
            continue
        for subfolder in os.listdir(test_cases_dir):
            subfolder_path = os.path.join(test_cases_dir, subfolder)
            if os.path.isdir(subfolder_path):
                for filename in os.listdir(subfolder_path):
                    if filename.endswith(".json"):
                        file_path = os.path.join(subfolder_path, filename)
                        with open(file_path, "r", encoding="utf-8") as f:
                            data = json.load(f)
                            test_cases.append({
                                "type": category,
                                "business_object_name": subfolder,
                                "input": data.get("input"),
                                "output": harmonize_output(data.get("output")),
                                "file": file_path
                            })
    print("Loaded", len(test_cases), "test cases.")
    return test_cases

def send_test_case_to_rag_model(test_case):
    input_data = test_case["input"]

    if cfg.get('process_orchestration', {}).get('rag_framework') == 'RAG':
        from test_case_processor import rag_framework 
        response = rag_framework.get_rag_model_response(input_data)
    elif cfg.get('process_orchestration', {}).get('rag_framework') == 'SelfRAG':
        from test_case_processor import self_rag_framework
        response = self_rag_framework.get_self_rag_model_response(input_data)
    elif cfg.get('process_orchestration', {}).get('rag_framework') == 'CoRAG':
        from test_case_processor import corag_framework 
        response = corag_framework.get_corag_model_response(input_data)
    else:
        raise ValueError("Invalid RAG framework specified in config.yaml")
    
    llm_provider = cfg.get('llm', {}).get('provider')
    test_case['framework'] = cfg.get('process_orchestration', {}).get('rag_framework')
    test_case['llm_provider'] = llm_provider
    test_case['model'] = cfg.get('llm', {} ).get(f'{llm_provider}_model') 
    test_case["test_output"] = response
     
    return test_case

def process_test_cases():

    raw_test_cases = load_test_cases()
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = f"{timestamp}_{cfg.get('process_orchestration', {}).get('rag_framework')}_{cfg.get('llm', {}).get('provider')}_{cfg.get('process_orchestration', {}).get('knowledge_basis')}"
    results_dir = os.path.join("test_output", folder_name)
    os.makedirs(results_dir, exist_ok=True)
    
    #store a copy of the config file in the results folder
    with open(os.path.join(results_dir, "config.yaml"), "w", encoding="utf-8") as f:
        yaml.dump(cfg, f)

    for test_case in raw_test_cases:
        processed_test_case = send_test_case_to_rag_model(test_case)
        original_filename = os.path.basename(test_case["file"])
        result_file_path = os.path.join(results_dir, original_filename)
        with open(result_file_path, "w", encoding="utf-8") as f:
            json.dump(processed_test_case, f, ensure_ascii=False, indent=2)
    print(f"Results saved to {results_dir}")