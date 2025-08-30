import yaml


########################################################
##### Security check of config.yaml values ###########
########################################################
with open('config.yaml', 'r') as f:
    cfg = yaml.safe_load(f)

    # Check specific values in cfg and prompt user if they DO match expected values
    
    if cfg.get('process_orchestration', {}).get('update_chroma_db', True):
        expected_values = {
            'system_documentation': {'source_folder': 'system_documentation'},
            'test_cases': {'source_folder': 'test_cases'}
        }
    else:
        expected_values = {
            'test_cases': {'source_folder': 'test_cases'}
        }

    for section, keys in expected_values.items():
        warnings = []
        for key, expected in keys.items():
            actual = cfg.get(section, {}).get(key)
            if actual == expected:
                warnings.append(f"{section} is '{actual}', do you really want to process all {section}'?")

        if warnings:
            print("WARNING:")
            for w in warnings:
                print(w)
            security_check = input("Continue anyway?\n(y to continue):   ")
            if security_check.lower() != 'y':
                print("Aborted by user.")
                exit(1)

########################################################
##### End of security check ############################
########################################################


print("Started")
if cfg.get('process_orchestration', {}).get('update_chroma_db', True):
    from system_documentation_processor import api_documentation_processor, business_object_description_processor, db_documentation_processor

    chroma_check = input("Did you manually delete the any existing ChromaDB folder (chroma_db)? If not, please do so now. Otherwise, the new data will be appended to the existing data.\nType 'y' to continue: ")
    if chroma_check.lower() != 'y':
                print("Aborted by user.")
                exit(1)
    print("Updating ChromaDB...")
    api_documentation_processor.preprocess_api_documentation()
    db_documentation_processor.preprocess_db_documentation()
    business_object_description_processor.preprocess_business_object_description()
    print("ChromaDB update finished.")


if cfg.get('process_orchestration', {}).get('run_test_cases', True):
    from test_case_processor.handler import process_test_cases
    process_test_cases()

if cfg.get('process_orchestration', {}).get('run_evaluation', True): 
    print("Hello")

print("Finished")


