import os
import yaml
import hashlib
from copy import deepcopy
from chroma_handler.ingestion import store_text_embedding


root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
cfg_path = os.path.join(root_dir, 'config.yaml')
with open(cfg_path, 'r') as f:
    cfg = yaml.safe_load(f)
    def get_business_object_description_files():
        business_object_descriptions = []
        source_folder = cfg.get('system_documentation', {}).get('source_folder')
        for root, dirs, files in os.walk(source_folder):
            for file in files:
                if file.endswith(".txt"):
                    file_path = os.path.join(root, file)
                    # Get the subfolder name as business_object_name
                    rel_path = os.path.relpath(root, source_folder)
                    business_object_name = os.path.basename(rel_path) if rel_path != '.' else ''
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            business_object_description = f.read()
                        business_object_descriptions.append({
                            'business_object_name': business_object_name,
                            'file_name': file,
                            'business_object_description': business_object_description
                        })
                    except Exception as e:
                        # Optionally log or handle error
                        pass
        return business_object_descriptions




def preprocess_business_object_description():

    business_object_descriptions = get_business_object_description_files()
    for business_object_description in business_object_descriptions:
        business_object = business_object_description['business_object_name']
        file_name = business_object_description['file_name']
        business_object_description_content = business_object_description['business_object_description']

 
        source_type = "TXT"

        # Create SHA256 hash using embedding to be used as unique ID
        hash_value = hashlib.sha256(str(business_object_description_content).encode("utf-8")).hexdigest()

        store_text_embedding(
            text=business_object_description_content,
            file_name=file_name,
            business_object=business_object,
            source_type=source_type,
            id=hash_value,
        )
    