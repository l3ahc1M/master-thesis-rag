# table_chunking.py
import os
import yaml
import hashlib
from copy import deepcopy
from chroma_handler.ingestion import store_text_embedding


root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
cfg_path = os.path.join(root_dir, 'config.yaml')
with open(cfg_path, 'r') as f:
    cfg = yaml.safe_load(f)
    def get_db_description_files():
        db_descriptions = []
        source_folder = cfg.get('system_documentation', {}).get('source_folder')
        for root, dirs, files in os.walk(source_folder):
            for file in files:
                if file.endswith(".json") and file.startswith('DB_'):
                    file_path = os.path.join(root, file)
                    # Get the subfolder name as business_object_name
                    rel_path = os.path.relpath(root, source_folder)
                    business_object_name = os.path.basename(rel_path) if rel_path != '.' else ''
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f: # type: ignore
                            db_description = f.read()
                        db_descriptions.append({
                            'business_object_name': business_object_name,
                            'file_name': file,
                            'db_description': db_description
                        })
                    except Exception as e:
                        # Optionally log or handle error
                        pass
        return db_descriptions




def preprocess_db_documentation():

    db_descriptions = get_db_description_files()
    for db_description in db_descriptions:
        business_object = db_description['business_object_name']
        file_name = db_description['file_name']
        db_description_content = db_description['db_description']

 
        source_type = "DB"

        # Create SHA256 hash using embedding to be used as unique ID
        hash_value = hashlib.sha256(str(db_description_content).encode("utf-8")).hexdigest()

        store_text_embedding(
            text=db_description_content,
            file_name=file_name,
            business_object=business_object,
            source_type=source_type,
            id=hash_value,
        )
    