import yaml
import os   
from llm_integration.openAI import process_with_openAI
from llm_integration.x_ai import process_with_xai

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
cfg_path = os.path.join(root_dir, 'config.yaml')
with open(cfg_path, 'r') as f:
    cfg = yaml.safe_load(f)

def send_to_llm(user_prompt, system_prompt):
    provider = cfg.get('llm', {}).get('provider')
    if provider == "xai":
        return process_with_xai(user_prompt, system_prompt)
    elif provider == "openAI":
        return process_with_openAI(user_prompt, system_prompt)
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")
