import os
import yaml
from xai_sdk import Client
from xai_sdk.chat import user, system
from dotenv import load_dotenv


load_dotenv()
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
cfg_path = os.path.join(root_dir, 'config.yaml')
with open(cfg_path, 'r') as f:
    cfg = yaml.safe_load(f)


def process_with_xai(user_prompt, system_prompt):


    client = Client(
    api_key=os.getenv("XAI_API_KEY"),
    timeout=3600,  # Override default timeout with longer timeout for reasoning models
    )

    chat = client.chat.create(
        model=cfg.get('llm', {}).get('xai_model'), 
        temperature=0
        )
    chat.append(system(system_prompt))
    chat.append(user(user_prompt))

    response = chat.sample()
    return response.content


#print(process_with_xai("how do I update a pip module via the cmd line?", "Help me"))