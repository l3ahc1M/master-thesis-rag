from chroma_handler.retrieval import retrieve_n_closest_entries
from llm_integration.handler import send_to_llm


basic_system_prompt = """
You are a coding assistant that is supposed to provide either SQL queries for data retrieval or API calls for action triggering.
Only answer with the SQL query or the API call, nothing else.
The system has the following specifications:
"""

def get_rag_model_response(input_text):
    retrieve_n_closest_entries(input_text)
    final_system_prompt = basic_system_prompt + "\n".join([f"- {entry['document']}" for entry in retrieve_n_closest_entries(input_text)])

    llm_response = send_to_llm(input_text, final_system_prompt)

    return llm_response
