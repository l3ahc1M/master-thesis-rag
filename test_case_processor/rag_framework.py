from chroma_handler.retrieval import retrieve_n_closest_entries
from llm_integration.handler import send_to_llm


basic_system_prompt = """
You are a coding assistant that is supposed to provide either SQL queries for data retrieval or API calls for action triggering.
SQL answers must only contain the SQL query, no additional text or explanation.
API answers must be provided in the following structure - only provide the body, if it is required, otherwise leave it out:
{
    "method": "PATCH",
    "endpoint": "/BankAccountContractCancelRequests/01234567-89ab-cdef-0123-456789abcdef",
    "body": {
      "ReasonCode": "109",
      "ReasonName": "Account Closure"
    }

Always consider May 1st, 2025 as the current date for all your answers.
The system has the following specifications:
"""

def get_rag_model_response(input_text):
    retrieve_n_closest_entries(input_text)
    final_system_prompt = basic_system_prompt + "\n".join([f"- {entry['document']}" for entry in retrieve_n_closest_entries(input_text)])

    llm_response = send_to_llm(input_text, final_system_prompt)

    return llm_response
