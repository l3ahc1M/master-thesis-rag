import os
import json
import logging
import time
from pathlib import Path
from typing import List, Any


import openai
from openai.types.chat import ChatCompletionMessageParam, ChatCompletionSystemMessageParam, ChatCompletionUserMessageParam
import chromadb
from dotenv import load_dotenv

# ────────────────── CONFIGURATION ──────────────────
load_dotenv()  # Load environment variables from .env

# ChromaDB persistence settings (matching ingest_docs_full.py)
PERSIST_DIR     = 'chromadb_data'
COLLECTION_NAME = 'system_documentation'
EMBEDDING_MODEL = 'text-embedding-3-small'

# RAG runner settings
OPENAI_API_KEY     = os.getenv('OPENAI_API_KEY')
GENERATION_MODEL  = os.getenv('OPENAI_GEN_MODEL', 'gpt-3.5-turbo')
RAG_TOP_K         = int(os.getenv('RAG_TOP_K', '5'))

# Directories for input test cases
BASE_DIR = Path('test_cases')
API_DIR  = BASE_DIR / 'API'
SQL_DIR  = BASE_DIR / 'SQL'

# Directories for output results
OUTPUT_BASE = Path('test_cases_results')
OUTPUT_API  = OUTPUT_BASE / 'API'
OUTPUT_SQL  = OUTPUT_BASE / 'SQL'
for d in (OUTPUT_API, OUTPUT_SQL):
    d.mkdir(parents=True, exist_ok=True)

# ────────────────── LOGGING ──────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)
logger = logging.getLogger(__name__)

# Initialize OpenAI API key
openai.api_key = OPENAI_API_KEY

def init_chroma():
    """
    Initialize ChromaDB PersistantClient and return the collection
    configured with the OpenAI embedding function.
    """
    logger.info('Initializing ChromaDB client at %s', PERSIST_DIR)
    client = chromadb.PersistentClient(path=PERSIST_DIR)
    collection = client.get_or_create_collection("openai_embeddings")
    
    logger.info('ChromaDB collection "%s" ready', COLLECTION_NAME)
    return collection

def get_embedding(text, model=EMBEDDING_MODEL): # type: ignore
    response = openai.embeddings.create(input=text, model=model) # type: ignore
    return response.data[0].embedding

def retrieve_context(collection: Any, query: str, top_k: int = RAG_TOP_K) -> str:
    logger.info('Retrieving top %d contexts for query: %s', top_k, query)
    query_embedding = get_embedding(query)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=2
    )
    docs = results['documents'][0]
    context = "\n---\n".join(docs)
    return context


def chat_completion(messages: List[ChatCompletionMessageParam], retries: int = 3, delay: int = 2):
    """
    Attempt to call OpenAI API with retries on failure.
    """
    for attempt in range(retries):
        try:
            return openai.chat.completions.create(
                model=GENERATION_MODEL,
                messages=messages,
                temperature=0
            )
        except Exception as e:
            logger.error("API call failed (attempt %d/%d): %s", attempt + 1, retries, e)
            if attempt == retries - 1:
                raise
            time.sleep(delay)

def process_test_file(collection: Any, in_path: Path, out_path: Path):
    """
    Load a test case JSON, enrich with RAG context, call the OpenAI API,
    and write the enriched test case to output.
    """
    logger.info('Processing test file %s', in_path)
    try:
        with open(in_path, 'r', encoding='utf-8') as f:
            test_case = json.load(f)
    except Exception as e:
        logger.error('Failed to load test case %s: %s', in_path.name, e)
        return

    user_input = test_case.get('input', '').strip()
    if not user_input:
        logger.warning('Skipping %s: missing input', in_path.name)
        test_case['rag_response'] = None
        test_case['skipped'] = 'no input'
        tmp_path = out_path.with_suffix('.tmp')
        with open(tmp_path, 'w', encoding='utf-8') as f:
            json.dump(test_case, f, indent=2, ensure_ascii=False)
        os.replace(tmp_path, out_path)
        return

    try:
        context = retrieve_context(collection, user_input)
        test_case['rag_context'] = context  # Add the retrieved context to the test case
        system_msg: ChatCompletionSystemMessageParam = {
            "role": "system",
            "content": (
                "You are an AI assistant specialized on the interaction with a core banking system by generating SQL queries or API calls."
                "You will receive natural language user input and relevant documentation excerpts. Your task is to generate a precise and correct response based on the provided context."
                "If you need to trigger an action in the core banking system, you need to use API calls. Do not use UPDATE, DELETE or INDERT statements."
                "IF you want to retrieve data from the core banking system, you need to use SQL SELECT statements. Do not use API calls for data retrieval."
                """Here is an example of the desired response format for API calls:\n
{
    "method": "PATCH",
    "endpoint": "/BankCardContractLockRequests/105",
    "body": {
        "ValidityEndDate": "2024-12-31"
    }
}"""
                """Here is an example of the desired response format for SQL queries:\n
SELECT UUID FROM BankAccountContract WHERE TypeName = 'Loan Contract' AND BorrowerPartyIdentifyingElements = 'specific_borrower_id'
                """
                "It is very important that you only answer with a single SQL statement or a single API call, without any additional text."
                "In every response consider May 1st 2025 as the current date."
                "Use the following documentation excerpts to answer the user precisely."
                f"\n\nDocumentation excerpts:\n{context}"
            ),
        }
        user_msg: ChatCompletionUserMessageParam = {
            "role": "user",
            "content": user_input,
        }

        logger.info(f"Calling chat_completion for file: {in_path}")
        response = chat_completion([system_msg, user_msg])
        print(response)
        content = ''
        if response is not None and hasattr(response, 'choices') and response.choices:
            content = response.choices[0].message.content

        if not content:
            logger.warning('Received empty response content for %s', in_path.name)
            test_case['rag_response'] = None
            test_case['error'] = 'empty content'
        else:
            test_case['rag_response'] = content.strip()

    except Exception as e:
        logger.error('Error processing %s: %s', in_path.name, e)
        test_case['rag_response'] = None
        test_case['error'] = str(e)

    tmp_path = out_path.with_suffix('.tmp')
    with open(tmp_path, 'w', encoding='utf-8') as f:
        json.dump(test_case, f, indent=2, ensure_ascii=False)
    os.replace(tmp_path, out_path)
    logger.info('Saved result to %s', out_path)

def self_rag_loop(collection: Any, user_input: str, max_steps: int = 5) -> dict[str, Any]:
    """
    Implements the Self-RAG loop: lets the LLM decide when to retrieve, reflect, and finalize the answer.
    Returns a dict with all intermediate steps and the final answer.
    """
    steps: List[dict[str, Any]] = []
    current_answer = ""
    retrieved_context = ""
    for step in range(max_steps):
        # 1. System prompt for self-reflective generation
        system_msg: ChatCompletionSystemMessageParam = {
            "role": "system",
            "content": (
                "You are an AI assistant specialized on the interaction with a core banking system by generating SQL queries or API calls. "
                "You will receive natural language user input and, if needed, relevant documentation excerpts. "
                "You may use the special token [RETRIEVE] in your answer if you need more information from documentation. "
                "If you want to critique or reflect on your answer, use the token [REFLECT] and explain your reasoning. "
                "If you have enough information, answer directly. Otherwise, use [RETRIEVE] or [REFLECT] as needed. "
                "If you need to trigger an action in the core banking system, use API calls. Do not use UPDATE, DELETE or INDERT statements. "
                "If you want to retrieve data, use SQL SELECT statements. Do not use API calls for data retrieval. "
                "Here is an example of the desired response format for API calls:\n"
                "{\n    \"method\": \"PATCH\",\n    \"endpoint\": \"/BankCardContractLockRequests/105\",\n    \"body\": {\n        \"ValidityEndDate\": \"2024-12-31\"\n    }\n}"
                "Here is an example of the desired response format for SQL queries:\n"
                "SELECT UUID FROM BankAccountContract WHERE TypeName = 'Loan Contract' AND BorrowerPartyIdentifyingElements = 'specific_borrower_id' "
                "It is very important that you only answer with a single SQL statement or a single API call, without any additional text. "
                "In every response consider May 1st 2025 as the current date. "
            )
        }
        # 2. User message: user input + (optional) retrieved context + (optional) previous answer
        user_content = user_input
        if retrieved_context:
            user_content += f"\n\n[DOCUMENTATION]\n{retrieved_context}"
        if current_answer:
            user_content += f"\n\n[PREVIOUS_ANSWER]\n{current_answer}"
        user_msg: ChatCompletionUserMessageParam = {"role": "user", "content": user_content}
        response = chat_completion([system_msg, user_msg])
        answer = response.choices[0].message.content if response and hasattr(response, 'choices') and response.choices else ""
        steps.append({
            "step": step + 1,
            "answer": answer,
            "retrieved_context": retrieved_context
        })
        # 3. Check for [RETRIEVE] or [REFLECT] tokens
        if answer is not None and "[RETRIEVE]" in answer:
            # Extract a retrieval query (could be the user_input or a new question)
            # For simplicity, use the user_input as the retrieval query, or extract a query from the answer
            retrieval_query = user_input
            # Optionally, parse a more specific query from the answer
            retrieved_context = retrieve_context(collection, retrieval_query)
            current_answer = answer.replace("[RETRIEVE]", "")
            continue
        elif answer is not None and "[REFLECT]" in answer:
            # Let the LLM reflect and possibly revise its answer
            current_answer = answer.replace("[REFLECT]", "")
            continue
        else:
            # Final answer (no more retrieval or reflection needed)
            return {
                "steps": steps,
                "final_answer": answer.strip() if answer is not None else ""
            }
    # If max_steps reached, return last answer
    return {
        "steps": steps,
        "final_answer": steps[-1]["answer"] if steps else ""
    }

def process_test_file_selfrag(collection: Any, in_path: Path, out_path: Path):
    """
    Load a test case JSON, run the Self-RAG loop, and write all steps and the final answer to output.
    """
    logger.info('Processing test file %s', in_path)
    try:
        with open(in_path, 'r', encoding='utf-8') as f:
            test_case = json.load(f)
    except Exception as e:
        logger.error('Failed to load test case %s: %s', in_path.name, e)
        return

    user_input = test_case.get('input', '').strip()
    if not user_input:
        logger.warning('Skipping %s: missing input', in_path.name)
        test_case['selfrag_steps'] = []
        test_case['selfrag_final_answer'] = None
        test_case['skipped'] = 'no input'
        tmp_path = out_path.with_suffix('.tmp')
        with open(tmp_path, 'w', encoding='utf-8') as f:
            json.dump(test_case, f, indent=2, ensure_ascii=False)
        os.replace(tmp_path, out_path)
        return

    try:
        selfrag_result = self_rag_loop(collection, user_input)
        test_case['selfrag_steps'] = selfrag_result.get('steps', [])
        test_case['selfrag_final_answer'] = selfrag_result.get('final_answer', None)
    except Exception as e:
        logger.error('Error processing %s: %s', in_path.name, e)
        test_case['selfrag_steps'] = []
        test_case['selfrag_final_answer'] = None
        test_case['error'] = str(e)

    tmp_path = out_path.with_suffix('.tmp')
    with open(tmp_path, 'w', encoding='utf-8') as f:
        json.dump(test_case, f, indent=2, ensure_ascii=False)
    os.replace(tmp_path, out_path)
    logger.info('Saved result to %s', out_path)

if __name__ == '__main__':
    # Main execution: init Chroma and process all test files
    collection = init_chroma()

    # Recursively search for .json files in all subdirectories of API and SQL
    api_files = list(API_DIR.rglob('*.json'))
    sql_files = list(SQL_DIR.rglob('*.json'))
    logger.info(f"Found {len(api_files)} API test cases and {len(sql_files)} SQL test cases.")

    # Process API-confirmed cases, preserving subdirectory structure
    for infile in api_files:
        rel_path = infile.relative_to(API_DIR)
        outfile = OUTPUT_API / rel_path
        outfile.parent.mkdir(parents=True, exist_ok=True)
        process_test_file_selfrag(collection, infile, outfile)

    # Process SQL-confirmed cases, preserving subdirectory structure
    for infile in sql_files:
        rel_path = infile.relative_to(SQL_DIR)
        outfile = OUTPUT_SQL / rel_path
        outfile.parent.mkdir(parents=True, exist_ok=True)
        process_test_file_selfrag(collection, infile, outfile)

    logger.info('All test cases processed.')
