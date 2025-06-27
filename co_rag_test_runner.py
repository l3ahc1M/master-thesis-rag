"""
co_rag_test_runner.py

This script implements a Chain-of-Retrieval Augmented Generation (CoRAG) framework for multi-hop reasoning and complex query answering.
CoRAG iteratively decomposes complex queries into sub-questions, retrieves evidence for each, and chains the results to produce a final answer.
It is designed for use in both QA and code generation scenarios, supporting multi-step reasoning and retrieval.

Key functionalities:
- Loads test cases from specified directories (API, SQL, etc.)
- For each test case, runs a CoRAG loop: decomposes the query, retrieves context for each sub-query, and generates a final answer
- Stores intermediate steps and final results for analysis
- Supports batch processing and logging for reproducibility

Usage:
Run this script directly to process all test cases and generate results. Configure paths and parameters as needed for your environment.
"""
import os
import json
import logging
import time
from pathlib import Path
from typing import List, Any

import openai
from openai.types.chat import ChatCompletionMessageParam
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
API_DIR  = BASE_DIR / 'API_confirmed'
SQL_DIR  = BASE_DIR / 'SQL_confirmed'

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
        n_results=top_k
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

def decompose_query(query: str, context: str = "") -> List[str]:
    """
    Use the LLM to decompose a complex query into a list of sub-questions.
    Optionally, use context from previous steps.
    """
    system_prompt = (
        "You are an expert at breaking down complex questions into a sequence of simpler sub-questions "
        "that, when answered in order, will allow a system to answer the original question. "
        "Given the following user query, return a list of sub-questions in the order they should be answered. "
        "If the query is simple, return it as a single-item list."
    )
    user_prompt = f"Original query: {query}"
    if context:
        user_prompt += f"\nContext so far: {context}"
    messages: List[ChatCompletionMessageParam] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    response = chat_completion(messages)
    content = getattr(response.choices[0].message, 'content', '') if response and hasattr(response, 'choices') and response.choices else ''
    if not content:
        return [query]
    try:
        sub_questions = json.loads(content if content is not None else "[]")
        if isinstance(sub_questions, list):
            return [str(q).strip() for q in sub_questions if isinstance(q, str) and q.strip()] # type: ignore
    except Exception:
        if isinstance(content, str):
            return [line.strip("- ").strip() for line in content.splitlines() if line.strip()]
    return [query]

def corag_chain(collection: Any, original_query: str, max_steps: int = 3) -> dict[str, Any]:
    """
    Implements the CoRAG loop: iteratively decomposes the query, retrieves context, and generates answers.
    Returns a dict with all intermediate steps and the final answer.
    """
    steps: List[dict[str, Any]] = []
    current_query: str = original_query
    context_so_far = ""
    for step_num in range(max_steps):
        if step_num == 0:
            sub_questions = decompose_query(current_query)
        else:
            sub_questions = [current_query]
        for subq in sub_questions:
            if not subq:
                continue
            retrieved_context = retrieve_context(collection, subq)
            system_msg: ChatCompletionMessageParam = {
                "role": "system",
                "content": (
                    "You are an AI assistant specialized on the interaction with a core banking system by generating SQL queries or API calls. "
                    "You will receive natural language user input and relevant documentation excerpts. Your task is to generate a precise and correct response based on the provided context. "
                    "If you need to trigger an action in the core banking system, you need to use API calls. Do not use UPDATE, DELETE or INDERT statements. "
                    "IF you want to retrieve data from the core banking system, you need to use SQL SELECT statements. Do not use API calls for data retrieval. "
                    "Here is an example of the desired response format for API calls:\n"
                    "{\n    \"method\": \"PATCH\",\n    \"endpoint\": \"/BankCardContractLockRequests/105\",\n    \"body\": {\n        \"ValidityEndDate\": \"2024-12-31\"\n    }\n}"
                    "Here is an example of the desired response format for SQL queries:\n"
                    "SELECT UUID FROM BankAccountContract WHERE TypeName = 'Loan Contract' AND BorrowerPartyIdentifyingElements = 'specific_borrower_id' "
                    "It is very important that you only answer with a single SQL statement or a single API call, without any additional text. "
                    "In every response consider May 1st 2025 as the current date. "
                    "Use the following documentation excerpts to answer the user precisely. "
                    f"\n\nDocumentation excerpts:\n{retrieved_context} "
                    "If you have gathered all the information needed to answer the original user query, respond with the final answer only and prepend the phrase 'FINAL ANSWER:' to your response. If you need to ask a follow-up question or require more information, state your next sub-question instead."
                ),
            }
            user_msg: ChatCompletionMessageParam = {"role": "user", "content": subq}
            response = chat_completion([system_msg, user_msg])
            answer = getattr(response.choices[0].message, 'content', '') if response and hasattr(response, 'choices') and response.choices else ''
            steps.append({
                "step": step_num + 1,
                "sub_question": subq,
                "retrieved_context": retrieved_context,
                "answer": answer
            })
            if step_num == max_steps - 1 or (isinstance(answer, str) and "final answer" in answer.lower()):
                return {
                    "original_query": original_query,
                    "steps": steps,
                    "final_answer": answer
                }
            current_query = answer if answer else current_query
            context_so_far += f"\n{subq}\n{retrieved_context}\n{answer}\n"
    return {
        "original_query": original_query,
        "steps": steps,
        "final_answer": steps[-1]["answer"] if steps else ""
    }

def process_test_file_corag(collection: Any, in_path: Path, out_path: Path):
    """
    Load a test case JSON, run the CoRAG chain, and write all steps and the final answer to output.
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
        test_case['corag_steps'] = []
        test_case['corag_final_answer'] = None
        test_case['skipped'] = 'no input'
        tmp_path = out_path.with_suffix('.tmp')
        with open(tmp_path, 'w', encoding='utf-8') as f:
            json.dump(test_case, f, indent=2, ensure_ascii=False)
        os.replace(tmp_path, out_path)
        return

    try:
        corag_result = corag_chain(collection, user_input)
        test_case['corag_steps'] = corag_result.get('steps', [])
        test_case['corag_final_answer'] = corag_result.get('final_answer', None)
    except Exception as e:
        logger.error('Error processing %s: %s', in_path.name, e)
        test_case['corag_steps'] = []
        test_case['corag_final_answer'] = None
        test_case['error'] = str(e)

    tmp_path = out_path.with_suffix('.tmp')
    with open(tmp_path, 'w', encoding='utf-8') as f:
        json.dump(test_case, f, indent=2, ensure_ascii=False)
    os.replace(tmp_path, out_path)
    logger.info('Saved result to %s', out_path)

if __name__ == '__main__':
    # Main execution: init Chroma and process all test files
    collection = init_chroma()

    # Recursively search for .json files in all subdirectories of API_confirmed and SQL_confirmed
    api_files = list(API_DIR.rglob('*.json'))
    sql_files = list(SQL_DIR.rglob('*.json'))
    logger.info(f"Found {len(api_files)} API test cases and {len(sql_files)} SQL test cases.")

    # Process API-confirmed cases, preserving subdirectory structure
    for infile in api_files:
        rel_path = infile.relative_to(API_DIR)
        outfile = OUTPUT_API / rel_path
        outfile.parent.mkdir(parents=True, exist_ok=True)
        process_test_file_corag(collection, infile, outfile)

    # Process SQL-confirmed cases, preserving subdirectory structure
    for infile in sql_files:
        rel_path = infile.relative_to(SQL_DIR)
        outfile = OUTPUT_SQL / rel_path
        outfile.parent.mkdir(parents=True, exist_ok=True)
        process_test_file_corag(collection, infile, outfile)

    logger.info('All test cases processed.')
