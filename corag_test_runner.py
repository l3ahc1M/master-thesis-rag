import os
import json
import logging
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

# The co-rag framework is assumed to provide a CoRAG class that handles
# both retrieval and generation. As this package may not be available in
# the execution environment, the imports are written as placeholders.
try:
    from corag import CoRAG
except Exception:  # pragma: no cover - library might not be installed
    CoRAG = None  # type: ignore

# ────────────────── CONFIGURATION ──────────────────
load_dotenv()  # Load environment variables from .env

PERSIST_DIR = 'corag_data'
COLLECTION_NAME = 'system_documentation'
EMBEDDING_MODEL = 'text-embedding-3-small'

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
GENERATION_MODEL = os.getenv('OPENAI_GEN_MODEL', 'gpt-3.5-turbo')
RAG_TOP_K = int(os.getenv('RAG_TOP_K', '5'))

BASE_DIR = Path('test_cases')
API_DIR = BASE_DIR / 'API_confirmed'
SQL_DIR = BASE_DIR / 'SQL_confirmed'

OUTPUT_BASE = Path('test_cases_results_corag')
OUTPUT_API = OUTPUT_BASE / 'API'
OUTPUT_SQL = OUTPUT_BASE / 'SQL'
for d in (OUTPUT_API, OUTPUT_SQL):
    d.mkdir(parents=True, exist_ok=True)

# ────────────────── LOGGING ──────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ────────────────── CO-RAG INITIALIZATION ──────────────────

def init_corag() -> Any:
    """Initialize the co-rag client and return it."""
    if CoRAG is None:
        raise ImportError("co-rag package is not installed")

    logger.info("Initializing CoRAG client at %s", PERSIST_DIR)
    return CoRAG(
        persist_path=PERSIST_DIR,
        collection_name=COLLECTION_NAME,
        embedding_model=EMBEDDING_MODEL,
        api_key=OPENAI_API_KEY,
        generation_model=GENERATION_MODEL,
    )


def safe_generate(client: Any, query: str, top_k: int = RAG_TOP_K, retries: int = 3, delay: int = 2):
    """Generate a response using the co-rag client with basic retry logic."""
    for attempt in range(retries):
        try:
            return client.run(query, top_k=top_k)
        except Exception as exc:  # pragma: no cover - external call
            logger.error("CoRAG call failed (attempt %d/%d): %s", attempt + 1, retries, exc)
            if attempt == retries - 1:
                raise
            time.sleep(delay)


# ────────────────── TEST FILE PROCESSING ──────────────────

def process_test_file(client: Any, in_path: Path, out_path: Path) -> None:
    """Load a test case, run it through co-rag and save the result."""
    logger.info('Processing test file %s', in_path)
    try:
        with open(in_path, 'r', encoding='utf-8') as f:
            test_case = json.load(f)
    except Exception as e:  # pragma: no cover - file handling
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
        result = safe_generate(client, user_input)
        if result is None:
            raise RuntimeError('no result returned')

        # The CoRAG result is assumed to have `answer` and `context` fields.
        test_case['rag_response'] = getattr(result, 'answer', None)
        test_case['rag_context'] = getattr(result, 'context', None)
    except Exception as e:  # pragma: no cover - external call
        logger.error('Error processing %s: %s', in_path.name, e)
        test_case['rag_response'] = None
        test_case['error'] = str(e)

    tmp_path = out_path.with_suffix('.tmp')
    with open(tmp_path, 'w', encoding='utf-8') as f:
        json.dump(test_case, f, indent=2, ensure_ascii=False)
    os.replace(tmp_path, out_path)
    logger.info('Saved result to %s', out_path)


# ────────────────── MAIN EXECUTION ──────────────────

if __name__ == '__main__':
    corag_client = init_corag()

    api_files = list(API_DIR.rglob('*.json'))
    sql_files = list(SQL_DIR.rglob('*.json'))
    logger.info("Found %d API test cases and %d SQL test cases.", len(api_files), len(sql_files))

    for infile in api_files:
        rel_path = infile.relative_to(API_DIR)
        outfile = OUTPUT_API / rel_path
        outfile.parent.mkdir(parents=True, exist_ok=True)
        process_test_file(corag_client, infile, outfile)

    for infile in sql_files:
        rel_path = infile.relative_to(SQL_DIR)
        outfile = OUTPUT_SQL / rel_path
        outfile.parent.mkdir(parents=True, exist_ok=True)
        process_test_file(corag_client, infile, outfile)

    logger.info('All test cases processed.')
