"""
ingest_docs_full.py 

This script ingests system documentation and database schema files into a ChromaDB vector database.
It performs the following steps:
- Loads and chunks documentation and schema files (OpenAPI, DB schema, and text docs)
- Expands abbreviations and normalizes text
- Stores raw chunks for traceability
- Computes and stores metadata (hashes, timestamps, synonyms)
- Embeds and add data into ChromaDB with robust retry logic

"""
# Import necessary libraries for the script
import hashlib
import json
import logging
import os
import re
import uuid
import openai
import numpy as np
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple, Iterator

# Load environment variables from .env file
from dotenv import load_dotenv
import chromadb
import chromadb.utils.embedding_functions as embedding_functions
from tiktoken import encoding_for_model

# retry/backoff for handling errors and retrying operations

# ─────────────────────────────────── CONFIG ──────────────────────────────────────────
# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("openai_api_key")


# Set up important paths and constants
DOC_ROOT        = Path("system_documentation")  # Root directory for documentation
PERSIST_DIR     = Path("chromadb_data")         # Directory for ChromaDB persistence
RAW_STORE       = PERSIST_DIR / "raw_chunks"    # Directory for storing raw text chunks
COLLECTION_NAME = "system_documentation"        # ChromaDB collection name
OPENAI_MODEL    = "text-embedding-3-small"      # OpenAI embedding model
MAX_TOKENS      = 300                           # Max tokens per chunk
BATCH_SIZE      = 1                             # Batch size for adding
RAW_STORE.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────── LOGGING ─────────────────────────────────────────
# Set up logging to a file for debugging and tracking
PERSIST_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    filename=str(PERSIST_DIR / "ingest.log"),
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)
logger = logging.getLogger(__name__)

# chromadb logger for detailed debug output
chromadb_logger = logging.getLogger("chromadb")
chromadb_logger.setLevel(logging.DEBUG)
chromadb_logger.addHandler(logging.StreamHandler())

import sys

# Configure root logger
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(str(PERSIST_DIR / "ingest.log"))
    ]
)

# Explicitly set ChromaDB submodules to DEBUG
for name in [
    "chromadb",
    "chromadb.api",
    "chromadb.segment",
    "chromadb.ingest",
    "chromadb.db",
    "chromadb.utils",
    "chromadb.telemetry"
]:
    logging.getLogger(name).setLevel(logging.INFO)

# ───────────────────────────────── TOKENIZER & HELPERS ────────────────────────────────
# Set up the tokenizer for the OpenAI model
_tok = encoding_for_model(OPENAI_MODEL)

# Dictionary to expand common abbreviations in text
ABBREV_MAP = {
    r"\bID\b": "identifier",
    r"\bUUID\b": "universal unique identifier",
    r"\bUSD\b": "United States Dollar",
    r"\bEUR\b": "Euro",
    r"\bJPY\b": "Japanese Yen",
    r"\bGBP\b": "British Pound Sterling",
    r"\bAUD\b": "Australian Dollar",
    r"\bCAD\b": "Canadian Dollar",
    r"\bCHF\b": "Swiss Franc",
    r"\bCNY\b": "Chinese Yuan",
    r"\bHKD\b": "Hong Kong Dollar",
    r"\bNZD\b": "New Zealand Dollar",
    r"\bSEK\b": "Swedish Krona",
    r"\bKRW\b": "South Korean Won",
    r"\bSGD\b": "Singapore Dollar",
    r"\bNOK\b": "Norwegian Krone",
    r"\bMXN\b": "Mexican Peso",
    r"\bINR\b": "Indian Rupee",
    r"\bRUB\b": "Russian Ruble",
    r"\bZAR\b": "South African Rand",
    r"\bTRY\b": "Turkish Lira",
    r"\bBRL\b": "Brazilian Real",
    r"\bTWD\b": "New Taiwan Dollar",
    r"\bDKK\b": "Danish Krone",
    r"\bPLN\b": "Polish Zloty",
    r"\bTHB\b": "Thai Baht",
    r"\bIDR\b": "Indonesian Rupiah",
    r"\bHUF\b": "Hungarian Forint",
    r"\bCZK\b": "Czech Koruna",
    r"\bILS\b": "Israeli New Shekel",
    r"\bCLP\b": "Chilean Peso",
    r"\bPHP\b": "Philippine Peso",
    r"\bAED\b": "United Arab Emirates Dirham",
    r"\bCOP\b": "Colombian Peso",
    r"\bSAR\b": "Saudi Riyal",
    r"\bMYR\b": "Malaysian Ringgit",
    r"\bRON\b": "Romanian Leu",
}

# Tokenize a string into a list of token IDs
def tokenize(text: str) -> List[int]:
    return _tok.encode(text)

# Count the number of tokens in a string
def count_tokens(text: str) -> int:
    return len(tokenize(text))

# Replace abbreviations in text with their full forms
def expand_abbrev(text: str) -> str:
    for pat, full in ABBREV_MAP.items():
        text = re.sub(pat, full, text)
    return text

# Convert a string to snake_case (lowercase with underscores)
def snake_case(name: str) -> str:
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    s2 = re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1)
    return s2.replace('-', '_').lower()

# Create a SHA-256 hash of a string
def hash_text(text: str) -> str:
    return hashlib.sha256(text.encode('utf-8')).hexdigest()

# Split text into chunks, each with at most max_tokens tokens
def chunk_text(text: str, max_tokens: int = MAX_TOKENS) -> List[str]:
    """
    Split text into chunks, each with at most max_tokens tokens.
    Chunks are split on sentence boundaries for semantic coherence.
    """
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    chunks: List[str] = []
    current: List[str] = []
    for sentence in sentences:
        cand = " ".join(current + [sentence])
        if current and count_tokens(cand) > max_tokens:
            chunks.append(" ".join(current))
            current = [sentence]
        else:
            current.append(sentence)
    if current:
        chunks.append(" ".join(current))
    return chunks

# ───────────────────────────────── DB / API CHUNKERS ─────────────────────────────────
def _inline_ref(ref: str, comps: Dict[str, Any]) -> Any:
    """
    Helper to resolve $ref references in OpenAPI specs.
    Returns the referenced object or a dict with $ref if not found.
    """
    if not ref.startswith("#/"):
        return {"$ref": ref}
    node = comps
    for part in ref.lstrip("#/components/").split("/"):
        node = node.get(part, {})
    return node

def _api_chunks(api: Dict[str, Any], bo: str, fname: str) -> Iterator[Dict[str, Any]]:
    """
    Create text chunks from OpenAPI JSON files for ingestion.
    Each chunk contains HTTP method, path, summary, parameters, and responses.
    """
    meta_base: Dict[str, Any] = {
        'source_type':'openapi','business_object':bo,'file_name':fname,
        'api_version': api.get('info',{}).get('version')
    }
    comps = api.get('components',{})
    for path, verbs in api.get('paths',{}).items():
        for verb, spec_raw in verbs.items():
            if not isinstance(spec_raw, dict): continue
            from typing import cast
            spec = cast(Dict[str, Any], spec_raw)  # explicit type cast for type checker
            params = verbs.get('parameters', [])
            for p in params:
                if '$ref' in p:
                    p.update(_inline_ref(p['$ref'], comps))
            block = (
                f"{verb.upper()} {path}\n"
                f"Summary: {spec.get('summary','')}\n"
                f"Params: {json.dumps(params, ensure_ascii=False)}\n"
                f"Resp: {json.dumps(spec.get('responses',{}), ensure_ascii=False)}"
            )
            for txt in chunk_text(block):
                yield {
                    'id': str(uuid.uuid4()),
                    'text': txt,
                    'meta': {**meta_base, 'http_method': verb.upper(), 'path': path}
                }

# Create chunks from database schema JSON files
def _db_chunks(schema: Dict[str, Any], bo: str, fname: str) -> Iterator[Dict[str, Any]]:
    """
    Create text chunks from database schema JSON files for ingestion.
    Each chunk contains table and column information.
    """
    base = {'source_type':'db_schema','business_object':bo,'file_name':fname}
    for table in schema.get('tables',[]):
        tid = str(uuid.uuid4())
        yield {
            'id': tid,
            'text': f"Table {table['name']} {table.get('description','')}",
            'meta': {**base, 'table_name':table['name'], 'deprecated':table.get('deprecated', False)}
        }
        for col in table.get('columns',[]):
            cid = str(uuid.uuid4())
            yield {
                'id': cid,
                'text': f"Column {col['name']} type {col.get('format')} desc {col.get('description','')}",
                'meta': {**base, 'table_name': table['name'], 'column_name': col['name']}
            }

from typing import List

def get_embedding(text, model=OPENAI_MODEL): # type: ignore
    response = openai.embeddings.create(input=text, model=model) # type: ignore
    return response.data[0].embedding # type: ignore

# ─────────────────────────────────── ingest documentation ─────────────────────────────────

def safe_add(collection: Any, batch: List[Tuple[str,str,Dict[str,Any]]]) -> None:
    """
    Add a batch of documents into the ChromaDB collection with retry and exponential backoff.
    Logs success or failure for each batch.
    """
    
    (print(f"Adding batch of {len(batch)} docs..."))
    ids, docs, metas = zip(*batch)

    # Validation: ids is sequence of non-null strings, docs is sequence of non-null strings, metas is sequence of non-null dicts
    if not all(isinstance(i, str) for i in ids):
        raise ValueError("ids must be a sequence of non-null strings")
    if not all(isinstance(d, str) for d in docs):
        raise ValueError("docs must be a sequence of non-null strings")
    if not all(m is not None and isinstance(m, dict) for m in metas):
        raise ValueError("metas must be a sequence of non-null dicts")


    embeddings: List[Any] = []
    for doc in docs:
        embedding = get_embedding(doc) # debugging line
        embeddings.append(embedding)
    embeddings = [np.array(e, dtype=np.float32).tolist() for e in embeddings]

# add the batch into the collection
    print("len(ids): " + str(len(ids))) # debugging line
    print("len(docs): " + str(len(docs)))   # debugging line
    print("len(metas): " + str(len(metas))) # debugging line    
    print("len(embeddings): " + str(len(embeddings))) # debugging line
    print(type(embeddings[0]), len(embeddings[0]), type(embeddings[0][0])) # debugging line # type: ignore
    try:
        collection.add(
            ids=list(ids), 
            documents=list(docs), 
            metadatas=list(metas),
            embeddings=embeddings 
        )
    except Exception as e:
        logger.error(f"Failed to add batch: {e}")
        raise
    print (f"Added batch of {len(batch)} docs.")
    logger.debug(f"Successfully added batch of {len(batch)} docs.")


# ────────────────────────────────────── INGEST ────────────────────────────────────────
def ingest() -> None:
    """
    Main function to ingest all documentation and schema files into ChromaDB.
    Handles chunking, metadata enrichment, raw storage, and robust adding.
    """
    logger.info("Starting ingestion workflow.")
    # Connect to the ChromaDB database
    client = chromadb.PersistentClient(path=str(PERSIST_DIR))
    # Set up the OpenAI embedding function for ChromaDB
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=OPENAI_API_KEY,
        model_name=OPENAI_MODEL
    )
    try: 
        print("initializing collection")
        collection = client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=openai_ef # type: ignore
        )
        print("collection initialized")
    except Exception as e:
        print(f"Failed to initialize collection: {e}")
        logger.critical(f"Failed to initialize collection: {e}")
        return  # Exit the function early if collection is not initialized
        
    docs: List[Tuple[str,str,Dict[str,Any]]] = []

    def process_chunk(text: str, meta: Dict[str,Any]) -> None:
        """
        Process a single text chunk:
        - Expand abbreviations
        - Add metadata (hash, timestamp, synonym)
        - Store raw chunk to disk
        - Append to docs list for adding
        """
        expanded = expand_abbrev(text)
        if 'column_name' in meta:
            meta['synonym'] = snake_case(meta['column_name'])
        meta.update({
            'drift_hash': hash_text(expanded),
            'timestamp': datetime.now(timezone.utc).isoformat()
        })
        # ensure ID is in meta and is a string
        if not meta.get('id'):
            meta['id'] = str(uuid.uuid4())
        else:
            meta['id'] = str(meta['id'])
        # write raw
        try:
            Path(RAW_STORE / f"{meta['id']}.txt").write_text(text)
        except Exception as e:
            logger.error(f"Failed to write raw chunk {meta['id']}: {e}")
        docs.append((meta['id'], expanded, meta))
        logger.info(f"Queued chunk {meta['id']} tokens={count_tokens(expanded)}")

    # --- load & chunk combined schema ---
    cp = DOC_ROOT / 'combined_db.json'
    if cp.exists():
        schema = json.loads(cp.read_text())
        for entry in _db_chunks(schema, '__all__', 'combined_db.json'):
            process_chunk(entry['text'], entry['meta'])

    # --- per-business-object dir ---
    for obj in DOC_ROOT.iterdir():
        if not obj.is_dir(): continue
        bo = obj.name
        # DB_*.json
        for f in obj.glob('DB_*.json'):
            schema = json.loads(f.read_text())
            for entry in _db_chunks(schema, bo, f.name):
                process_chunk(entry['text'], entry['meta'])
        # API_*.json
        for f in obj.glob('API_*.json'):
            api = json.loads(f.read_text())
            for entry in _api_chunks(api, bo, f.name):
                process_chunk(entry['text'], entry['meta'])
        # *.txt docs
        for f in obj.glob('*.txt'):
            txt = f.read_text()
            # If the text fits within MAX_TOKENS, keep as one chunk; otherwise, split
            if count_tokens(txt) <= MAX_TOKENS:
                process_chunk(txt, {
                    'id': str(uuid.uuid4()),
                    'source_type':'semantic_doc',
                    'business_object':bo,
                    'file_name':f.name
                })
            else:
                for chunk in chunk_text(txt):
                    process_chunk(chunk, {
                        'id': str(uuid.uuid4()),
                        'source_type':'semantic_doc',
                        'business_object':bo,
                        'file_name':f.name
                    })

    # --- add in batches with retry ---
    total = len(docs)
    num_batches = (total + BATCH_SIZE - 1) // BATCH_SIZE
    logger.info(f"Adding {total} chunks in {num_batches} batches (size={BATCH_SIZE})")

    print(f"=> Adding {total} chunks in {num_batches} batches (size={BATCH_SIZE})")

    for i in range(num_batches):
        batch = docs[i*BATCH_SIZE : (i+1)*BATCH_SIZE]
        try:
            print("1") # debugging line
            safe_add(collection, batch)
        except Exception as e:
            logger.critical(f"Batch {i+1} failed after retries: {e}")
            raise

    # --- final sanity check ---
    stored = collection.count()
    logger.info(f"Ingestion complete: {stored}/{total} embeddings stored in “{COLLECTION_NAME}”")
    print(f"=> Stored {stored} of {total} chunks in collection “{COLLECTION_NAME}”")

# If this script is run directly, start the ingestion process
if __name__ == '__main__':
    ingest()
