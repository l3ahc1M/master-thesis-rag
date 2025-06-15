"""
ingest_docs_full.py
-------------------

Enhanced ingestion pipeline covering:
1. Unit-of-retrieval chunking
2. $ref inlining & abbreviation expansion
3. Rich metadata (service_group, visibility)
4. Identifier normalization & synonyms
5. Versioning & soft-delete flags
6. Raw-text store beside vectors
7. Encryption-ready ChromaDB configuration
8. Observability & drift (logging, hash)
9. Test harness for dev-style queries
"""

import hashlib            # for computing hashes to detect drift
import json               # for reading JSON docs
import logging            # for observability
import os                 # for env vars
import re                 # for regex-based splitting & normalization
import uuid               # for unique chunk IDs
from datetime import datetime, timezone  # for timestamping
from pathlib import Path  # filesystem paths
from typing import Any, Dict, List, Tuple, Iterator
from dotenv import load_dotenv


import chromadb
import chromadb.utils.embedding_functions as embedding_functions
from tiktoken import encoding_for_model  # token count
from tqdm import tqdm                   # progress bars



# ──────────────────────────── ENVIRONMENT SETUP ────────────────────────────────────────
load_dotenv()
openai_api_key = os.getenv("openai_api_key")


# ────────────────────────────────────── CONFIG ─────────────────────────────────────────
DOC_ROOT        = Path("system_documentation")
PERSIST_DIR     = Path("chromadb_data")
RAW_STORE       = PERSIST_DIR / "raw_chunks"  # store raw chunk text for quoting
COLLECTION_NAME = "system_docs_v1"            # include version for namespacing
OPENAI_MODEL    = "text-embedding-3-small"     # code-aware embedding model
MAX_TOKENS      = 300                            # chunk size limit
SERVICE_GROUP   = os.getenv("SERVICE_GROUP", "DEFAULT_SERVICE")
VISIBILITY      = os.getenv("DEFAULT_VISIBILITY", "internal")


# Ensure raw store directory exists
RAW_STORE.mkdir(parents=True, exist_ok=True)

# Set up logging to file for ingestion operations
logging.basicConfig(
    filename=str(PERSIST_DIR / "ingest.log"),
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)
logger = logging.getLogger(__name__)

# Initialize tokenizer for counting tokens
_tok = encoding_for_model(OPENAI_MODEL)

# Abbreviations to expand for better semantic matching
ABBREV_MAP = {
    r"\bID\b": "identifier",
    r"\bUUID\b": "universal unique identifier",
}

# ────────────────────────── TOKEN & NORMALIZATION UTILITIES ────────────────────────────

def tokenize(text: str) -> List[int]:
    """Return a list of token IDs for the given text."""
    return _tok.encode(text)


def count_tokens(text: str) -> int:
    """Count tokens using the embedding model's tokenizer."""
    return len(tokenize(text))


def expand_abbrev(text: str) -> str:
    """Replace known abbreviations with full forms for clarity."""
    for pattern, full in ABBREV_MAP.items():
        text = re.sub(pattern, full, text)
    return text


def snake_case(name: str) -> str:
    """Convert CamelCase or kebab-case identifiers into snake_case."""
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    s2 = re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1)
    return s2.replace('-', '_').lower()


def hash_text(text: str) -> str:
    """Compute SHA256 hash for change detection (drift)."""
    return hashlib.sha256(text.encode('utf-8')).hexdigest()

# ────────────────────────────────── CHUNKING FUNCTION ───────────────────────────────────

def chunk_text(text: str, max_tokens: int = MAX_TOKENS) -> List[str]:
    """Split text into sentence-based chunks within the token budget."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    chunks: List[str] = []
    current: List[str] = []
    for sentence in sentences:
        candidate = " ".join(current + [sentence])
        # If adding this sentence exceeds budget, flush current chunk
        if current and count_tokens(candidate) > max_tokens:
            chunks.append(" ".join(current))
            current = [sentence]
        else:
            current.append(sentence)
    if current:
        chunks.append(" ".join(current))
    return chunks

# ────────────────────────────────── SCHEMA & API CHUNKERS ─────────────────────────────────

def _inline_ref(ref: str, components: Dict[str, Any]) -> Any:
    """Inline internal JSON `$ref` pointers in OpenAPI specs."""
    if not ref.startswith("#/"):
        return {"$ref": ref}
    node = components
    for part in ref.lstrip("#/components/").split("/"):
        node = node.get(part, {})
    return node


def _api_chunks(api: Dict[str, Any], bo: str, fname: str) -> Iterator[Dict[str, Any]]:
    """Yield endpoint-verb chunks with inlined `$ref` and metadata."""
    meta_base: Dict[str, Any] = {
        'source_type': 'openapi', 'business_object': bo, 'file_name': fname,
        'api_version': api.get('info', {}).get('version')
    }
    comps = api.get('components', {})
    for path, verbs in api.get('paths', {}).items():
        for verb, spec_untyped in verbs.items():
            if not isinstance(spec_untyped, dict):
                continue
            spec: Dict[str, Any] = spec_untyped  # type: ignore
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


def _db_chunks(schema: Dict[str, Any], bo: str, fname: str) -> Iterator[Dict[str, Any]]:
    """Yield table- and column-level chunks from a DB schema."""
    base = {'source_type': 'db_schema', 'business_object': bo, 'file_name': fname}
    for table in schema.get('tables', []):
        tid = str(uuid.uuid4())
        yield {'id': tid, 'text': f"Table {table['name']} {table.get('description','')}", 'meta': {**base, 'table_name': table['name'], 'deprecated': table.get('deprecated', False)}}
        for col in table.get('columns', []):
            cid = str(uuid.uuid4())
            yield {'id': cid, 'text': f"Column {col['name']} type {col.get('format')} desc {col.get('description','')}", 'meta': {**base, 'table_name': table['name'], 'column_name': col['name']}}

# ────────────────────────────────── INGESTION PIPELINE ─────────────────────────────────

def ingest() -> None:
    """Main ingestion workflow: scan, chunk, embed, and persist."""
    # Initialize embedder
    
    embedder = embedding_functions.OpenAIEmbeddingFunction( # type: ignore
                    api_key=openai_api_key,
                    model_name=OPENAI_MODEL
                )
    # Configure ChromaDB
    client = chromadb.PersistentClient(path=str(PERSIST_DIR))
    collection = client.get_or_create_collection(name=COLLECTION_NAME, embedding_function=embedder)  # type: ignore

    docs: List[Tuple[str, str, Dict[str, Any]]] = []

    def process_chunk(text: str, meta: Dict[str, Any]) -> None:
        """Normalize text, enrich metadata, store raw, and queue for embedding."""
        expanded = expand_abbrev(text)
        if 'column_name' in meta:
            meta['synonym'] = snake_case(meta['column_name'])
        meta['drift_hash'] = hash_text(expanded)
        meta.update({'service_group': SERVICE_GROUP,  'visibility': VISIBILITY, 'timestamp': datetime.now(timezone.utc).isoformat()})
        Path(RAW_STORE / f"{meta['id']}.txt").write_text(text)
        logger.info(f"Processed chunk {meta['id']} tokens={count_tokens(expanded)}")
        docs.append((meta['id'], expanded, meta))

    # Combined schema
    cp = DOC_ROOT / 'combined_db.json'
    if cp.exists():
        schema = json.loads(cp.read_text())
        for entry in _db_chunks(schema, '__all__', 'combined_db.json'):
            # ensure meta contains the chunk ID
            entry['meta']['id'] = entry['id']
            process_chunk(entry['text'], entry['meta'])

    # Per-object directories
    for obj in DOC_ROOT.iterdir():
        if not obj.is_dir(): continue
        bo = obj.name
        for f in obj.glob('DB_*.json'):
            for entry in _db_chunks(json.loads(f.read_text()), bo, f.name):
                entry['meta']['id'] = entry['id']
                process_chunk(entry['text'], entry['meta'])
        for f in obj.glob('API_*.json'):
            for entry in _api_chunks(json.loads(f.read_text()), bo, f.name):
                # ensure meta contains the chunk ID
                entry['meta']['id'] = entry['id']
                process_chunk(entry['text'], entry['meta'])
        for f in obj.glob('*.txt'):
            txt = f.read_text()
            for chunk in chunk_text(txt):
                process_chunk(chunk, {'id': str(uuid.uuid4()), 'source_type': 'semantic_doc', 'business_object': bo, 'file_name': f.name})

    # Batch upsert
    for i in tqdm(range(0, len(docs), 2), desc='Embedding batches'):
        batch = docs[i:i+2]
        ids, texts, metas = zip(*batch)
        collection.upsert(ids=list(ids), documents=list(texts), metadatas=list(metas))


    logger.info(f"Ingested {len(docs)} chunks into {COLLECTION_NAME}")

if __name__ == '__main__':
    ingest()
