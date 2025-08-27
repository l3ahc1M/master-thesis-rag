import yaml
import os
import chromadb
from openai import OpenAI 
from dotenv import load_dotenv

from langchain_text_splitters import TokenTextSplitter
import hashlib
import json



# Get the root directory (assuming this script is in a subfolder)
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# load config yaml
cfg_path = os.path.join(root_dir, 'config.yaml')
with open(cfg_path, 'r') as f:
    cfg = yaml.safe_load(f)


# Set up ChromaDB persistence directory
chroma_persist_dir = os.path.join(root_dir, cfg.get('embedding', {}).get('chroma_persist_dir'))

# Load relevant variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
openai_embedding_model = cfg.get('embedding', {}).get('openAI_embedding_model')
top_n_entries = cfg.get('embedding', {}).get('top_n_entries')

text_splitter = TokenTextSplitter(
    chunk_size=cfg.get('embedding', {}).get('max_chunk_size'),
    chunk_overlap=cfg.get('embedding', {}).get('chunk_overlap'),
    )


def count_total_embeddings(db_path=chroma_persist_dir):
    client = chromadb.PersistentClient(path=db_path)
    total_embeddings = 0
    for collection in client.list_collections():
        coll = client.get_collection(collection.name)
        total_embeddings += coll.count()
    print(f"Total embeddings in ChromaDB: {total_embeddings}")



def store_chunk_text_file(text, file_name):
    """
    Stores the given text chunk as a .txt file in the ChromaDB persistence directory.
    The file will be named using the file_name parameter.
    """
    # Ensure the chroma_db directory exists
    os.makedirs(f"{chroma_persist_dir}/debug", exist_ok=True)
    # Define the path for the text file
    chunk_file_path = os.path.join(f"{chroma_persist_dir}/debug", f"{file_name}.txt")
    # Write the text to the file
    with open(chunk_file_path, "w", encoding="utf-8") as f:
        f.write(text)
   

def store_text_embedding(
    text,
    file_name,
    business_object,
    source_type,
    id,
    method=None,
    path=None,
    table=None,
    column=None
    ):



    raw_chunks = text_splitter.split_text(text)
    print(f"Number of of raw chunks: {len(raw_chunks)}")
    i = 1
    for raw_chunk in raw_chunks:
        

        if not isinstance(raw_chunk, str):
            raise TypeError("Input text must be a string")

        # Create embedding using OpenAI API
        client = OpenAI()
        response = client.embeddings.create(
            input=raw_chunk,
            model=openai_embedding_model
        )
        # Prepare metadata, filtering out None values
        metadata = {
            k: v for k, v in {
                "id": id + f"_{i}",
                "file_name": file_name,
                "business_object": business_object,
                "source_type": source_type,
                "method": method,
                "path": path,
                "table": table,
                "column": column,
            }.items() if v is not None
        }
        embedding = response.data[0].embedding

        # Create SHA256 hash using embedding to be used as unique ID
        hash_value = hashlib.sha256(str(embedding).encode("utf-8")).hexdigest()

        # Initialize ChromaDB client with persistence 

        client = chromadb.PersistentClient(
            path=chroma_persist_dir  
        )

        # Create or get collection based on knowledge type
        if source_type == "API":
            collection = client.get_or_create_collection(name="API_embeddings")
        elif source_type == "DB":
            collection = client.get_or_create_collection(name="DB_embeddings")
        elif source_type == "TXT":
            collection = client.get_or_create_collection(name="TXT_embeddings")
        else:
            raise ValueError(f"Unknown source type: {source_type}")

        # Store embedding with metadata
        collection.add(
            ids=[hash_value],  # Provide a unique ID for the embedding
            embeddings=[embedding],  # type: ignore
            documents=[text],
            metadatas=[metadata]  # type: ignore
        )
        store_chunk_text_file(
            text=raw_chunk,
            file_name=f"{business_object}_{file_name}_{i}_{hash_value}"
        )
        i += 1


    count_total_embeddings()

def get_embeddings_by_file_name(file_name, db_path=chroma_persist_dir):
    client = chromadb.PersistentClient(path=db_path)
    embeddings = []
    for collection in client.list_collections():
        coll = client.get_collection(collection.name)
        results = coll.get(where={"file_name": file_name})
        embeddings_list = results.get("embeddings") or []
        documents_list = results.get("documents") or []
        metadatas_list = results.get("metadatas") or []
        for emb, doc, meta in zip(embeddings_list, documents_list, metadatas_list):
            embeddings.append({
                "embedding": emb,
                "document": doc,
                "metadata": meta,
                "collection": collection.name
            })
    print(embeddings)


