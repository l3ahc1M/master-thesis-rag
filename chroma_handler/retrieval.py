import yaml
import os
import chromadb
from openai import OpenAI 
from dotenv import load_dotenv

load_dotenv()
# load config yaml
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
cfg_path = os.path.join(root_dir, 'config.yaml')
with open(cfg_path, 'r') as f:
    cfg = yaml.safe_load(f)

# Set up ChromaDB persistence directory path
chroma_persist_dir = os.path.join(root_dir, cfg.get('embedding', {}).get('chroma_persist_dir'))

def retrieve_n_closest_entries(query_text):
    # Create embedding for the query text
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    query_embedding = client.embeddings.create(
        input=query_text,
        model=cfg.get('embedding', {}).get('openAI_embedding_model')
    ).data[0].embedding
    # Initialize ChromaDB client
    chroma_client = chromadb.PersistentClient(path=chroma_persist_dir)
    # Get collections based on knowledge basis from config -> based on scenario, different collections are queried
    knowledge_basis = cfg.get('knowledge_basis', '').upper()
    if knowledge_basis == "DB":
        collections = ["DB_embeddings", "TXT_embeddings"]
    elif knowledge_basis == "API":
        collections = ["API_embeddings", "TXT_embeddings"]
    else:
        collections = [col.name for col in chroma_client.list_collections()]
    results = []
    for collection_name in collections:
        collection = chroma_client.get_collection(collection_name)
        query_result = collection.query(
            query_embeddings=[query_embedding],
            n_results=cfg.get('embedding', {}).get('top_n_entries')
        )
        # Collect results with collection name
        for i in range(len(query_result['ids'][0])):
            results.append({
                "collection": collection_name,
                "id": query_result['ids'][0][i],
                "document": query_result['documents'][0][i], # type: ignore
                "metadata": query_result['metadatas'][0][i], # type: ignore
                "distance": query_result['distances'][0][i] # type: ignore
            })
    # Sort all results by distance (ascending)
    results.sort(key=lambda x: x['distance'])
    # Return top n closest entries across all collections
    return results[:cfg.get('embedding', {}).get('top_n_entries')]

