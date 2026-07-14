"""
Create the OpenSearch index for papers, with both a BM25-searchable
text mapping and a knn_vector field reserved for SPECTER2 embeddings
(768-dim) added in the next step. Defining the vector field now avoids
a reindex later -- OpenSearch needs dimension fixed at creation time.

Usage:
    python index/create_index.py
"""
from opensearchpy import OpenSearch

INDEX_NAME = "papers"
EMBEDDING_DIM = 768  # SPECTER2 output dimension

client = OpenSearch(hosts=[{"host": "localhost", "port": 9200}])

INDEX_BODY = {
    "settings": {
        "index": {
            "knn": True,
            "knn.algo_param.ef_search": 100,
        }
    },
    "mappings": {
        "properties": {
            "id": {"type": "keyword"},
            "title": {"type": "text", "analyzer": "english"},
            "abstract": {"type": "text", "analyzer": "english"},
            "authors": {"type": "keyword"},
            "year": {"type": "integer"},
            "categories": {"type": "keyword"},
            "citation_count": {"type": "integer"},
            "embedding": {
                "type": "knn_vector",
                "dimension": EMBEDDING_DIM,
                "method": {
                    "name": "hnsw",
                    "space_type": "cosinesimil",
                    "engine": "nmslib",
                },
            },
        }
    },
}

def main():
    if client.indices.exists(index=INDEX_NAME):
        print(f"Index '{INDEX_NAME}' already exists, deleting to recreate...")
        client.indices.delete(index=INDEX_NAME)
    client.indices.create(index=INDEX_NAME, body=INDEX_BODY)
    print(f"Created index '{INDEX_NAME}'")

if __name__ == "__main__":
    main()