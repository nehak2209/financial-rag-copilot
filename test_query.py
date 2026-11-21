import os
import chromadb
from sentence_transformers import SentenceTransformer

# 1. Load the exact same embedding model
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# 2. Connect to the existing local ChromaDB directory
script_dir = os.path.dirname(os.path.abspath(__file__))
db_path = os.path.join(script_dir, "chroma_db")
client = chromadb.PersistentClient(path=db_path)
collection = client.get_collection(name="financial_records")

# 3. Test question
query_text = "What is Acme Logistics Group's Net Operating Income and debt service?"
print(f"🔍 Query: {query_text}\n")

# Embed the query and retrieve top 2 closest chunks
query_embedding = model.encode([query_text]).tolist()
results = collection.query(
    query_embeddings=query_embedding,
    n_results=2
)

# 4. Display matched chunks
for idx, (doc, meta) in enumerate(zip(results['documents'][0], results['metadatas'][0])):
    print(f"--- Result {idx + 1} (Source: {meta['source']}) ---")
    print(doc)
    print()