import os
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import chromadb
from sentence_transformers import SentenceTransformer

app = FastAPI(title="Local Financial RAG Engine")

# 1. Enable CORS for Next.js dev server (handles OPTIONS preflight automatically)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. Path-safe directory resolution for ChromaDB
script_dir = os.path.dirname(os.path.abspath(__file__))
db_path = os.path.join(script_dir, "chroma_db")

# 3. Load model and vector store once into memory at startup
print("Loading local embedding model (all-MiniLM-L6-v2)...")
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

chroma_client = chromadb.PersistentClient(path=db_path)
collection = chroma_client.get_collection("financial_records")

class QueryRequest(BaseModel):
    query: str
    top_k: int = 3

@app.post("/retrieve-financial-docs")
def retrieve_docs(request: QueryRequest):
    # Embed query locally
    query_vector = model.encode([request.query]).tolist()
    
    # Query ChromaDB with cosine similarity
    results = collection.query(
        query_embeddings=query_vector,
        n_results=request.top_k
    )
    
    # Format grounded excerpts for Next.js / Bedrock
    retrieved_chunks = []
    if results.get("documents") and results["documents"] and results.get("metadatas"):
        for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
            retrieved_chunks.append({
                "source": meta.get("source", "internal_filings"),
                "content": doc
            })
            
    return {"results": retrieved_chunks}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)