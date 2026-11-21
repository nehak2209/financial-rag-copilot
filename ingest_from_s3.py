import os
import boto3
import chromadb
from sentence_transformers import SentenceTransformer

# --- Configuration ---
BUCKET_NAME = "financial-copilot-assistant-data"  # Replace with your bucket name
PREFIX = "documents/"
DB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chroma_db")

# 1. Initialize local embedding model
print("Loading local embedding model (all-MiniLM-L6-v2)...")
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# 2. Connect to local persistent ChromaDB
client = chromadb.PersistentClient(path=DB_DIR)
collection = client.get_or_create_collection(
    name="financial_records",
    metadata={"hnsw:space": "cosine"}
)

# 3. Fetch documents directly from Amazon S3
s3 = boto3.client("s3")
print(f"Connecting to S3 bucket: {BUCKET_NAME}...")

response = s3.list_objects_v2(Bucket=BUCKET_NAME, Prefix=PREFIX)

documents = []
metadatas = []
ids = []

if "Contents" not in response:
    print(f"No objects found in s3://{BUCKET_NAME}/{PREFIX}")
    exit(1)

for obj in response["Contents"]:
    file_key = obj["Key"]
    
    # Skip directory markers and non-text files
    if file_key.endswith("/") or not file_key.endswith((".txt", ".md", ".json")):
        continue
        
    print(f"📥 Downloading and reading: s3://{BUCKET_NAME}/{file_key}")
    file_obj = s3.get_object(Bucket=BUCKET_NAME, Key=file_key)
    content = file_obj["Body"].read().decode("utf-8")
    
    # Split content into meaningful chunks by blank lines
    raw_chunks = [c.strip() for c in content.split("\n\n") if len(c.strip()) > 30]
    
    filename = os.path.basename(file_key)
    for idx, chunk in enumerate(raw_chunks):
        chunk_id = f"s3_{filename}_chunk_{idx}"
        documents.append(chunk)
        metadatas.append({"source": f"s3://{BUCKET_NAME}/{file_key}"})
        ids.append(chunk_id)

# 4. Generate embeddings locally and store in ChromaDB
if not documents:
    print("No valid text chunks found to embed.")
    exit(1)

print(f"\nGenerating local embeddings for {len(documents)} chunks from S3...")
embeddings = model.encode(documents).tolist()

collection.upsert(
    ids=ids,
    embeddings=embeddings,
    documents=documents,
    metadatas=metadatas
)

print(f"✅ Ingestion successful! Stored {len(documents)} vectors in {DB_DIR}")