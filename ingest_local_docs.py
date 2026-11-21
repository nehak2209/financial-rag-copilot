import os
import sys
import chromadb
from sentence_transformers import SentenceTransformer

# 1. Initialize local embedding model
print("Loading local embedding model...")
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# 2. Initialize persistent local ChromaDB
# Uses directory of the script to prevent path mismatch
script_dir = os.path.dirname(os.path.abspath(__file__))
db_path = os.path.join(script_dir, "chroma_db")

client = chromadb.PersistentClient(path=db_path)
collection = client.get_or_create_collection(
    name="financial_records",
    metadata={"hnsw:space": "cosine"}
)

# 3. Locate files dynamically
# Check both the script's root directory and common subdirectories
candidate_folders = [
    script_dir,
    os.path.join(script_dir, "data", "raw")
]

target_filenames = [
    "acme_logistics_10k_summary.txt",
    "commercial_lending_underwriting_policy.txt",
    "kyc_aml_compliance_framework.txt"
]

documents = []
metadatas = []
ids = []

for file_name in target_filenames:
    found_path = None
    for folder in candidate_folders:
        potential_path = os.path.join(folder, file_name)
        if os.path.exists(potential_path):
            found_path = potential_path
            break
            
    if not found_path:
        print(f"❌ File not found: {file_name}")
        continue

    print(f"📄 Reading: {found_path}")
    with open(found_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Split into chunks (separated by blank lines)
    chunks = [c.strip() for c in content.split("\n\n") if len(c.strip()) > 30]

    for idx, chunk in enumerate(chunks):
        chunk_id = f"{file_name}_chunk_{idx}"
        documents.append(chunk)
        metadatas.append({"source": file_name})
        ids.append(chunk_id)

# 4. Guard against empty lists
if not documents:
    print("\n🚨 ERROR: No chunks were created. Please verify that:")
    print(" 1. The .txt files actually contain text.")
    print(" 2. The files are not named with duplicate extensions (e.g., .txt.txt).")
    print(f" Searched directories:\n - " + "\n - ".join(candidate_folders))
    sys.exit(1)

# 5. Generate local embeddings and save to vector store
print(f"\nGenerating local embeddings for {len(documents)} chunks...")
embeddings = model.encode(documents).tolist()

collection.upsert(
    ids=ids,
    embeddings=embeddings,
    documents=documents,
    metadatas=metadatas
)

print(f"✅ Ingestion complete! Successfully stored {len(documents)} vectors in: {db_path}")