import PyPDF2
import torch
from transformers import AutoModel, AutoTokenizer
import os
import numpy as np
import chromadb
import uuid
from chromadb import PersistentClient
from chromadb.config import Settings
from util import chunk_text, extract_text_from_pdf

# Embedding
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

def get_embeddings(text: str):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings.squeeze().numpy()
    



# PDF reading


# Chunking function (sliding window)


# Embedding function

# Paths
pdf_directory = "./docs"
chromadb_folder = "./chromadb"
collection_name = "pdf_documents"

# Setup ChromaDB
client = PersistentClient(path=chromadb_folder)

if collection_name not in client.list_collections():
    collection = client.create_collection(name=collection_name)
else:
    collection = client.get_collection(name=collection_name)
    existing_ids = collection.get()["ids"]
    if existing_ids:
        collection.delete(ids=existing_ids)

  

# Process and store documents
doc_count = 0
for pdf_file in os.listdir(pdf_directory):
    if not pdf_file.endswith(".pdf"):
        continue
    pdf_path = os.path.join(pdf_directory, pdf_file)
    full_text = extract_text_from_pdf(pdf_path)
    chunks = chunk_text(full_text)

    for chunk in chunks:
        embedding = get_embeddings(chunk)
        doc_id = str(uuid.uuid4())
        collection.add(
            ids=[doc_id],
            documents=[chunk],
            metadatas=[{"source": pdf_file}],
            embeddings=[embedding]
        )
        doc_count += 1
        print(f"Stored chunk from {pdf_file}...")

print(f"\nStored total chunks: {doc_count}")
print(f"Documents and embeddings stored successfully in Chroma at '{chromadb_folder}'.")
