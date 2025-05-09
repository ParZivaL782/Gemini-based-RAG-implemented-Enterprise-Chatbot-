import os
import uuid
from util import chunk_text, extract_text_from_pdf
from chromadb import PersistentClient
from chromadb.config import Settings
from chromadb.utils.embedding_functions import CohereEmbeddingFunction
import time
import random

MAX_RETRIES = 5

# ------------------ Configuration Paths ------------------

pdf_directory = "./docs"
chromadb_folder = "./chromadb_cohere"
collection_name = "pdf_documents"
cohere_api_key = "XHAIlBUHvy9PwpgBVNWk0PV8m36axqZxOJbbEYXn"

# ------------------ Setup ChromaDB + Gemini ------------------

cohere_ef = CohereEmbeddingFunction(
                api_key=cohere_api_key,
                model_name="embed-english-v3.0"
            )
client = PersistentClient(path=chromadb_folder)

if collection_name not in client.list_collections():
    collection = client.create_collection(name=collection_name, embedding_function=cohere_ef)
else:
    collection = client.get_collection(name=collection_name, embedding_function=cohere_ef)
    # Clear existing data if any
    existing_ids = collection.get()["ids"]
    if existing_ids:
        collection.delete(ids=existing_ids)



# ------------------ Process and Store Documents ------------------

doc_count = 0
for pdf_file in os.listdir(pdf_directory):
    if not pdf_file.endswith(".pdf"):
        continue

    pdf_path = os.path.join(pdf_directory, pdf_file)
    full_text = extract_text_from_pdf(pdf_path)
    chunks = chunk_text(full_text)

    for chunk in chunks:
        if len(chunk) < 30 or 'doi.org' in chunk.lower():
            continue
        doc_id = str(uuid.uuid4())

        for attempt in range(MAX_RETRIES):
            try:
                collection.add(
                    ids=[doc_id],
                   documents=[chunk],
                    metadatas=[{"source": pdf_file}]
                )
                print(f"Stored chunk from {pdf_file} as ID {doc_id}")
                doc_count += 1

                time.sleep(1.5)  # basic throttle
                break  # success, break retry loop

            except Exception as e:
                wait_time = 2 ** attempt + random.uniform(0, 1)
                print(f"Retry {attempt+1}/{MAX_RETRIES} for chunk due to error: {e}")
                time.sleep(wait_time)
        else:
            print(f"Failed to process chunk after {MAX_RETRIES} retries.")

print(f"\nStored total chunks: {doc_count}")
print(f"Documents and embeddings stored successfully in Chroma at '{chromadb_folder}'.")
