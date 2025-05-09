import os
import uuid
import voyageai
from util import chunk_text, extract_text_from_pdf
from chromadb import PersistentClient
import time
import random

MAX_RETRIES = 5

# ------------------ Custom Voyage Embedding Function ------------------


from chromadb.api.types import EmbeddingFunction

class CustomVoyageEmbeddingFunction(EmbeddingFunction):
    def __init__(self, api_key, model="voyage-2"):
        self.api_key = api_key
        self.model = model
        voyageai.api_key = api_key

    def __call__(self, input):
        if not isinstance(input, list):
            input = [input]  # Ensure input is a list

        # Call the embed function (adjust to how it actually returns embeddings)
        response = voyageai.get_embeddings(input, model=self.model)
        
        # If response is a list of embeddings, return it directly
        return response  # Assumes response is a list of embeddings

# ------------------ Config ------------------
pdf_directory = "./docs"
chromadb_folder = "./chromadb_voyage"
collection_name = "pdf_documents"
voyage_api_key = "pa-7i_oLWDfuNdeLN4teyNCkza9z7ZpuTdidKMBC69hNSX"

voyage_ef = CustomVoyageEmbeddingFunction(api_key=voyage_api_key, model="voyage-3")
client = PersistentClient(path=chromadb_folder)

# ------------------ Create or Clear Collection ------------------
if collection_name not in client.list_collections():
    collection = client.create_collection(name=collection_name, embedding_function=voyage_ef)
else:
    collection = client.get_collection(name=collection_name, embedding_function=voyage_ef)
    existing_ids = collection.get()["ids"]
    if existing_ids:
        collection.delete(ids=existing_ids)

# ------------------ Process PDFs ------------------
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
