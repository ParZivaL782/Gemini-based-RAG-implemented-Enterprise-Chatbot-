import os
import uuid
from util import chunk_text, extract_text_from_pdf
from chromadb import PersistentClient
from chromadb.config import Settings
import chromadb.utils.embedding_functions as embedding_functions




# ------------------ Configuration Paths ------------------

pdf_directory = "./docs"
chromadb_folder = "./chromadb2"
collection_name = "pdf_documents"
google_api_key = "AIzaSyB7QnDF_mdps4fCH7klvrh1pAWBAjxzsq8"

# ------------------ Setup ChromaDB + Gemini ------------------

google_ef = embedding_functions.GoogleGenerativeAiEmbeddingFunction(api_key=google_api_key)

client = PersistentClient(path=chromadb_folder)

if collection_name not in [col.name for col in client.list_collections()]:
    collection = client.create_collection(name=collection_name, embedding_function=google_ef)
else:
    collection = client.get_collection(name=collection_name, embedding_function=google_ef)
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
        # chunk=chunk.strip()
        # Skip empty chunks or those containing 'doi.org'
        if len(chunk) <30 or 'doi.org' in chunk.lower():
            continue
        try:
            doc_id = str(uuid.uuid4())
            collection.add(
                ids=[doc_id],
                documents=[chunk],
                metadatas=[{"source": pdf_file}]
            )
            doc_count += 1

            print(f"Chunk Preview: {chunk[:100]}")      #debugging
            print(f"Stored chunk from {pdf_file} as ID {doc_id}")
        except Exception as e:
            print(f"Failed to process chunk: {e}")

print(f"\nStored total chunks: {doc_count}")
print(f"Documents and embeddings stored successfully in Chroma at '{chromadb_folder}'.")
