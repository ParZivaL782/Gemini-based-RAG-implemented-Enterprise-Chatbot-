import os
import uuid
from util import chunk_text, extract_text_from_pdf
from chromadb import PersistentClient
from chromadb.config import Settings
from chromadb.utils.embedding_functions import JinaEmbeddingFunction


# ------------------ Configuration Paths ------------------

pdf_directory = "./docs"
chromadb_folder = "./chromadb3"
collection_name = "pdf_documents"
jina_api_key = "jina_9ea0c05a3ece41beb434af45dc058c31QdxlAA7aty8e62u6x8V5SkgVU9Ez"

# ------------------ Setup ChromaDB + Gemini ------------------

jinaai_ef = JinaEmbeddingFunction(
                api_key=jina_api_key,
                model_name="jina-clip-v2"
            )
client = PersistentClient(path=chromadb_folder)

if collection_name not in client.list_collections():
    collection = client.create_collection(name=collection_name, embedding_function=jinaai_ef)
else:
    collection = client.get_collection(name=collection_name, embedding_function=jinaai_ef)
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
        if len(chunk) <100 or 'doi.org' in chunk.lower():
            continue
        try:
            doc_id = str(uuid.uuid4())
            collection.add(
                ids=[doc_id],
                documents=[chunk],
                metadatas=[{"source": pdf_file}]
            )
            doc_count += 1

            # print(f"Chunk Preview: {chunk[:100]}")      #debugging
            print(f"Stored chunk from {pdf_file} as ID {doc_id}")
        except Exception as e:
            print(f"Failed to process chunk: {e}")

print(f"\nStored total chunks: {doc_count}")
print(f"Documents and embeddings stored successfully in Chroma at '{chromadb_folder}'.")
