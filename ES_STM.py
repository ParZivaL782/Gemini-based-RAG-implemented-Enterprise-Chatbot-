import os
import uuid
# Assuming util.py contains chunk_text and extract_text_from_pdf
from util import chunk_text, extract_text_from_pdf
from chromadb import PersistentClient
from chromadb.config import Settings
# Import the SentenceTransformerEmbeddingFunction
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

# ------------------ Configuration Paths ------------------

pdf_directory = "./docs"
chromadb_folder = "./chromadb_st" # Changed folder name to distinguish
collection_name = "pdf_documents" # Changed collection name to distinguish

# Define the Sentence Transformer model to use
# all-MiniLM-L6-v2 is a common default, fast and good for many tasks.
# You can change this to other models supported by the sentence-transformers library
# For example: 'paraphrase-MiniLM-L3-v2', 'multi-qa-MiniLM-L6-cos-v1', etc.
sentence_transformer_model = "all-MiniLM-L6-v2"

# Jina API key is not needed for Sentence Transformers running locally
# jina_api_key = "YOUR_JINA_API_KEY" # This line can be removed or commented

# ------------------ Setup ChromaDB + Sentence Transformers ------------------

# Instantiate the Sentence Transformer Embedding Function
# This will automatically download the model if it's not cached locally
sentence_transformer_ef = SentenceTransformerEmbeddingFunction(
    model_name=sentence_transformer_model
)

# Initialize the ChromaDB Persistent Client
client = PersistentClient(path=chromadb_folder)

# Check if the collection exists, create or get it, using the new embedding function
# Important: If you change the embedding function for an existing collection,
# you should ideally re-add all documents to generate embeddings with the new function.
# The code below handles this by clearing the collection if it exists.
if collection_name not in client.list_collections():
    print(f"Creating collection '{collection_name}' with Sentence Transformer model '{sentence_transformer_model}'")
    collection = client.create_collection(name=collection_name, embedding_function=sentence_transformer_ef)
else:
    print(f"Getting existing collection '{collection_name}'. Clearing existing data.")
    collection = client.get_collection(name=collection_name, embedding_function=sentence_transformer_ef)
    # Clear existing data to use the new embedding function consistently
    try:
        existing_ids = collection.get(include=[])["ids"] # Use include=[] for efficiency
        if existing_ids:
            print(f"Deleting {len(existing_ids)} existing items from the collection.")
            collection.delete(ids=existing_ids)
            print("Existing data cleared.")
    except Exception as e:
         print(f"Error clearing existing data: {e}")
         # Decide how to handle this error - maybe exit or proceed with caution
         # For robustness, you might want to ensure the collection is truly empty or recreate it.


# ------------------ Process and Store Documents ------------------

doc_count = 0
for pdf_file in os.listdir(pdf_directory):
    if not pdf_file.endswith(".pdf"):
        continue

    pdf_path = os.path.join(pdf_directory, pdf_file)
    print(f"\nProcessing PDF: {pdf_file}") # Added print for clarity
    full_text = extract_text_from_pdf(pdf_path)
    chunks = chunk_text(full_text)

    processed_chunks_count = 0 # Counter for chunks processed per file
    for chunk in chunks:
        # chunk=chunk.strip() # strip() is fine if you want leading/trailing whitespace removed
        # Skip empty chunks or those containing 'doi.org'
        if len(chunk) < 30 or 'doi.org' in chunk.lower():
             # print(f"Skipping small or unwanted chunk: {chunk[:50]}...") # Optional: uncomment for debugging skipped chunks
             continue

        try:
            doc_id = str(uuid.uuid4())
            # ChromaDB will automatically use the collection's embedding_function (SentenceTransformer in this case)
            # when you add documents.
            collection.add(
                ids=[doc_id],
                documents=[chunk],
                metadatas=[{"source": pdf_file}]
            )
            doc_count += 1
            processed_chunks_count += 1

            # print(f"Chunk Preview: {chunk[:100]}") # debugging
            # print(f"Stored chunk from {pdf_file} as ID {doc_id}") # This can be noisy, maybe print less often
        except Exception as e:
            print(f"Failed to process chunk from {pdf_file}: {e}")

    print(f"Finished processing {pdf_file}. Stored {processed_chunks_count} chunks.")


print(f"\nStored total chunks across all documents: {doc_count}")
print(f"Documents and embeddings stored successfully in Chroma at '{chromadb_folder}' using Sentence Transformers.")
