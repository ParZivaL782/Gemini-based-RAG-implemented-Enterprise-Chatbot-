import os
import torch
from chromadb import PersistentClient
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from sentence_transformers import CrossEncoder
from util import llm_parsing2 

# ------------------ Configuration ------------------

chromadb_folder = "./chromadb_st"
collection_name = "pdf_documents"
embedding_model_name = "all-MiniLM-L6-v2" # Or "all-mpnet-base-v2", "BAAI/bge-base-en-v1.5", etc.
reranker_model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"


# ------------------ Embedding + Chroma Setup ------------------
# Instantiate the SAME Embedding Function used for data ingestion
sentence_transformer_ef = SentenceTransformerEmbeddingFunction(
    model_name=embedding_model_name
)

client = PersistentClient(path=chromadb_folder)

try:
    # Get the existing collection. Make sure the embedding_function matches!
    collection = client.get_collection(name=collection_name, embedding_function=sentence_transformer_ef)
    print(f"Successfully loaded collection '{collection_name}'.")
except Exception as e:
    print(f"Error loading collection '{collection_name}': {e}")
    print(f"Please ensure you have run the ingestion script to create the collection at '{chromadb_folder}' with the embedding model '{embedding_model_name}'.")
    exit() # Exit if collection loading fails

# ------------------ Reranker Setup ------------------
# Load the Sentence Transformer Cross-Encoder model for re-ranking
try:
    reranker = CrossEncoder(reranker_model_name)
    print(f"Successfully loaded re-ranker model '{reranker_model_name}'.")
except Exception as e:
    print(f"Error loading re-ranker model '{reranker_model_name}': {e}")
    print("Please ensure you have the 'sentence-transformers' library installed (`pip install sentence-transformers`).")
    exit() # Exit if reranker loading fails


# ------------------ Retrieve + Rerank ------------------
def retrieve_and_rerank(query: str, initial_fetch_k: int = 50, final_top_k: int = 10):
    """
    Performs initial retrieval using vector similarity and then re-ranks
    the results using a cross-encoder Sentence Transformer model.

    Args:
        query: The user's query string.
        initial_fetch_k: The number of documents to retrieve initially from ChromaDB.
                         Needs to be >= final_top_k.
        final_top_k: The number of top results to return after re-ranking.

    Returns:
        A list of dictionaries, each containing 'document', 'metadata',
        and 'rerank_score', sorted by rerank_score,
        or an empty list if no results are found.
    """
    if initial_fetch_k < final_top_k:
        print(f"Warning: initial_fetch_k ({initial_fetch_k}) should be >= final_top_k ({final_top_k}). Adjusting initial_fetch_k.")
        initial_fetch_k = final_top_k


    print(f"\nPerforming initial retrieval for query: '{query}' (fetching {initial_fetch_k} results)...")

    # 1. Initial Retrieval using ChromaDB (Vector Similarity)
    # ChromaDB uses the collection's embedding function (Sentence Transformer bi-encoder)
    # to embed the query and find the nearest neighbors.
    try:
        results = collection.query(
            query_texts=[query], # Pass query text directly, collection uses its ef
            n_results=initial_fetch_k,
            include=["documents", "metadatas"] # No need to include embeddings for reranking
        )
        retrieved_chunks = results.get("documents", [[]])[0]
        retrieved_metadatas = results.get("metadatas", [[]])[0]

        if not retrieved_chunks:
            print("No documents found for the initial query.")
            return []

        print(f"Initial retrieval complete. Found {len(retrieved_chunks)} potential candidates.")

    except Exception as e:
        print(f"Error during initial retrieval: {e}")
        return []

    # 2. Prepare Data for Re-ranking
    # The reranker expects pairs of (query, document_text)
    sentence_pairs = [[query, chunk] for chunk in retrieved_chunks]

    # 3. Re-rank the Results using the Cross-Encoder
    print(f"Re-ranking {len(sentence_pairs)} candidates...")
    try:
        # The reranker.predict method takes pairs and outputs scores
        rerank_scores = reranker.predict(sentence_pairs)

        # Combine original data with re-rank scores
        scored_chunks = []
        for i, score in enumerate(rerank_scores):
             scored_chunks.append({
                 "document": retrieved_chunks[i],
                 "metadata": retrieved_metadatas[i],
                 "rerank_score": score
             })

        # Sort the chunks by the re-rank score in descending order
        scored_chunks.sort(key=lambda x: x["rerank_score"], reverse=True)

        print("Re-ranking complete.")

        # Debugging: Print top 3 re-ranked chunks
        print(f"\n--- Top {min(final_top_k, len(scored_chunks))} Re-Ranked Chunks ---")
        for i, chunk_info in enumerate(scored_chunks[:final_top_k], 1):
            print(f"\nRank {i} | Re-rank Score: {chunk_info['rerank_score']:.4f}")
            print(f"Source: {chunk_info['metadata'].get('source', 'unknown')}")
            # Print a preview, handling potential shorter chunks
            preview = chunk_info['document']
            print(f"Content Preview: {preview[:]}")
            print("-" * 60)

        # Return the top_k re-ranked results
        return scored_chunks[:final_top_k]

    except Exception as e:
        print(f"Error during re-ranking: {e}")
        return []


# ------------------ Optimized Prompt Construction ------------------
def build_prompt(context: str, query: str):
    """
    Constructs the prompt for the LLM, including the retrieved context.

    Args:
        context: The combined text from the top retrieved document chunks.
        query: The user's original query.

    Returns:
        A formatted prompt string.
    """
    return f"""
You are a highly knowledgeable assistant.

Context:
\"\"\"
{context}
\"\"\"

Based on the above context, answer the following question.

Question: {query}

Instructions:
- Do NOT mention that the answer is based on the provided context.
- If the context lacks the answer, give your best informed response (don't say "I don't know").

Answer:
"""

# ------------------ Main ------------------
if __name__ == "__main__":
    query = input("Enter your query: ")

    # Retrieve and re-rank. Get top 3 results after re-ranking for the prompt.
    top_results = retrieve_and_rerank(query, initial_fetch_k=50, final_top_k=3)

    if not top_results:
        print("\nCould not retrieve relevant context for the query.")
        response = llm_parsing2(build_prompt("No relevant information found in documents.", query)) # Use a fallback context
        print(f"\nResponse (fallback): {response}")
    else:
        # Combine the documents from the top results to create the context for the LLM
        context_for_llm = "\n\n---\n\n".join([item["document"] for item in top_results])

        prompt = build_prompt(context_for_llm, query)

        print("\n--- Sending Prompt to LLM ---")
        # print(prompt) # Uncomment to see the full prompt sent to the LLM
        response = llm_parsing2(prompt)

        print(f"\nResponse: {response}")