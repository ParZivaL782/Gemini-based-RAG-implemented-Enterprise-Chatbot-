import os
import torch
from chromadb import PersistentClient
import voyageai
from util import llm_parsing2




# ------------------ Custom Voyage Embedding Function ------------------


from chromadb.api.types import EmbeddingFunction

class CustomVoyageEmbeddingFunction(EmbeddingFunction):
    def __init__(self, api_key, model="voyage-3"):
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
    


# ------------------ Configuration ------------------
chromadb_folder = "./chromadb_voyage"
collection_name = "pdf_documents"
voyage_api_key = "pa-7i_oLWDfuNdeLN4teyNCkza9z7ZpuTdidKMBC69hNSX"



# ------------------ Embedding + Chroma Setup ------------------
voyage_ef = CustomVoyageEmbeddingFunction(api_key=voyage_api_key, model="voyage-2")
client = PersistentClient(path=chromadb_folder)
collection = client.get_collection(name=collection_name, embedding_function=voyage_ef)

# ------------------ Retrieve + Rerank ------------------
def retrieve_and_rerank(query, top_k=20):

    #........................................................Approach 1..........................................................


    query_embedding = voyage_ef([query])[0]
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "embeddings"]
    )

    # Score results manually using cosine similarity
    query_tensor = torch.tensor(query_embedding)
    scored_chunks = []
    for doc, metadata, embedding in zip(results["documents"][0], results["metadatas"][0], results["embeddings"][0]):
        score = torch.nn.functional.cosine_similarity(query_tensor, torch.tensor(embedding), dim=0).item()
        scored_chunks.append(({
            "document": doc,
            "metadata": metadata,
            "embedding": embedding
        }, score))

    scored_chunks.sort(key=lambda x: x[1], reverse=True)

    # Debugging: Print top 3 chunks
    print("\n--- Top 3 Ranked Chunks ---")
    for i, (chunk, score) in enumerate(scored_chunks[:3], 1):
        print(f"\nRank {i} | Score: {score:.4f}")
        print(f"Source: {chunk['metadata'].get('source', 'unknown')}")
        print(f"Content Preview: {chunk['document'][:]}...")
        print("-" * 60)

    return scored_chunks  # Return the top-ranked chunk

    #........................................................Approach 2..........................................................
    # query_embedding = jinaai_ef([query])[0]
    # results = collection.query(
    # query_embeddings=[query_embedding],
    # n_results=top_k,
    # include=["documents", "metadatas", "embeddings"]
    # )

    # top_chunk = rerank_with_jina_reranker(query, results)
    # return top_chunk

    #........................................................Approach 3..........................................................
    # # Embed the query using Jina
    # query_embedding = jinaai_ef([query])[0]

    # # Query ChromaDB using the embedding
    # results = collection.query(
    #     query_embeddings=[query_embedding],
    #     n_results=top_k,
    #     include=["documents", "metadatas"]
    # )

    # # Format and return top-k results
    # top_chunks = []
    # print(f"\n--- Top {top_k} Chunks (Native Ranking) ---")
    # for i in range(len(results["documents"][0])):
    #     doc = results["documents"][0][i]
    #     metadata = results["metadatas"][0][i]
    #     print(f"\nRank {i+1}")
    #     print(f"Source: {metadata.get('source', 'unknown')}")
    #     print(f"Content Preview: {doc[:300]}...")
    #     print("-" * 60)
    #     top_chunks.append({
    #         "document": doc,
    #         "metadata": metadata
    #     })

    # return top_chunks

# ------------------ Optimized Prompt Construction ------------------
def build_prompt(context_chunks, query):
    context = "\n\n---\n\n".join(context_chunks)
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
    top_chunks = retrieve_and_rerank(query)

    # Extract top 3 documents
    top_contexts = [top_chunks[i][0]['document'] for i in range(min(3, len(top_chunks)))]

    prompt = build_prompt(top_contexts, query)
    response = llm_parsing2(prompt)

    print(f"\nResponse: {response}")
# if __name__ == "__main__":
#     query = input("Enter your query: ")
#     top_chunks = retrieve_and_rerank(query, top_k=3)

#     top_contexts = [chunk['document'] for chunk in top_chunks]
#     prompt = build_prompt(top_contexts, query)
#     response = llm_parsing2(prompt)

#     print(f"\nResponse: {response}")

