import os
import requests
import torch
from chromadb import PersistentClient
from util import llm_parsing,rerank_results
import chromadb.utils.embedding_functions as embedding_functions

# # Embedding
# # Load the model and tokenizer
# model_name = "distilbert-base-uncased"
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModel.from_pretrained(model_name)

# def get_embedding(text: str):
#     inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
#     with torch.no_grad():
#         outputs = model(**inputs)
#     embedding = outputs.last_hidden_state.mean(dim=1)
#     return embedding.squeeze()

#.................................Setup ChromaDB and Gemini..............................................

chromadb_folder = "./chromadb2"
collection_name = "pdf_documents"
google_api_key = "AIzaSyB7QnDF_mdps4fCH7klvrh1pAWBAjxzsq8"

client = PersistentClient(path=chromadb_folder)

google_ef=embedding_functions.GoogleGenerativeAiEmbeddingFunction(api_key=google_api_key)
collection=client.get_collection(name=collection_name, embedding_function=google_ef)

#.............................. Retrieve and Rerank Function...................................................

def retrieve_and_rerank(query, top_k=5):

    #...............................................approach 1..................................................
    query_embedding = google_ef([query])[0]  # Get the embedding for the query
    query_tensor=torch.tensor(query_embedding)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "embeddings"]
    )
    search_results = []
    for doc, metadata, embedding in zip(results["documents"][0], results["metadatas"][0], results["embeddings"][0]):
        search_results.append({
            "document": doc,
            "metadata": metadata,
            "embedding": embedding
        })

    reranked_results = rerank_results(search_results, query)
    top_chunk = reranked_results[0]  # Get the top chunk
    return top_chunk


    #...............................................approach 2..................................................
    
    
    
    
    # query_embedding = google_ef([query])[0]  # Generate embedding for the query

    # results = collection.query(
    #     query_embeddings=[query_embedding],
    #     n_results=top_k,
    #     include=["documents", "metadatas", "embeddings"]
    # )

    # # Assume the top-most result from Chroma is the most relevant
    # top_doc = results["documents"][0][0]
    # top_meta = results["metadatas"][0][0]
    # top_embedding = results["embeddings"][0][0]

    # top_chunk = {
    #     "document": top_doc,
    #     "metadata": top_meta,
    #     "embedding": top_embedding
    # }

    # return top_chunk

    #...............................................approach 3..................................................


    # query_embedding = google_ef([query])[0]  # Generate embedding for the query
    # query_tensor = torch.tensor(query_embedding)

    # results = collection.query(
    #     query_embeddings=[query_embedding],
    #     n_results=top_k,
    #     include=["documents", "metadatas", "embeddings"]
    # )

    # # Collect top-k search results from Chroma
    # search_results = []
    # for doc, metadata, embedding in zip(results["documents"][0], results["metadatas"][0], results["embeddings"][0]):
    #     search_results.append({
    #         "document": doc,
    #         "metadata": metadata,
    #         "embedding": embedding
    #     })

    # # Compute cosine similarity manually (rerank)
    # reranked = []
    # for doc in search_results:
    #     doc_embedding = torch.tensor(doc['embedding'])
    #     sim_score = torch.nn.functional.cosine_similarity(query_tensor, doc_embedding, dim=0).item()
    #     reranked.append((doc, sim_score))

    # # Sort by similarity score (descending)
    # reranked.sort(key=lambda x: x[1], reverse=True)

    # top_chunk = reranked[0]  # Highest similarity
    # return top_chunk


#..................................main function..................................................
query = input("Enter your query: ")
# top_chunk,score= retrieve_and_rerank(query)
top_chunk= retrieve_and_rerank(query)
print(f"Top Document: {top_chunk['metadata'].get('source', 'unknown')}")
# print(f"Top Document: {top_chunk['metadata'].get('source', 'unknown')}, Score: {score:.4f}")

print(f"Content: {top_chunk['document'][:700]}\n\n....................................")

# final_prompt="This is a RAG implemented chatbot consider below imformation as context frame the response based on query. "+top_chunk['document']+"\n\n Query: "+query+"\n\n"

final_prompt = f"""
You are an expert assistant.

Context:
\"\"\"
{top_chunk['document']}
\"\"\"

Now, based on the above context, answer the following question as precisely as possible:

Question: {query}

Instructions:
- Do NOT mention that the answer is based on the provided context.
- Do NOT speculate beyond the given information.
- If the answer is not in the context, answer it yourself as close as as possible.
- If the answer is not in the context, do NOT say "I don't know".

Answer:
"""

print(f"\nResponse: {llm_parsing(final_prompt)}")

# gemini_response =llm_parsing(top_chunk['document']+query)
# print(f"Response: {gemini_response}")