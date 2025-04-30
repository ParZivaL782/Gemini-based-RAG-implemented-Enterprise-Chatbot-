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
    query_embedding = google_ef([query])[0]  # Get the embedding for the query

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
    query_tensor=torch.tensor(query_embedding)

    reranked_results = rerank_results(search_results, query_tensor)
    top_chunk = reranked_results[1]  # Get the top chunk
    return top_chunk


#..................................main function..................................................
query = input("Enter your query: ")
top_chunk,score = retrieve_and_rerank(query)

print(f"Top Document: {top_chunk['metadata'].get('source', 'unknown')}, Score: {score:.4f}")
print(f"Content: {top_chunk['document'][:300]}\n\n....................................")

final_prompt=top_chunk['document']+"\n\nbased on data above answer following: "+query+"\ndon't print 'based on provided text' "
print(f"\nResponse: {llm_parsing(final_prompt)}")

# gemini_response =llm_parsing(top_chunk['document']+query)
# print(f"Response: {gemini_response}")