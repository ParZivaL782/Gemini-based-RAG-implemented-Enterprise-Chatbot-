import os
import torch
from transformers import AutoTokenizer, AutoModel
from chromadb import PersistentClient

# Load model and tokenizer
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

def get_embedding(text: str):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    embedding = outputs.last_hidden_state.mean(dim=1)
    return embedding.squeeze()

# Setup Chroma (new style)
chroma_db_path = "./chromadb"
client = PersistentClient(path=chroma_db_path)

# Ensure the collection name matches the one used during storing
collection_name = "pdf_documents"  # This should match the name used in extract_and_store.py
collection = client.get_collection(name=collection_name)


def rerank_results(search_results, query_embedding):
    reranked = []
    for doc in search_results:
        doc_embedding = torch.tensor(doc['embedding'])  # stored embeddings must be list of floats
        cosine_sim = torch.nn.functional.cosine_similarity(query_embedding, doc_embedding, dim=0)
        reranked.append((doc, cosine_sim.item()))
    reranked.sort(key=lambda x: x[1], reverse=True)
    return reranked

def retrieve_and_rerank(query, top_k=5):
    query_embedding = get_embedding(query)
    query_embedding_list = query_embedding.tolist()

    results = collection.query(
        query_embeddings=[query_embedding_list],
        n_results=top_k,
        include=["documents", "metadatas", "embeddings"]
    )

    # print(f"Raw search results: {results}")

    search_results = []
    for doc, metadata, embedding in zip(results["documents"][0], results["metadatas"][0], results["embeddings"][0]):
        search_results.append({
            "document": doc,
            "metadata": metadata,
            "embedding": embedding
        })

    reranked_results = rerank_results(search_results, query_embedding)
    return reranked_results

# Input query
query = input("Enter your query: ")
retrieved_documents = retrieve_and_rerank(query)


if retrieved_documents:
    for (doc_info, score) in retrieved_documents:
        source = doc_info["metadata"].get("source", "unknown")
        print(f"Document: {source}, Score: {score:.4f}")
        print(f"Content: {doc_info['document'][:300]}...\n")
else:
    print("No relevant documents found.")
