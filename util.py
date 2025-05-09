import PyPDF2
import torch
import os
import numpy as np
import uuid
import requests
from google import genai
# from transformers import AutoTokenizer, AutoModelForSequenceClassification
# import torch.nn.functional as F


# # Load the Jina reranker
# reranker_model = AutoModelForSequenceClassification.from_pretrained("jinaai/jina-reranker-m0")
# reranker_tokenizer = AutoTokenizer.from_pretrained("jinaai/jina-reranker-m0")
# def rerank_with_jina_reranker(query, results, top_n=3):
#     reranked = []

#     for doc, metadata, embedding in zip(results["documents"][0], results["metadatas"][0], results["embeddings"][0]):
#         # Combine query and document
#         inputs = reranker_tokenizer(query, doc, return_tensors="pt", truncation=True, padding=True)
#         outputs = reranker_model(**inputs)
#         score = F.softmax(outputs.logits, dim=1)[0][1].item()  # relevance score

#         reranked.append(({
#             "document": doc,
#             "metadata": metadata,
#             "embedding": embedding
#         }, score))

#     reranked.sort(key=lambda x: x[1], reverse=True)

    # # Print top 3
    # print("\n--- Top 3 Ranked by Jina Reranker ---")
    # for i, (chunk, score) in enumerate(reranked[:3], 1):
    #     print(f"\nRank {i} | Score: {score:.4f}")
    #     print(f"Source: {chunk['metadata'].get('source', 'unknown')}")
    #     print(f"Content Preview: {chunk['document'][:300]}...")
    #     print("-" * 60)

    # return reranked[0][0]  # top chunk


def chunk_text(text, chunk_size=1500, overlap=300):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

def rerank_results(search_results, query):
    
    #................................................approach 1..................................................
    
    # reranked = []
    # for doc in search_results:
    #     doc_embedding = torch.tensor(doc['embedding'])  # stored embeddings must be list of floats
    #     cosine_sim = torch.nn.functional.cosine_similarity(query_embedding, doc_embedding, dim=0)
    #     reranked.append((doc, cosine_sim.item()))
    # reranked.sort(key=lambda x: x[1], reverse=True)
    # return reranked


    #................................................approach 2..................................................
    from google import genai
    client = genai.Client(api_key="AIzaSyBhKa9_v_DekKezcKbsVXe9hVx_s54iESc")
    reranked = []

    for result in search_results:
        document = result["document"]
        prompt = (
            f"Score from 1 to 10 how well this passage answers the question.\n\n"
            f"Passage:\n{document}\n\n"
            f"Question:\n{query}\n\n"
            f"Only return a number."
        )

        try:
            response = client.models.generate_content(
                model="gemini-1.5-flash",
                contents=prompt
            )
            score = float(response.text.strip())
            reranked.append((result, score))
        except Exception as e:
            print(f"Failed to score with Gemini: {e}")
            reranked.append((result, 0.0))

    reranked.sort(key=lambda x: x[1], reverse=True)
    return reranked[0]

    # print(f"Raw search results: {results}")


    

def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
    return text


def llm_parsing(text):
    from google import genai
    client = genai.Client(api_key="AIzaSyBhKa9_v_DekKezcKbsVXe9hVx_s54iESc")
    response = client.models.generate_content(
        model="gemini-1.5-flash",
        contents=text
        )
    return response.text
import requests

def llm_parsing2(text):
    url = "https://api.jina.ai/v1/chat/completions"
    headers = {
        "Authorization": "Bearer jina_9ea0c05a3ece41beb434af45dc058c31QdxlAA7aty8e62u6x8V5SkgVU9Ez",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "jina-chat-bge-large-en",  # or "gpt-4", "llama2-13b-chat", etc.
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": text}
        ]
    }

    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()

    return response.json()["choices"][0]["message"]["content"]

# def llm_parsing2(query_text, context_text=""):
#     import requests

#     url = "https://api.jina.ai/v1/deepsearch/query"
#     headers = {
#         "Authorization": "Bearer jina_1df2b997be01407b9d29fb1260a8d539crtpuV_d4pLnqv6RnHcmfpH7-sXy",
#         "Content-Type": "application/json"
#     }

#     payload = {
#         "query": query_text,
#         "context": context_text,
#         "model": "jina-search-bge-base-en",  # Optional: Use your preferred reranker model
#         "top_k": 3
#     }

#     response = requests.post(url, headers=headers, json=payload)
#     response.raise_for_status()

#     return response.json()["results"][0]["answer"]
