import PyPDF2
import torch
import os
import numpy as np
import uuid
import requests
from google import genai

def chunk_text(text, chunk_size=500, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

def rerank_results(search_results, query_embedding):
    reranked = []
    for doc in search_results:
        doc_embedding = torch.tensor(doc['embedding'])  # stored embeddings must be list of floats
        cosine_sim = torch.nn.functional.cosine_similarity(query_embedding, doc_embedding, dim=0)
        reranked.append((doc, cosine_sim.item()))
    reranked.sort(key=lambda x: x[1], reverse=True)
    return reranked


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