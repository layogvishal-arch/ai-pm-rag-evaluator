"""
retrieval.py — Semantic Vector Search for RAG

Uses OpenAI embeddings + numpy-based cosine similarity.
Semantic-only approach — won the Day 1 comparison overall.

Day 2 learning: Hybrid search (semantic + keyword) was tested
but introduced regressions on queries that were already working.
Keeping semantic-only as primary retriever, using reranker in
generation.py to fix edge cases.
"""

import numpy as np
from typing import List, Tuple
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()


def get_embedding_model():
    """OpenAI's text-embedding-3-small. Fast, cheap, high quality."""
    print("Setting up OpenAI embedding model...")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    print("Embedding model ready.")
    return embeddings


class SimpleVectorStore:
    """
    Vector store using numpy cosine similarity.
    No external DB dependencies.
    """

    def __init__(self, embeddings):
        self.embeddings = embeddings
        self.documents = []
        self.vectors = []

    def add_documents(self, documents: List[Document]):
        """Embed all chunks and store them."""
        texts = [doc.page_content for doc in documents]
        print(f"  Embedding {len(texts)} chunks via OpenAI API...")
        embedded = self.embeddings.embed_documents(texts)
        self.documents = documents
        self.vectors = np.array(embedded)
        print(f"  Done. Each chunk is now a {self.vectors.shape[1]}-dimensional vector.")

    def search(self, query: str, top_k: int = 3) -> List[Tuple[Document, float]]:
        """
        Search by cosine similarity.
        Returns (document, similarity_score) tuples, highest first.
        """
        query_vector = np.array(self.embeddings.embed_query(query))
        norms = np.linalg.norm(self.vectors, axis=1, keepdims=True)
        normalized_vecs = self.vectors / norms
        query_norm = query_vector / np.linalg.norm(query_vector)
        similarities = np.dot(normalized_vecs, query_norm)
        
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            results.append((self.documents[idx], float(similarities[idx])))
        
        return results
