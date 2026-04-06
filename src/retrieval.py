"""
retrieval.py — Embedding & Vector Search for RAG

Uses OpenAI embeddings + LangChain's in-memory vector store.
No ChromaDB, no FAISS, no compiled dependencies.
Works on any Python version.

Cost: text-embedding-3-small costs ~$0.02 per 1M tokens.
Your entire knowledge base is maybe 5000 tokens = $0.0001.
"""

import numpy as np
from typing import List, Tuple
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()


# ──────────────────────────────────────────────
# STEP 1: Set up the embedding model
# ──────────────────────────────────────────────

def get_embedding_model():
    """
    OpenAI's text-embedding-3-small.
    Fast, cheap, high quality. The default choice for most RAG apps.
    """
    print("Setting up OpenAI embedding model...")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    print("Embedding model ready.")
    return embeddings


# ──────────────────────────────────────────────
# STEP 2: Simple vector store (no external DB)
# ──────────────────────────────────────────────

class SimpleVectorStore:
    """
    A vector store built with just numpy. No ChromaDB, no FAISS.

    WHY BUILD THIS YOURSELF:
    Every vector database (Pinecone, Weaviate, ChromaDB) does
    exactly what this class does — just with better indexing
    for millions of vectors. For your 7-28 chunks, numpy is
    faster and has zero dependency issues.

    PM INSIGHT: Understanding this helps you ask the right
    questions in technical discussions:
    - "What similarity metric are we using?" (cosine vs dot product)
    - "What's our recall at top-5?" (does the right chunk appear?)
    - "What's the latency at 100K chunks vs 1M?" (when to scale)
    """

    def __init__(self, embeddings):
        self.embeddings = embeddings
        self.documents = []       # Original text chunks
        self.vectors = []         # Their embeddings (numpy arrays)

    def add_documents(self, documents: List[Document]):
        """Embed all chunks and store them."""
        texts = [doc.page_content for doc in documents]
        print(f"  Embedding {len(texts)} chunks via OpenAI API...")

        # This is the API call — each text becomes a vector
        embedded = self.embeddings.embed_documents(texts)

        self.documents = documents
        self.vectors = np.array(embedded)
        print(f"  Done. Each chunk is now a {self.vectors.shape[1]}-dimensional vector.")

    def search(self, query: str, top_k: int = 3) -> List[Tuple[Document, float]]:
        """
        Search for the most similar chunks to a query.

        THE ENTIRE SEARCH IN 3 STEPS:
        1. Embed the query (same model as documents)
        2. Compute cosine similarity against every stored vector
        3. Return the top_k highest scoring chunks

        COSINE SIMILARITY:
        - 1.0 = identical meaning
        - 0.7+ = strong match
        - 0.5 = somewhat related
        - < 0.3 = probably not relevant
        """
        # Step 1: Embed the query
        query_vector = np.array(self.embeddings.embed_query(query))

        # Step 2: Cosine similarity = dot product of normalized vectors
        # Normalize all vectors to unit length
        norms = np.linalg.norm(self.vectors, axis=1, keepdims=True)
        normalized_vecs = self.vectors / norms
        query_norm = query_vector / np.linalg.norm(query_vector)

        # Dot product gives cosine similarity
        similarities = np.dot(normalized_vecs, query_norm)

        # Step 3: Get top_k indices (highest similarity first)
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for idx in top_indices:
            results.append((self.documents[idx], float(similarities[idx])))

        return results


# ──────────────────────────────────────────────
# STEP 3: Helper to print results nicely
# ──────────────────────────────────────────────

def print_search_results(query: str, results: List[Tuple[Document, float]]):
    """Pretty-print search results."""
    print(f"\n  QUERY: \"{query}\"")
    print(f"  {'-' * 50}")

    for i, (doc, score) in enumerate(results):
        strategy = doc.metadata.get("chunk_strategy", "unknown")
        section = doc.metadata.get("section_name", "")

        print(f"  Result {i + 1} | similarity: {score:.4f} | strategy: {strategy}")
        if section:
            print(f"  Section: {section}")
        preview = doc.page_content[:200].replace("\n", " ")
        print(f"  Preview: {preview}...")
        print()


# ──────────────────────────────────────────────
# STEP 4: Compare retrieval across all strategies
# ──────────────────────────────────────────────

def compare_retrieval(data_dir: str = "data"):
    """
    Run the same queries against all 3 chunking strategies.

    YOUR JOB: Look at results and observe:
    1. Which strategy's top result actually answers the question?
    2. Which has highest similarity for correct answers?
    3. Which handles unanswerable questions best?
    """
    from ingestion import load_documents, chunk_fixed_size, chunk_recursive, chunk_by_sections

    print("Loading documents...")
    docs = load_documents(data_dir)

    if not docs:
        print("No documents found in data/ folder!")
        return

    strategies = {
        "fixed_size": chunk_fixed_size(docs),
        "recursive": chunk_recursive(docs),
        "section_based": chunk_by_sections(docs),
    }

    # Load embedding model ONCE
    embeddings = get_embedding_model()

    test_queries = [
        # EASY: answer in one clear place
        "Where was Vishal born?",
        "What is Vishal's phone number?",

        # MEDIUM: needs the right section
        "What did Vishal do at HireQuotient?",
        "Who is Sejal?",

        # HARD: spans sections
        "What are Vishal's leadership experiences?",

        # TRICK: answer does NOT exist
        "What is Vishal's salary?",
    ]

    for strategy_name, chunks in strategies.items():
        print(f"\n{'#' * 60}")
        print(f"  STRATEGY: {strategy_name.upper()} ({len(chunks)} chunks)")
        print(f"{'#' * 60}")

        store = SimpleVectorStore(embeddings)
        store.add_documents(chunks)

        for query in test_queries:
            results = store.search(query, top_k=2)
            print_search_results(query, results)

    print("\n" + "=" * 60)
    print("WHAT TO OBSERVE:")
    print("=" * 60)
    print("""
  SIMILARITY SCORES (higher = better match):
  - > 0.7  = strong match, answer is likely in this chunk
  - 0.5-0.7 = related but may not contain the answer
  - < 0.5  = weak match, probably irrelevant

  KEY QUESTIONS:
  1. "Where was Vishal born?"
     → Does the top result contain "Singtam"?
     → Compare scores across strategies.

  2. "What is Vishal's salary?"
     → This DOESN'T EXIST in your docs.
     → Which strategy gives the LOWEST scores here?
       (Low score for wrong answers = good behavior)

  3. "What did Vishal do at HireQuotient?"
     → Fixed-size: probably a fragment
     → Section-based: full career section
     → Which is more useful for generating answers?
    """)


if __name__ == "__main__":
    compare_retrieval()
