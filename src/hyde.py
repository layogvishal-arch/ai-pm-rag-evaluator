"""
hyde.py — Hypothetical Document Embedding (HyDE)

At indexing time: for each chunk, use Claude to generate
questions that chunk can answer + a summary. Embed those
alongside the original chunk.

At query time: user's question matches against generated
questions (same vocabulary space) instead of raw document text.

WHY THIS WORKS:
"What is Vishal's current role?" matches poorly against
"Joined as Senior Inbound Product Manager at ServiceNow through Tech Mahindra"
(score: 0.27)

But it matches strongly against the generated question
"What is Vishal's current role?" (score: ~0.95)

Cost: one Haiku call per chunk at indexing time.
For 14 chunks: ~$0.003 total (one-time cost).
"""

import numpy as np
import json
import os
import anthropic
from typing import List, Tuple
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()


# ──────────────────────────────────────────────
# STEP 1: Generate questions for each chunk
# ──────────────────────────────────────────────

def generate_chunk_questions(chunk_text: str) -> dict:
    """
    Use Claude to generate questions and a summary for a chunk.
    
    Returns:
    {
        "questions": ["question 1", "question 2", ...],
        "summary": "one-line summary of what this chunk covers"
    }
    """
    client = anthropic.Anthropic()
    
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=300,
        system="You generate search queries for a RAG system. Be concise. Return ONLY valid JSON.",
        messages=[{
            "role": "user",
            "content": f"""Read this text chunk and generate:
1. Five questions that someone might ask that this chunk would answer. Include variations in phrasing (formal, casual, keyword-style).
2. A one-line summary of what this chunk is about.

Return ONLY a JSON object like:
{{"questions": ["q1", "q2", "q3", "q4", "q5"], "summary": "one line summary"}}

Text chunk:
{chunk_text[:1500]}"""
        }]
    )
    
    try:
        result_text = response.content[0].text.strip()
        result_text = result_text.replace("```json", "").replace("```", "").strip()
        return json.loads(result_text)
    except (json.JSONDecodeError, IndexError):
        return {"questions": [], "summary": ""}


# ──────────────────────────────────────────────
# STEP 2: Build the HyDE index
# ──────────────────────────────────────────────

class HyDEVectorStore:
    """
    Multi-vector store: each chunk has multiple embeddings.
    
    For each chunk, we store embeddings of:
    - The original chunk text
    - Each generated question
    - The generated summary
    
    At search time, the query is compared against ALL embeddings.
    If any embedding matches well, the chunk surfaces.
    
    This gives each chunk more "surface area" to be discovered.
    """
    
    def __init__(self, embeddings):
        self.embeddings = embeddings
        self.chunks = []           # Original chunk documents
        self.all_vectors = []      # All embeddings (multiple per chunk)
        self.vector_to_chunk = []  # Maps each vector index to its chunk index
        self.hyde_metadata = []    # Generated questions/summaries per chunk
    
    def build_index(self, chunks: List[Document], cache_path: str = "eval/hyde_cache.json"):
        """
        Generate questions for each chunk and embed everything.
        
        Uses a cache file so you don't re-generate questions on every run.
        """
        self.chunks = chunks
        
        # Check cache
        cached_metadata = {}
        if os.path.exists(cache_path):
            with open(cache_path, "r") as f:
                cached_metadata = json.load(f)
            print(f"  Loaded HyDE cache ({len(cached_metadata)} chunks)")
        
        # Generate questions for each chunk
        print(f"  Generating questions for {len(chunks)} chunks...")
        all_texts_to_embed = []
        self.vector_to_chunk = []
        self.hyde_metadata = []
        
        for i, chunk in enumerate(chunks):
            chunk_key = chunk.page_content[:100]  # Use first 100 chars as cache key
            
            if chunk_key in cached_metadata:
                meta = cached_metadata[chunk_key]
            else:
                print(f"    Generating for chunk {i+1}/{len(chunks)}...")
                meta = generate_chunk_questions(chunk.page_content)
                cached_metadata[chunk_key] = meta
            
            self.hyde_metadata.append(meta)
            
            # Add original chunk text
            all_texts_to_embed.append(chunk.page_content)
            self.vector_to_chunk.append(i)
            
            # Add generated questions
            for q in meta.get("questions", []):
                all_texts_to_embed.append(q)
                self.vector_to_chunk.append(i)
            
            # Add summary
            summary = meta.get("summary", "")
            if summary:
                all_texts_to_embed.append(summary)
                self.vector_to_chunk.append(i)
        
        # Save cache
        os.makedirs(os.path.dirname(cache_path) if os.path.dirname(cache_path) else ".", exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump(cached_metadata, f, indent=2)
        
        # Embed everything
        print(f"  Embedding {len(all_texts_to_embed)} vectors ({len(chunks)} chunks + {len(all_texts_to_embed) - len(chunks)} generated texts)...")
        embedded = self.embeddings.embed_documents(all_texts_to_embed)
        self.all_vectors = np.array(embedded)
        print(f"  HyDE index ready. {self.all_vectors.shape[0]} total vectors.")
    
    def search(self, query: str, top_k: int = 5) -> List[Tuple[Document, float]]:
        """
        Search across all vectors (original + generated questions + summaries).
        Return unique chunks ranked by their best matching vector.
        """
        # Embed query
        query_vector = np.array(self.embeddings.embed_query(query))
        
        # Cosine similarity against all vectors
        norms = np.linalg.norm(self.all_vectors, axis=1, keepdims=True)
        normalized = self.all_vectors / norms
        query_norm = query_vector / np.linalg.norm(query_vector)
        similarities = np.dot(normalized, query_norm)
        
        # Find best score per chunk (a chunk might match via question or via original text)
        chunk_best_scores = {}
        for vec_idx, score in enumerate(similarities):
            chunk_idx = self.vector_to_chunk[vec_idx]
            if chunk_idx not in chunk_best_scores or score > chunk_best_scores[chunk_idx]:
                chunk_best_scores[chunk_idx] = float(score)
        
        # Sort by score
        sorted_chunks = sorted(chunk_best_scores.items(), key=lambda x: x[1], reverse=True)
        
        results = []
        for chunk_idx, score in sorted_chunks[:top_k]:
            results.append((self.chunks[chunk_idx], score))
        
        return results
    
    def search_detailed(self, query: str, top_k: int = 5) -> List[dict]:
        """
        Search with details about which vector matched (original text vs generated question).
        Useful for debugging.
        """
        query_vector = np.array(self.embeddings.embed_query(query))
        
        norms = np.linalg.norm(self.all_vectors, axis=1, keepdims=True)
        normalized = self.all_vectors / norms
        query_norm = query_vector / np.linalg.norm(query_vector)
        similarities = np.dot(normalized, query_norm)
        
        # For each chunk, find which vector matched best
        chunk_details = {}
        
        # Track which text each vector corresponds to
        vec_idx = 0
        for chunk_idx, chunk in enumerate(self.chunks):
            meta = self.hyde_metadata[chunk_idx]
            texts = [chunk.page_content] + meta.get("questions", [])
            summary = meta.get("summary", "")
            if summary:
                texts.append(summary)
            
            for text in texts:
                score = float(similarities[vec_idx])
                if chunk_idx not in chunk_details or score > chunk_details[chunk_idx]["best_score"]:
                    is_question = text != chunk.page_content and text != summary
                    is_summary = text == summary
                    match_type = "question" if is_question else ("summary" if is_summary else "original")
                    chunk_details[chunk_idx] = {
                        "best_score": score,
                        "matched_via": match_type,
                        "matched_text": text[:100],
                    }
                vec_idx += 1
        
        sorted_chunks = sorted(chunk_details.items(), key=lambda x: x[1]["best_score"], reverse=True)
        
        results = []
        for chunk_idx, details in sorted_chunks[:top_k]:
            results.append({
                "chunk": self.chunks[chunk_idx],
                "score": details["best_score"],
                "matched_via": details["matched_via"],
                "matched_text": details["matched_text"],
            })
        
        return results


# ──────────────────────────────────────────────
# Quick test on the failing queries
# ──────────────────────────────────────────────

if __name__ == "__main__":
    from ingestion import load_documents, chunk_recursive
    from retrieval import get_embedding_model
    
    docs = load_documents("data")
    chunks = chunk_recursive(docs, chunk_size=1200, chunk_overlap=300)
    
    embeddings = get_embedding_model()
    
    # Build HyDE index
    print("\nBuilding HyDE index...")
    store = HyDEVectorStore(embeddings)
    store.build_index(chunks)
    
    # Test the previously failing queries
    test_queries = [
        "What is Vishal's current role?",
        "What programming languages does Vishal know?",
        "What did Vishal achieve at HireQuotient?",
        "What product tools does Vishal use?",
        "What are Vishal's leadership experiences?",
        "What failures or setbacks has Vishal faced?",
    ]
    
    for query in test_queries:
        results = store.search_detailed(query, top_k=3)
        print(f"\nQuery: \"{query}\"")
        for i, r in enumerate(results):
            chunk_preview = r["chunk"].page_content[:80]
            print(f"  #{i+1} score={r['score']:.4f} matched_via={r['matched_via']}")
            print(f"     matched: \"{r['matched_text']}\"")
            print(f"     chunk: \"{chunk_preview}...\"")
