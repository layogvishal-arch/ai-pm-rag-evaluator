"""
pipeline.py — End-to-end RAG pipeline.

This module wires together the three stages of a Retrieval-Augmented
Generation system:

  1. Ingestion  — load documents from disk and split them into chunks.
  2. Retrieval  — embed the chunks, store them in ChromaDB, and find the
                  chunks most relevant to the user's question.
  3. Generation — send the retrieved chunks + question to Claude and return
                  the grounded answer.

Usage (from the project root):

    python -m src.pipeline "What are Vishal's key PM skills?"

Or import run() directly:

    from src.pipeline import run
    answer = run("What experience does Vishal have with LLMs?")
    print(answer)
"""

import sys

from src.config import CHROMA_PERSIST_DIR, DATA_DIR, TOP_K
from src.generation import generate_answer
from src.ingestion import ingest
from src.retrieval import build_vector_store, search


def run(
    question: str,
    data_dir: str = DATA_DIR,
    persist_dir: str = CHROMA_PERSIST_DIR,
    top_k: int = TOP_K,
) -> str:
    """Run the full RAG pipeline for a single question.

    Pipeline stages
    ---------------
    Stage 1 — Ingestion
        Load every PDF and DOCX from *data_dir*, extract plain text, and split
        each document into overlapping chunks.  Chunking keeps individual
        vectors semantically focused and within the embedding model's token
        window.

    Stage 2 — Retrieval
        Embed all chunks using a sentence-transformer model and insert them into
        a ChromaDB collection.  Then embed *question* with the same model and
        return the *top_k* nearest chunks by cosine similarity.  These chunks
        are the evidence the LLM will use when answering.

    Stage 3 — Generation
        Pass the retrieved chunks (as a numbered context block) together with
        the original question to Claude.  A system prompt instructs the model
        to stay grounded in the provided context and to admit when it lacks
        sufficient information rather than hallucinating.

    Args:
        question:    The user's natural-language question.
        data_dir:    Directory that holds source documents.
        persist_dir: Where ChromaDB stores its vector index.
                     Use ":memory:" to skip disk I/O (e.g. in unit tests).
        top_k:       Number of context chunks passed to the LLM.

    Returns:
        Claude's answer as a plain string.
    """
    print(f"\n{'='*60}")
    print(f"Question: {question}")
    print(f"{'='*60}\n")

    # ------------------------------------------------------------------ #
    # Stage 1: Ingestion                                                   #
    # ------------------------------------------------------------------ #
    chunks = ingest(data_dir=data_dir)

    # ------------------------------------------------------------------ #
    # Stage 2: Retrieval                                                   #
    # ------------------------------------------------------------------ #
    collection = build_vector_store(chunks, persist_dir=persist_dir)
    context_chunks = search(collection, query=question, top_k=top_k)

    # ------------------------------------------------------------------ #
    # Stage 3: Generation                                                  #
    # ------------------------------------------------------------------ #
    answer = generate_answer(question=question, context_chunks=context_chunks)

    print(f"\nAnswer:\n{answer}\n")
    return answer


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m src.pipeline \"<your question>\"")
        sys.exit(1)

    user_question = " ".join(sys.argv[1:])
    run(user_question)
