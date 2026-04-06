"""
retrieval.py — Vector store creation and semantic search.

Responsibility: embed text chunks into a ChromaDB collection and retrieve the
most relevant chunks for a given user query.

Why ChromaDB?
  It is a lightweight, file-backed vector database that needs no external
  server.  The same client API works for in-memory stores (great for tests) and
  persistent stores (great for production).

Why sentence-transformers?
  They produce dense vector representations specifically optimised for semantic
  similarity.  "all-MiniLM-L6-v2" is small, fast, and good enough for most
  document Q&A use-cases.  Swap EMBEDDING_MODEL in config.py to upgrade.
"""

import uuid
from typing import List

import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from langchain_core.documents import Document

from src.config import CHROMA_PERSIST_DIR, EMBEDDING_MODEL, TOP_K

# Name of the ChromaDB collection that holds document chunks.
COLLECTION_NAME = "rag_documents"


def _make_client(persist_dir: str) -> chromadb.Client:
    """Create a ChromaDB client, persisting to disk (or in-memory for tests).

    Args:
        persist_dir: Directory path for on-disk storage.
                     Pass ":memory:" to use an ephemeral in-memory store.

    Returns:
        A configured chromadb.Client instance.
    """
    if persist_dir == ":memory:":
        return chromadb.Client()

    return chromadb.Client(
        Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory=persist_dir,
            anonymized_telemetry=False,
        )
    )


def build_vector_store(
    chunks: List[Document],
    persist_dir: str = CHROMA_PERSIST_DIR,
    embedding_model: str = EMBEDDING_MODEL,
) -> chromadb.Collection:
    """Embed *chunks* and store them in a ChromaDB collection.

    How it works:
      1. Each chunk's text is embedded by a sentence-transformer model into a
         high-dimensional float vector.
      2. The vectors (plus the original text and metadata) are stored in a
         ChromaDB collection so they can be queried by similarity later.

    This function always creates a fresh collection, dropping any existing one
    with the same name first.  For incremental updates, swap create_collection
    for get_or_create_collection and skip the delete step.

    Args:
        chunks:          Text chunks from ingestion.chunk_documents().
        persist_dir:     Where ChromaDB writes its files.  ":memory:" for tests.
        embedding_model: Sentence-Transformers model name used for encoding.

    Returns:
        The populated chromadb.Collection, ready for querying.
    """
    client = _make_client(persist_dir)

    # Always start fresh so stale vectors don't pollute retrieval results.
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass  # collection didn't exist yet — that's fine

    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=embedding_model
    )
    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},  # cosine similarity for text
    )

    # ChromaDB requires string IDs.  Use deterministic UUIDs derived from the
    # chunk index so re-runs produce the same IDs.
    ids = [str(uuid.uuid5(uuid.NAMESPACE_DNS, f"chunk-{i}")) for i in range(len(chunks))]
    texts = [chunk.page_content for chunk in chunks]
    metadatas = [chunk.metadata for chunk in chunks]

    # Add in one batch — ChromaDB handles its own batching internally.
    collection.add(documents=texts, ids=ids, metadatas=metadatas)

    print(f"[retrieval] Indexed {len(chunks)} chunks into ChromaDB "
          f"(model={embedding_model})")
    return collection


def search(
    collection: chromadb.Collection,
    query: str,
    top_k: int = TOP_K,
) -> List[str]:
    """Return the *top_k* most semantically relevant chunks for *query*.

    ChromaDB encodes *query* with the same embedding function used during
    indexing and returns the nearest neighbours by cosine similarity.

    Args:
        collection: A populated ChromaDB collection from build_vector_store().
        query:      The user's question in plain text.
        top_k:      Number of chunks to return.  More chunks give the LLM more
                    context but also increase token costs.

    Returns:
        A list of chunk texts ordered from most to least relevant.
    """
    results = collection.query(
        query_texts=[query],
        n_results=min(top_k, collection.count()),  # can't ask for more than exist
    )
    # results["documents"] is a list-of-lists (one list per query).
    texts: List[str] = results["documents"][0]
    print(f"[retrieval] Retrieved {len(texts)} chunks for query: {query!r}")
    return texts
