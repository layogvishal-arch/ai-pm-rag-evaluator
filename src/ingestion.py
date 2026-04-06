"""
ingestion.py — Document loading and chunking.

Responsibility: turn raw files on disk into a flat list of text chunks that
downstream steps can embed and index.

Why chunking matters for RAG:
  Embedding models have a fixed token window (typically 256-512 tokens).
  Feeding an entire document as one vector averages out the semantics and makes
  it hard to retrieve a *specific* passage.  Splitting into overlapping chunks
  keeps each vector focused while the overlap preserves context at boundaries.

Supported file types:
  - PDF  — via LangChain's PyPDFLoader (backed by pypdf)
  - DOCX — via LangChain's Docx2txtLoader (requires: pip install docx2txt)
"""

import os
from pathlib import Path
from typing import List

from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import DATA_DIR


def load_documents(data_dir: str = DATA_DIR) -> List[Document]:
    """Load every PDF and DOCX file found directly inside *data_dir*.

    Each file is opened with the appropriate LangChain loader.  PDFs are split
    per page (one Document per page) so that page metadata is preserved.  DOCX
    files are returned as a single Document whose page_content is the plain
    text of the whole file.

    Args:
        data_dir: Path to the folder that holds the source documents.
                  Defaults to the DATA_DIR setting in config.py.

    Returns:
        A list of LangChain Document objects.  Each Document has:
          - page_content: the extracted text
          - metadata: at minimum {"source": "<file path>"}

    Raises:
        FileNotFoundError: if *data_dir* does not exist.
    """
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    documents: List[Document] = []

    for file_path in sorted(data_path.iterdir()):
        suffix = file_path.suffix.lower()

        if suffix == ".pdf":
            loader = PyPDFLoader(str(file_path))
            documents.extend(loader.load())

        elif suffix in (".docx", ".doc"):
            loader = Docx2txtLoader(str(file_path))
            documents.extend(loader.load())

        # silently skip unrecognised file types (images, .DS_Store, etc.)

    print(f"[ingestion] Loaded {len(documents)} document page(s) from {data_dir}")
    return documents


def chunk_documents(
    documents: List[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> List[Document]:
    """Split a list of Documents into smaller, overlapping text chunks.

    Uses RecursiveCharacterTextSplitter, which tries to split on paragraph
    boundaries first, then sentence boundaries, then words — so chunks are as
    semantically coherent as possible given the size constraint.

    Why overlap?
      A chunk boundary might fall mid-sentence.  Repeating the last
      *chunk_overlap* characters in the next chunk ensures that no context is
      lost at the seam.

    Args:
        documents:    Documents returned by load_documents().
        chunk_size:   Target character count per chunk.  1 000 chars ≈ 200–250
                      tokens, which fits comfortably in most embedding models.
        chunk_overlap: Characters repeated between consecutive chunks.
                       20 % of chunk_size is a common starting point.

    Returns:
        A (longer) list of Documents where each item is one chunk.  Original
        metadata (e.g. source file path, page number) is preserved on every
        chunk so retrieved passages can be traced back to their origin.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        # Prefer splitting on double-newlines (paragraphs), then single
        # newlines, then sentences, then words, then characters as a last
        # resort.
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    print(f"[ingestion] Split into {len(chunks)} chunks "
          f"(size={chunk_size}, overlap={chunk_overlap})")
    return chunks


def ingest(data_dir: str = DATA_DIR) -> List[Document]:
    """Convenience wrapper: load files from disk and return ready-to-embed chunks.

    This is the main entry point used by pipeline.py.  It combines load and
    chunk into one call so callers don't need to know the two-step process.

    Args:
        data_dir: Folder containing source documents.

    Returns:
        List of text chunks as LangChain Document objects.
    """
    documents = load_documents(data_dir)
    return chunk_documents(documents)
