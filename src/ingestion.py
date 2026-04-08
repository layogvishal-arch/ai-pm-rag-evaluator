"""
ingestion.py — Document Loading & Chunking for RAG

WHY THIS FILE MATTERS (PM perspective):
Chunking is the single most impactful decision in a RAG pipeline.
Bad chunks = bad retrieval = bad answers, no matter how good your LLM is.

This file implements 3 strategies so you can compare them.
Think of this as an A/B test for your data layer.
"""

import os
from typing import List
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
)
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()

# ──────────────────────────────────────────────
# STEP 1: Load documents from the data/ folder
# ──────────────────────────────────────────────

def load_documents(data_dir: str = "data") -> List[Document]:
    """
    Load all PDF and DOCX files from the data directory.
    
    Returns a list of LangChain Document objects.
    Each Document has:
      - page_content: the actual text
      - metadata: source file, page number, etc.
    
    PM NOTE: Metadata matters! It lets you trace back which 
    document an answer came from. This is critical for:
    - Source attribution in your product
    - Debugging bad answers
    - User trust ("here's where I found this")
    """
    documents = []
    
    for filename in os.listdir(data_dir):
        filepath = os.path.join(data_dir, filename)
        
        if filename.endswith(".pdf"):
            loader = PyPDFLoader(filepath)
            docs = loader.load()
            # Add source filename to metadata
            for doc in docs:
                doc.metadata["source"] = filename
                doc.metadata["file_type"] = "pdf"
            documents.extend(docs)
            print(f"  Loaded PDF: {filename} ({len(docs)} pages)")
            
        elif filename.endswith(".docx"):
            loader = Docx2txtLoader(filepath)
            docs = loader.load()
            for doc in docs:
                doc.metadata["source"] = filename
                doc.metadata["file_type"] = "docx"
            documents.extend(docs)
            print(f"  Loaded DOCX: {filename} ({len(docs)} sections)")
    
    print(f"\nTotal documents loaded: {len(documents)}")
    return documents


# ──────────────────────────────────────────────
# STEP 2: Three chunking strategies
# ──────────────────────────────────────────────

def chunk_fixed_size(documents: List[Document], chunk_size: int = 500, chunk_overlap: int = 50) -> List[Document]:
    """
    STRATEGY 1: Fixed-size chunking (the naive approach)
    
    HOW IT WORKS:
    - Splits text every `chunk_size` characters
    - Doesn't care about sentences, paragraphs, or meaning
    - Overlap prevents losing context at boundaries
    
    WHEN TO USE:
    - Quick prototyping, baseline comparison
    - When documents have no clear structure
    
    PM TRADE-OFF:
    - Pro: Fast, predictable chunk count, easy to estimate costs
    - Con: Cuts mid-sentence, loses context, worst retrieval quality
    - Con: "Vishal was born in" might be in one chunk, "Singtam" in another
    """
    splitter = CharacterTextSplitter(
        separator="",           # Split on any character (truly fixed-size)
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    
    chunks = splitter.split_documents(documents)
    
    # Add chunking metadata — this is important for evaluation later
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_strategy"] = "fixed_size"
        chunk.metadata["chunk_index"] = i
        chunk.metadata["chunk_size_setting"] = chunk_size
        chunk.metadata["actual_length"] = len(chunk.page_content)
    
    return chunks


def chunk_recursive(documents: List[Document], chunk_size: int = 1200, chunk_overlap: int = 300) -> List[Document]:
    """
    STRATEGY 2: Recursive character splitting (the smart default)
    
    HOW IT WORKS:
    - Tries to split by paragraphs first (double newline)
    - If chunk is still too big, splits by single newline
    - Then by sentence (period + space)
    - Then by space (word boundary)
    - Last resort: splits by character
    
    This is called "recursive" because it works through the separator
    list recursively, trying the most natural break first.
    
    WHEN TO USE:
    - This is your go-to default for most RAG applications
    - Works well with documents that have natural paragraph structure
    
    PM TRADE-OFF:
    - Pro: Respects document structure, keeps sentences together
    - Pro: Chunks make sense when read standalone
    - Con: Chunk sizes vary (some might be very small)
    - Con: Still doesn't understand meaning — just structure
    """
    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ". ", " ", ""],  # Try each in order
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        add_start_index=True,   # Tracks position in original doc
    )
    
    chunks = splitter.split_documents(documents)
    
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_strategy"] = "recursive"
        chunk.metadata["chunk_index"] = i
        chunk.metadata["chunk_size_setting"] = chunk_size
        chunk.metadata["actual_length"] = len(chunk.page_content)
    
    return chunks


def chunk_by_sections(documents: List[Document]) -> List[Document]:
    """
    STRATEGY 3: Section-based / semantic chunking
    
    HOW IT WORKS:
    - Splits on section headers (like "Section 1:", "Section 2:")
    - Each section becomes one chunk
    - Keeps all related information together
    
    WHY THIS IS DIFFERENT:
    - Strategies 1 and 2 are generic — they work on any document
    - This strategy is CUSTOM to your document's structure
    - This is what you'd do in production: design your chunking
      around how your data is actually organized
    
    PM INSIGHT:
    - This is the highest-quality approach because you DESIGNED
      the document structure (in the docx I created) specifically
      for good retrieval
    - In enterprise RAG products, you often need custom chunking
      per document type (e.g., contracts vs. manuals vs. emails)
    - The PM decision: is it worth the engineering effort to build
      custom chunkers? (Almost always yes for production)
    
    PM TRADE-OFF:
    - Pro: Best retrieval quality — each chunk is self-contained
    - Pro: Metadata is rich (section name, topic)
    - Con: Requires knowing your document structure upfront
    - Con: Doesn't generalize to unknown document types
    """
    chunks = []
    
    # Section markers — these match the document structure I created
    section_markers = [
        "Section 1: Personal Identity",
        "Section 2: Education Timeline",
        "Section 3: Career Timeline",
        "Section 4: Key Relationships & Friendships",
        "Section 5: Interests, Likes & Hobbies",
        "Section 6: Technical & Product Skills",
        "Section 7: Defining Moments & Lessons",
    ]
    
    for doc in documents:
        text = doc.page_content
        
        for i, marker in enumerate(section_markers):
            # Find where this section starts
            start_idx = text.find(marker)
            if start_idx == -1:
                continue
            
            # Find where the next section starts (or end of doc)
            if i + 1 < len(section_markers):
                next_marker = section_markers[i + 1]
                end_idx = text.find(next_marker)
                if end_idx == -1:
                    end_idx = len(text)
            else:
                end_idx = len(text)
            
            section_text = text[start_idx:end_idx].strip()
            
            if section_text:  # Don't create empty chunks
                chunk = Document(
                    page_content=section_text,
                    metadata={
                        **doc.metadata,
                        "chunk_strategy": "section_based",
                        "chunk_index": i,
                        "section_name": marker,
                        "actual_length": len(section_text),
                    }
                )
                chunks.append(chunk)
    
    # FALLBACK: If section-based splitting didn't work
    # (e.g., document doesn't have our expected headers),
    # fall back to recursive chunking
    if not chunks:
        print("  Warning: No sections found. Falling back to recursive chunking.")
        return chunk_recursive(documents)
    
    return chunks


# ──────────────────────────────────────────────
# STEP 3: Compare all strategies (this is your experiment)
# ──────────────────────────────────────────────

def compare_chunking_strategies(documents: List[Document]) -> dict:
    """
    Run all 3 chunking strategies on the same documents and print comparison.
    
    This is what a PM should do: run the experiment, look at the data,
    then make the trade-off decision.
    """
    print("\n" + "=" * 60)
    print("CHUNKING STRATEGY COMPARISON")
    print("=" * 60)
    
    strategies = {
        "fixed_size": chunk_fixed_size(documents),
        "recursive": chunk_recursive(documents),
        "section_based": chunk_by_sections(documents),
    }
    
    for name, chunks in strategies.items():
        lengths = [len(c.page_content) for c in chunks]
        avg_len = sum(lengths) / len(lengths) if lengths else 0
        
        print(f"\n--- {name.upper()} ---")
        print(f"  Total chunks: {len(chunks)}")
        print(f"  Avg chunk length: {avg_len:.0f} chars")
        print(f"  Min chunk length: {min(lengths) if lengths else 0} chars")
        print(f"  Max chunk length: {max(lengths) if lengths else 0} chars")
        print(f"\n  First chunk preview (first 200 chars):")
        if chunks:
            print(f"  '{chunks[0].page_content[:200]}...'")
    
    return strategies


# ──────────────────────────────────────────────
# STEP 4: Run it! 
# ──────────────────────────────────────────────

if __name__ == "__main__":
    """
    Run this file directly to see the comparison:
    
        python src/ingestion.py
    
    WHAT TO LOOK FOR when you read the output:
    
    1. Fixed-size chunks: Do any of them cut mid-sentence? 
       (They will. That's the point.)
    
    2. Recursive chunks: Do they respect paragraph boundaries?
       (They should. Compare the previews.)
    
    3. Section-based chunks: Does each chunk cover one topic?
       (Yes — because we designed the document that way.)
    
    4. Which strategy gives you chunks that make sense as 
       STANDALONE pieces of text? That's the best strategy.
       A good chunk should be understandable WITHOUT reading
       the chunks before or after it.
    
    THIS IS YOUR FIRST EVAL — you're evaluating chunk quality
    by inspection. Days 3-4 will formalize this into metrics.
    """
    print("Loading documents...")
    docs = load_documents("data")
    
    if not docs:
        print("\nNo documents found in data/ folder!")
        print("Make sure you have your .docx or .pdf files in the data/ directory.")
    else:
        strategies = compare_chunking_strategies(docs)
        
        # Deep dive: print ALL chunks from each strategy so you can
        # visually inspect quality
        print("\n\n" + "=" * 60)
        print("DETAILED CHUNK INSPECTION")
        print("=" * 60)
        
        for name, chunks in strategies.items():
            print(f"\n\n{'=' * 40}")
            print(f"STRATEGY: {name}")
            print(f"{'=' * 40}")
            for i, chunk in enumerate(chunks):
                print(f"\n--- Chunk {i+1} of {len(chunks)} ---")
                print(f"Length: {len(chunk.page_content)} chars")
                print(f"Metadata: {chunk.metadata}")
                print(f"Content:\n{chunk.page_content[:500]}")
                if len(chunk.page_content) > 500:
                    print(f"  ...(truncated, {len(chunk.page_content) - 500} more chars)")
                print()
