"""
parent_child.py — Parent-Child Chunking + Retrieval

Drop-in module that replaces the standard chunking + search flow.
Small children for precise search, large parents for LLM context.

Usage:
    from parent_child import build_parent_child_store, search_parent_child
    
    store, parents = build_parent_child_store(documents, embeddings)
    results = search_parent_child(store, parents, query, top_k=5)
    # results contains PARENT chunks (full context) matched via CHILD search
"""

import numpy as np
from typing import List, Tuple
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


def create_parent_child_chunks(documents: List[Document], parent_size: int = 1500, child_size: int = 300, child_overlap: int = 50):
    """
    Split documents into parents and children.
    
    Returns: (parents_list, children_list)
    Each child has metadata["parent_index"] pointing to its parent.
    """
    # Step 1: Create large parent chunks
    parent_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ". ", " ", ""],
        chunk_size=parent_size,
        chunk_overlap=200,
        length_function=len,
    )
    parents = parent_splitter.split_documents(documents)
    
    for i, parent in enumerate(parents):
        parent.metadata["parent_index"] = i
        parent.metadata["chunk_strategy"] = "parent"
    
    # Step 2: Split each parent into small children
    child_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ". ", " ", ""],
        chunk_size=child_size,
        chunk_overlap=child_overlap,
        length_function=len,
    )
    
    children = []
    for parent_idx, parent in enumerate(parents):
        parent_doc = [Document(page_content=parent.page_content, metadata=parent.metadata.copy())]
        parent_children = child_splitter.split_documents(parent_doc)
        
        for child in parent_children:
            child.metadata["chunk_strategy"] = "child"
            child.metadata["parent_index"] = parent_idx
            child.metadata["child_index"] = len(children)
            children.append(child)
    
    print(f"  Created {len(parents)} parents, {len(children)} children")
    print(f"  Avg parent: {sum(len(p.page_content) for p in parents) // len(parents)} chars")
    print(f"  Avg child: {sum(len(c.page_content) for c in children) // len(children)} chars")
    
    return parents, children


def build_parent_child_store(documents, embeddings, parent_size=1500, child_size=300):
    """
    Build a vector store from children, keep parents for context.
    
    Returns: (store, parents)
    """
    from retrieval import SimpleVectorStore
    
    parents, children = create_parent_child_chunks(documents, parent_size, child_size)
    
    store = SimpleVectorStore(embeddings)
    store.add_documents(children)
    
    return store, parents


def search_parent_child(store, parents: List[Document], query: str, top_k: int = 5, return_parents: int = 3) -> List[Tuple[Document, float]]:
    """
    Search children, return their parents (deduplicated).
    
    1. Search children for top_k matches
    2. Map each child to its parent
    3. Deduplicate parents (multiple children may share a parent)
    4. Return top return_parents unique parents with best child score
    """
    # Search against children
    child_results = store.search(query, top_k=top_k)
    
    # Map to parents, keeping best score per parent
    parent_scores = {}
    for child_doc, score in child_results:
        parent_idx = child_doc.metadata["parent_index"]
        if parent_idx not in parent_scores or score > parent_scores[parent_idx]:
            parent_scores[parent_idx] = score
    
    # Sort parents by best child score
    sorted_parents = sorted(parent_scores.items(), key=lambda x: x[1], reverse=True)
    
    # Return parent documents with scores
    results = []
    for parent_idx, score in sorted_parents[:return_parents]:
        results.append((parents[parent_idx], score))
    
    return results


# ──────────────────────────────────────────────
# Quick test
# ──────────────────────────────────────────────

if __name__ == "__main__":
    from ingestion import load_documents
    from retrieval import get_embedding_model
    
    docs = load_documents("data")
    embeddings = get_embedding_model()
    
    store, parents = build_parent_child_store(docs, embeddings)
    
    test_queries = [
        "What is Vishal's current role?",
        "What programming languages does Vishal know?",
        "What did Vishal achieve at HireQuotient?",
        "What are Vishal's leadership experiences?",
    ]
    
    for query in test_queries:
        results = search_parent_child(store, parents, query, top_k=5, return_parents=3)
        print(f"\nQuery: \"{query}\"")
        for i, (doc, score) in enumerate(results):
            has_answer_hint = ""
            if "Tech Mahindra" in doc.page_content:
                has_answer_hint = " [has Tech Mahindra]"
            elif "Python" in doc.page_content and "Jupyter" in doc.page_content:
                has_answer_hint = " [has Skills]"
            elif "$1.2M" in doc.page_content:
                has_answer_hint = " [has HQ metrics]"
            print(f"  #{i+1} score={score:.4f}{has_answer_hint} | {doc.page_content[:100]}...")
