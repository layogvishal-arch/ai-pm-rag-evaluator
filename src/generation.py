"""
generation.py — LLM Answer Generation for RAG

This file does one thing:
Takes retrieved chunks + user question → sends to Claude → gets answer.

The quality of the answer depends on:
1. Did retrieval find the right chunks? (retrieval.py's job)
2. Is the prompt well-structured? (this file's job)
3. Is the model good enough? (model selection in config.py)

PM INSIGHT: You can't fix bad retrieval with a better prompt.
But you CAN ruin good retrieval with a bad prompt. The prompt
template is a product decision — it controls tone, accuracy,
and how the app handles uncertainty.
"""

import anthropic
from typing import List, Tuple
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()


# ──────────────────────────────────────────────
# STEP 1: The prompt template
# ──────────────────────────────────────────────

SYSTEM_PROMPT = """You are a helpful assistant that answers questions about Vishal Goyal 
based ONLY on the provided context. 

RULES:
1. Only use information from the context below to answer.
2. If the context does not contain enough information to answer, say: 
   "I don't have enough information to answer that based on the available documents."
3. Always cite which section or part of the context you used.
4. Be specific — use names, numbers, and dates from the context.
5. If the question is ambiguous, state your interpretation before answering.
6. Keep answers concise but complete.

Do NOT make up information. Do NOT use your general knowledge. 
Only use what's in the context."""

def build_prompt(query: str, retrieved_chunks: List[Tuple[Document, float]]) -> str:
    """
    Build the full prompt with retrieved context.
    
    PROMPT STRUCTURE:
    1. System message (rules for the model)
    2. Retrieved chunks with metadata (the "cheat sheet")
    3. User question
    4. Instruction to cite sources
    
    PM INSIGHT: Every line in the system prompt is a product decision.
    - "Only use context" → prevents hallucination but may refuse valid questions
    - "Say I don't know" → builds user trust but may frustrate users
    - "Cite sources" → adds transparency but makes answers longer
    
    These are trade-offs you'll tune based on user feedback.
    """
    # Format each chunk with its metadata
    context_parts = []
    for i, (doc, score) in enumerate(retrieved_chunks):
        source = doc.metadata.get("source", "unknown")
        section = doc.metadata.get("section_name", "")
        strategy = doc.metadata.get("chunk_strategy", "unknown")
        
        header = f"[Source {i+1}: {source}"
        if section:
            header += f" | {section}"
        header += f" | relevance: {score:.4f}]"
        
        context_parts.append(f"{header}\n{doc.page_content}")
    
    context = "\n\n---\n\n".join(context_parts)
    
    user_message = f"""CONTEXT:
{context}

QUESTION: {query}

Answer the question using only the context above. Cite which source(s) you used."""
    
    return user_message


# ──────────────────────────────────────────────
# STEP 2: Call Claude
# ──────────────────────────────────────────────

def generate_answer(query: str, retrieved_chunks: List[Tuple[Document, float]]) -> dict:
    """
    Send the prompt to Claude and get an answer.
    
    Returns a dict with:
    - answer: the generated text
    - model: which model was used
    - input_tokens: how many tokens the prompt used
    - output_tokens: how many tokens the answer used
    - total_cost_estimate: rough cost of this query
    
    PM INSIGHT: Token tracking is essential for AI products.
    Every query costs money. You need to know:
    - Average cost per query (for pricing your product)
    - Token distribution (is context too large? answers too long?)
    - Cost trends (are users asking longer questions over time?)
    """
    client = anthropic.Anthropic()
    
    user_message = build_prompt(query, retrieved_chunks)
    
    # Count chunks and their total length for logging
    total_context_chars = sum(len(doc.page_content) for doc, _ in retrieved_chunks)
    
    print(f"  Sending to Claude...")
    print(f"  Context: {len(retrieved_chunks)} chunks, {total_context_chars} chars")
    
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[
            {"role": "user", "content": user_message}
        ]
    )
    
    answer = response.content[0].text
    input_tokens = response.usage.input_tokens
    output_tokens = response.usage.output_tokens
    
    # Rough cost estimate for Claude Haiku
    # Haiku: $0.25 per 1M input tokens, $1.25 per 1M output tokens
    cost = (input_tokens * 0.25 / 1_000_000) + (output_tokens * 1.25 / 1_000_000)
    
    return {
        "answer": answer,
        "model": "claude-haiku-4-5-20251001",
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost_estimate": cost,
        "chunks_used": len(retrieved_chunks),
        "context_chars": total_context_chars,
    }


# ──────────────────────────────────────────────
# STEP 3: Pretty print the full result
# ──────────────────────────────────────────────

def print_result(query: str, result: dict):
    """Print the answer with all metadata."""
    print(f"\n{'=' * 60}")
    print(f"Q: {query}")
    print(f"{'=' * 60}")
    print(f"\nA: {result['answer']}")
    print(f"\n--- Stats ---")
    print(f"  Model: {result['model']}")
    print(f"  Input tokens: {result['input_tokens']}")
    print(f"  Output tokens: {result['output_tokens']}")
    print(f"  Estimated cost: ${result['cost_estimate']:.6f}")
    print(f"  Chunks used: {result['chunks_used']}")
    print(f"  Context size: {result['context_chars']} chars")


# ──────────────────────────────────────────────
# STEP 4: Run the full pipeline
# ──────────────────────────────────────────────

if __name__ == "__main__":
    from ingestion import load_documents, chunk_recursive
    from retrieval import get_embedding_model, SimpleVectorStore
    
    # Load and chunk documents (using recursive — our best all-rounder)
    print("Loading documents...")
    docs = load_documents("data")
    chunks = chunk_recursive(docs)
    
    # Create vector store
    print("\nCreating vector store...")
    embeddings = get_embedding_model()
    store = SimpleVectorStore(embeddings)
    store.add_documents(chunks)
    
    # Test queries
    test_queries = [
        "Where was Vishal born and where did he grow up?",
        "What did Vishal achieve at HireQuotient?",
        "Who is Sejal and how did Vishal meet her?",
        "What is Vishal's current role?",
        "What is Vishal's salary?",  # Unanswerable — should say "I don't know"
        "What programming languages does Vishal know?",
    ]
    
    print("\n\n" + "#" * 60)
    print("  RAG PIPELINE — FULL ANSWERS")
    print("#" * 60)
    
    total_cost = 0
    
    for query in test_queries:
        # Retrieve relevant chunks
        retrieved = store.search(query, top_k=3)
        
        # Generate answer
        result = generate_answer(query, retrieved)
        print_result(query, result)
        total_cost += result["cost_estimate"]
    
    print(f"\n\n{'=' * 60}")
    print(f"TOTAL COST for {len(test_queries)} queries: ${total_cost:.6f}")
    print(f"Average cost per query: ${total_cost/len(test_queries):.6f}")
    print(f"{'=' * 60}")
    print("""
  WHAT TO EVALUATE:
  
  1. Are the answers accurate? Compare against what you know.
  2. Did it refuse to answer the salary question?
  3. Are the citations correct?
  4. How much did it cost? Project this to 1000 queries/day.
  5. Are any answers too long or too short?
  
  These become your eval criteria for Days 3-4.
    """)
