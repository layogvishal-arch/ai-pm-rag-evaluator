"""
generation.py — RAG Generation with Reranking + Multi-turn

WHAT'S NEW (Day 2):
- Reranker: retrieves top 5 chunks, Claude picks the best 3
- Multi-turn: follow-up questions use conversation history
- Better error handling for edge cases
- Token and cost tracking per session

ARCHITECTURE DECISION:
We're keeping semantic-only retrieval (it won yesterday's comparison)
and adding a reranker to fix the cases it misses.
Hybrid search introduced regressions — a lesson in itself.
"""

import anthropic
from typing import List, Tuple, Optional
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()

MODEL = "claude-haiku-4-5-20251001"

# ──────────────────────────────────────────────
# SYSTEM PROMPT
# ──────────────────────────────────────────────

SYSTEM_PROMPT = """You are a helpful assistant that answers questions about Vishal Goyal 
based ONLY on the provided context. 

RULES:
1. Only use information from the context below to answer.
2. If the context does not contain enough information to answer, say: 
   "I don't have enough information to answer that based on the available documents."
3. Always cite which section or source you used.
4. Be specific — use names, numbers, and dates from the context.
5. If the question is ambiguous, state your interpretation before answering.
6. Keep answers concise but complete.
7. If this is a follow-up question, use the conversation history for context
   but still only cite from the provided documents.

Do NOT make up information. Do NOT use your general knowledge."""


# ──────────────────────────────────────────────
# RERANKER
# ──────────────────────────────────────────────

def rerank_chunks(query: str, chunks: List[Tuple[Document, float]], top_k: int = 3) -> List[Tuple[Document, float]]:
    """
    Use Claude to re-rank retrieved chunks by relevance.
    
    Takes top 5 from retrieval, asks Claude to pick the best 3.
    
    WHY THIS WORKS:
    Vector search finds "related" content. But related != answers the question.
    The reranker asks: "Does this chunk ACTUALLY answer THIS question?"
    
    Example from yesterday:
    - "What is Vishal's current role?" retrieved Deloitte and Presidency
    - Tech Mahindra chunk existed but scored lower
    - A reranker would see "Sept 2025 - Present" and rank it #1
    
    PM TRADE-OFF:
    - Adds ~$0.0002 per query (one extra Haiku call)
    - Adds ~500ms latency
    - Significantly improves answer relevance
    """
    if len(chunks) <= top_k:
        return chunks
    
    client = anthropic.Anthropic()
    
    # Format chunks for the reranker
    chunks_text = ""
    for i, (doc, score) in enumerate(chunks):
        preview = doc.page_content[:400]
        chunks_text += f"\n[Chunk {i+1}] (similarity: {score:.4f})\n{preview}\n"
    
    try:
        response = client.messages.create(
            model=MODEL,
            max_tokens=100,
            system="You rank text chunks by relevance to a query. Return ONLY a comma-separated list of chunk numbers, most relevant first. Example: 3,1,5,2,4",
            messages=[{
                "role": "user",
                "content": f"Query: {query}\n{chunks_text}\nRank ALL chunks by relevance (most relevant first). Numbers only:"
            }]
        )
        
        ranking_text = response.content[0].text.strip()
        ranking = [int(x.strip()) - 1 for x in ranking_text.split(",") if x.strip().isdigit()]
        
        reranked = []
        for idx in ranking[:top_k]:
            if 0 <= idx < len(chunks):
                reranked.append(chunks[idx])
        
        # Fill remaining slots if ranking was incomplete
        for chunk in chunks:
            if len(reranked) >= top_k:
                break
            if chunk not in reranked:
                reranked.append(chunk)
        
        print(f"  Reranker reordered: {[x+1 for x in ranking[:top_k]]}")
        return reranked[:top_k]
        
    except Exception as e:
        print(f"  Reranker failed ({e}), using original order")
        return chunks[:top_k]


# ──────────────────────────────────────────────
# PROMPT BUILDER
# ──────────────────────────────────────────────

def build_prompt(query: str, chunks: List[Tuple[Document, float]], conversation_history: Optional[List[dict]] = None) -> str:
    """Build the user message with context and optional history."""
    
    context_parts = []
    for i, (doc, score) in enumerate(chunks):
        source = doc.metadata.get("source", "unknown")
        section = doc.metadata.get("section_name", "")
        
        header = f"[Source {i+1}: {source}"
        if section:
            header += f" | {section}"
        header += f" | relevance: {score:.4f}]"
        
        context_parts.append(f"{header}\n{doc.page_content}")
    
    context = "\n\n---\n\n".join(context_parts)
    
    user_message = f"""CONTEXT:
{context}

QUESTION: {query}

Answer using only the context above. Cite which source(s) you used."""
    
    return user_message


# ──────────────────────────────────────────────
# ANSWER GENERATION
# ──────────────────────────────────────────────

def generate_answer(
    query: str,
    retrieved_chunks: List[Tuple[Document, float]],
    conversation_history: Optional[List[dict]] = None,
    use_reranker: bool = True,
) -> dict:
    """
    Generate an answer with optional reranking and conversation history.
    
    Pipeline:
    1. (Optional) Rerank retrieved chunks
    2. Build prompt with context
    3. Send to Claude with conversation history
    4. Return answer + metadata
    """
    client = anthropic.Anthropic()
    
    # Step 1: Rerank if enabled
    if use_reranker and len(retrieved_chunks) > 3:
        print(f"  Reranking {len(retrieved_chunks)} chunks...")
        chunks_to_use = rerank_chunks(query, retrieved_chunks, top_k=3)
    else:
        chunks_to_use = retrieved_chunks[:3]
    
    # Step 2: Build prompt
    user_message = build_prompt(query, chunks_to_use, conversation_history)
    
    # Step 3: Build messages array with history
    messages = []
    if conversation_history:
        messages.extend(conversation_history)
    messages.append({"role": "user", "content": user_message})
    
    total_context_chars = sum(len(doc.page_content) for doc, _ in chunks_to_use)
    print(f"  Sending to Claude ({len(chunks_to_use)} chunks, {total_context_chars} chars)...")
    
    # Step 4: Call Claude
    response = client.messages.create(
        model=MODEL,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=messages,
    )
    
    answer = response.content[0].text
    input_tokens = response.usage.input_tokens
    output_tokens = response.usage.output_tokens
    
    # Haiku pricing
    cost = (input_tokens * 0.25 / 1_000_000) + (output_tokens * 1.25 / 1_000_000)
    
    return {
        "answer": answer,
        "model": MODEL,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost_estimate": cost,
        "chunks_used": len(chunks_to_use),
        "context_chars": total_context_chars,
        "reranked": use_reranker,
    }


def print_result(query: str, result: dict):
    """Print answer with stats."""
    print(f"\n{'=' * 60}")
    print(f"Q: {query}")
    print(f"{'=' * 60}")
    print(f"\nA: {result['answer']}")
    print(f"\n--- Stats ---")
    print(f"  Model: {result['model']}")
    print(f"  Tokens: {result['input_tokens']} in / {result['output_tokens']} out")
    print(f"  Cost: ${result['cost_estimate']:.6f}")
    print(f"  Reranked: {result['reranked']}")


# ──────────────────────────────────────────────
# INTERACTIVE CHAT MODE
# ──────────────────────────────────────────────

def chat_mode(store):
    """
    Interactive chat with conversation history.
    
    This demonstrates multi-turn RAG:
    - Follow-up questions use prior context
    - "Tell me more about that" works because history is preserved
    
    PM INSIGHT: Multi-turn is essential for product UX.
    Users don't ask perfect standalone questions.
    They ask "What did Vishal do?" then "Tell me more about the AI stuff."
    """
    print("\n" + "#" * 60)
    print("  INTERACTIVE RAG CHAT")
    print("  Type your questions. Type 'quit' to exit.")
    print("  Follow-up questions use conversation history.")
    print("#" * 60)
    
    conversation_history = []
    total_cost = 0
    
    while True:
        query = input("\nYou: ").strip()
        if query.lower() in ("quit", "exit", "q"):
            print(f"\nSession cost: ${total_cost:.6f}")
            break
        if not query:
            continue
        
        # Retrieve top 5 chunks (reranker will pick best 3)
        retrieved = store.search(query, top_k=5)
        
        # Generate answer with history
        result = generate_answer(
            query, retrieved,
            conversation_history=conversation_history,
            use_reranker=True,
        )
        
        print(f"\nAssistant: {result['answer']}")
        print(f"  [tokens: {result['input_tokens']}+{result['output_tokens']} | cost: ${result['cost_estimate']:.6f}]")
        
        # Add to conversation history
        conversation_history.append({"role": "user", "content": query})
        conversation_history.append({"role": "assistant", "content": result["answer"]})
        total_cost += result["cost_estimate"]
        
        # Keep history manageable (last 6 turns)
        if len(conversation_history) > 12:
            conversation_history = conversation_history[-12:]


# ──────────────────────────────────────────────
# MAIN: Compare with and without reranker
# ──────────────────────────────────────────────

if __name__ == "__main__":
    from ingestion import load_documents, chunk_recursive
    from retrieval import get_embedding_model, SimpleVectorStore
    
    print("Loading documents...")
    docs = load_documents("data")
    chunks = chunk_recursive(docs)
    
    print("\nCreating vector store...")
    embeddings = get_embedding_model()
    store = SimpleVectorStore(embeddings)
    store.add_documents(chunks)
    
    # Test: queries that failed yesterday
    failing_queries = [
        "What is Vishal's current role?",
        "What programming languages does Vishal know?",
        "What is Vishal's salary?",
    ]
    
    print("\n\n" + "#" * 60)
    print("  RERANKER COMPARISON: Without vs With")
    print("#" * 60)
    
    for query in failing_queries:
        retrieved = store.search(query, top_k=5)
        
        # Without reranker
        result_no_rerank = generate_answer(query, retrieved, use_reranker=False)
        print(f"\n{'=' * 60}")
        print(f"Q: {query}")
        print(f"{'=' * 60}")
        print(f"\nWithout reranker:")
        print(f"  A: {result_no_rerank['answer'][:300]}...")
        
        # With reranker
        result_rerank = generate_answer(query, retrieved, use_reranker=True)
        print(f"\nWith reranker:")
        print(f"  A: {result_rerank['answer'][:300]}...")
    
    # Interactive mode
    print("\n\nStarting interactive chat mode...")
    print("Try asking follow-up questions!")
    print("Example: 'What did Vishal do at HireQuotient?' then 'Tell me more about the AI work'")
    chat_mode(store)
