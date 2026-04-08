"""
eval_runner.py — Automated RAG Evaluation Framework

Runs every question in the golden dataset through the RAG pipeline,
then scores each answer on multiple metrics.

METRICS:
1. Key fact recall: did the answer contain the expected key facts?
2. Hallucination check: for unanswerable questions, did it refuse?
3. Retrieval precision: did the right chunks get retrieved?
4. LLM-as-judge: use Claude to grade answer quality (0-5 scale)

PM INSIGHT: This is the framework that would have caught the
hybrid search regressions on Day 2 automatically. Run this
after every pipeline change. If any score drops, investigate
before shipping.
"""

import json
import time
import anthropic
from typing import List, Tuple
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()


# ──────────────────────────────────────────────
# METRIC 1: Key Fact Recall
# ──────────────────────────────────────────────

def score_key_fact_recall(answer: str, key_facts: list) -> dict:
    """
    Check what percentage of expected key facts appear in the answer.
    
    Simple but effective: if the expected answer contains "Singtam"
    and "Sikkim", check if both appear in the generated answer.
    
    Score: 0.0 to 1.0 (percentage of facts found)
    
    PM INSIGHT: This is your most important automated metric.
    It's not perfect (the model might paraphrase), but it catches
    the worst failures: completely wrong answers that miss all facts.
    """
    if not key_facts:
        return {"score": 1.0, "found": [], "missed": [], "total": 0}
    
    answer_lower = answer.lower()
    found = []
    missed = []
    
    for fact in key_facts:
        if fact.lower() in answer_lower:
            found.append(fact)
        else:
            missed.append(fact)
    
    score = len(found) / len(key_facts) if key_facts else 1.0
    
    return {
        "score": score,
        "found": found,
        "missed": missed,
        "total": len(key_facts),
    }


# ──────────────────────────────────────────────
# METRIC 2: Hallucination Detection
# ──────────────────────────────────────────────

def score_hallucination(answer: str, is_answerable: bool) -> dict:
    """
    Check if the system correctly handles unanswerable questions.
    
    For unanswerable questions:
    - PASS if answer contains refusal language ("don't have", "not available", etc.)
    - FAIL if answer provides a confident (hallucinated) response
    
    For answerable questions:
    - PASS if answer does NOT contain unnecessary refusal
    - FAIL if answer refuses when it should have answered
    
    PM INSIGHT: Hallucination rate is the #1 metric investors
    and enterprise buyers ask about. "What's your hallucination rate?"
    is the AI equivalent of "What's your uptime?"
    """
    refusal_phrases = [
        "don't have enough information",
        "not available",
        "cannot answer",
        "no information",
        "not mentioned",
        "not specified",
        "not provided",
        "does not contain",
        "i don't know",
        "not answerable",
        "cannot determine",
        "unable to answer",
        "not enough information",
    ]
    
    answer_lower = answer.lower()
    has_refusal = any(phrase in answer_lower for phrase in refusal_phrases)
    
    if not is_answerable:
        # Should refuse
        return {
            "score": 1.0 if has_refusal else 0.0,
            "correct_behavior": "refuse",
            "actual_behavior": "refused" if has_refusal else "hallucinated",
        }
    else:
        # Should answer
        return {
            "score": 1.0 if not has_refusal else 0.5,  # 0.5 not 0.0 — refusing is better than hallucinating
            "correct_behavior": "answer",
            "actual_behavior": "answered" if not has_refusal else "refused",
        }


# ──────────────────────────────────────────────
# METRIC 3: Retrieval Precision
# ──────────────────────────────────────────────

def score_retrieval(retrieved_chunks: List[Tuple[Document, float]], expected_sources: list) -> dict:
    """
    Check if the expected source sections were retrieved.
    
    Looks at whether chunks from the expected sections appear
    in the top K retrieved results.
    
    PM INSIGHT: This metric tells you if the problem is in
    retrieval or generation. If retrieval precision is high
    but answers are wrong, fix the prompt. If retrieval is low,
    fix chunking or search.
    """
    if not expected_sources:
        return {"score": 1.0, "found_sources": [], "missed_sources": [], "detail": "No expected sources (unanswerable)"}
    
    # Check chunk content for expected section references
    retrieved_text = " ".join([doc.page_content for doc, _ in retrieved_chunks])
    retrieved_lower = retrieved_text.lower()
    
    found = []
    missed = []
    
    for source in expected_sources:
        # Check if key terms from the source section appear
        source_terms = source.lower().replace("section", "").strip()
        if source_terms in retrieved_lower or any(word in retrieved_lower for word in source_terms.split(":") if len(word.strip()) > 3):
            found.append(source)
        else:
            missed.append(source)
    
    score = len(found) / len(expected_sources) if expected_sources else 1.0
    
    return {
        "score": score,
        "found_sources": found,
        "missed_sources": missed,
    }


# ──────────────────────────────────────────────
# METRIC 4: LLM-as-Judge
# ──────────────────────────────────────────────

def score_with_llm_judge(question: str, expected_answer: str, actual_answer: str, is_answerable: bool) -> dict:
    """
    Use Claude to grade answer quality on a 1-5 scale.
    
    WHY LLM-AS-JUDGE:
    Key fact recall is brittle — "Bengaluru" vs "Bangalore" would
    fail even though both are correct. An LLM judge understands
    paraphrasing, partial correctness, and nuance.
    
    SCALE:
    5 = Perfect answer, all facts correct, well-cited
    4 = Good answer, minor missing details
    3 = Acceptable, has the main point but missing context
    2 = Partial, some correct info but significant gaps
    1 = Wrong or hallucinated
    
    PM TRADE-OFF:
    - Cost: ~$0.0003 per judgment (Haiku)
    - For 30 test cases: ~$0.01 total
    - Worth it for the nuance it captures
    """
    client = anthropic.Anthropic()
    
    if not is_answerable:
        judge_prompt = f"""You are evaluating a RAG system's response to an UNANSWERABLE question.

Question: {question}
The correct behavior is to REFUSE to answer or say "I don't know."

System's response: {actual_answer}

Score from 1-5:
5 = Correctly refused, clearly stated info is not available
4 = Refused but with unnecessary hedging
3 = Partially refused but also provided some speculative info
2 = Provided an answer but with caveats
1 = Confidently provided a hallucinated answer

Respond with ONLY a JSON object: {{"score": N, "reason": "brief explanation"}}"""
    else:
        judge_prompt = f"""You are evaluating a RAG system's answer quality.

Question: {question}
Expected answer: {expected_answer}
System's response: {actual_answer}

Score from 1-5:
5 = Perfect, contains all key facts from expected answer, well-structured
4 = Good, has most key facts, minor omissions
3 = Acceptable, has the main point but missing important details
2 = Partial, some correct info but significant gaps or errors
1 = Wrong, hallucinated, or refused when it should have answered

Respond with ONLY a JSON object: {{"score": N, "reason": "brief explanation"}}"""

    try:
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=150,
            messages=[{"role": "user", "content": judge_prompt}]
        )
        
        result_text = response.content[0].text.strip()
        # Parse JSON from response
        result_text = result_text.replace("```json", "").replace("```", "").strip()
        result = json.loads(result_text)
        
        return {
            "score": result.get("score", 0) / 5.0,  # Normalize to 0-1
            "raw_score": result.get("score", 0),
            "reason": result.get("reason", ""),
        }
    except Exception as e:
        return {"score": 0, "raw_score": 0, "reason": f"Judge error: {e}"}


# ──────────────────────────────────────────────
# FULL EVALUATION RUNNER
# ──────────────────────────────────────────────

def run_evaluation(use_llm_judge: bool = True):
    """
    Run the complete evaluation pipeline.
    
    For each test case:
    1. Run the question through the RAG pipeline
    2. Score with all metrics
    3. Aggregate results
    4. Print a report
    """
    from eval_dataset import get_dataset
    from ingestion import load_documents, chunk_recursive
    from retrieval import get_embedding_model, SimpleVectorStore
    from generation import generate_answer
    
    # Setup pipeline
    print("Setting up RAG pipeline...")
    docs = load_documents("data")
    chunks = chunk_recursive(docs, chunk_size=1200, chunk_overlap=300)
    
    embeddings = get_embedding_model()
    store = SimpleVectorStore(embeddings)
    store.add_documents(chunks)
    
    dataset = get_dataset()
    print(f"\nRunning {len(dataset)} test cases...\n")
    
    results = []
    
    for i, test_case in enumerate(dataset):
        qid = test_case["id"]
        question = test_case["question"]
        
        print(f"[{i+1}/{len(dataset)}] {qid}: {question[:50]}...")
        
        # Run RAG pipeline
        retrieved = store.search(question, top_k=3)
        rag_result = generate_answer(question, retrieved)
        answer = rag_result["answer"]
        
        # Score with all metrics
        fact_recall = score_key_fact_recall(answer, test_case["key_facts"])
        hallucination = score_hallucination(answer, test_case["is_answerable"])
        retrieval = score_retrieval(retrieved, test_case["expected_sources"])
        
        llm_judge = {"score": 0, "raw_score": 0, "reason": "skipped"}
        if use_llm_judge:
            llm_judge = score_with_llm_judge(
                question, test_case["expected_answer"], answer, test_case["is_answerable"]
            )
            time.sleep(0.5)  # Rate limit protection
        
        result = {
            "id": qid,
            "question": question,
            "category": test_case["category"],
            "difficulty": test_case["difficulty"],
            "is_answerable": test_case["is_answerable"],
            "answer": answer,
            "scores": {
                "key_fact_recall": fact_recall["score"],
                "hallucination": hallucination["score"],
                "retrieval_precision": retrieval["score"],
                "llm_judge": llm_judge["score"],
            },
            "details": {
                "fact_recall": fact_recall,
                "hallucination": hallucination,
                "retrieval": retrieval,
                "llm_judge": llm_judge,
            },
            "cost": rag_result["cost_estimate"],
        }
        
        results.append(result)
    
    # Print report
    print_report(results)
    
    # Save results
    with open("eval/eval_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nDetailed results saved to eval/eval_results.json")
    
    return results


def print_report(results: list):
    """Print a formatted evaluation report."""
    print(f"\n\n{'=' * 70}")
    print(f"  EVALUATION REPORT")
    print(f"{'=' * 70}")
    
    # Overall scores
    metrics = ["key_fact_recall", "hallucination", "retrieval_precision", "llm_judge"]
    
    print(f"\n  OVERALL SCORES:")
    for metric in metrics:
        scores = [r["scores"][metric] for r in results]
        avg = sum(scores) / len(scores)
        print(f"    {metric:25s}: {avg:.2%}")
    
    total_cost = sum(r["cost"] for r in results)
    print(f"\n    Total eval cost: ${total_cost:.4f}")
    
    # By category
    print(f"\n  BY CATEGORY:")
    categories = set(r["category"] for r in results)
    for cat in sorted(categories):
        cat_results = [r for r in results if r["category"] == cat]
        avg_fact = sum(r["scores"]["key_fact_recall"] for r in cat_results) / len(cat_results)
        avg_judge = sum(r["scores"]["llm_judge"] for r in cat_results) / len(cat_results)
        print(f"    {cat:25s}: fact_recall={avg_fact:.2%} | llm_judge={avg_judge:.2%} ({len(cat_results)} cases)")
    
    # By difficulty
    print(f"\n  BY DIFFICULTY:")
    for diff in ["easy", "medium", "hard"]:
        diff_results = [r for r in results if r["difficulty"] == diff]
        if diff_results:
            avg_fact = sum(r["scores"]["key_fact_recall"] for r in diff_results) / len(diff_results)
            avg_judge = sum(r["scores"]["llm_judge"] for r in diff_results) / len(diff_results)
            print(f"    {diff:25s}: fact_recall={avg_fact:.2%} | llm_judge={avg_judge:.2%} ({len(diff_results)} cases)")
    
    # Failures (score < 0.5)
    print(f"\n  FAILURES (key_fact_recall < 50%):")
    failures = [r for r in results if r["scores"]["key_fact_recall"] < 0.5 and r["is_answerable"]]
    if failures:
        for f in failures:
            print(f"    {f['id']:8s} | {f['question'][:45]:45s} | recall={f['scores']['key_fact_recall']:.2%}")
            if f["details"]["fact_recall"]["missed"]:
                print(f"             Missed facts: {f['details']['fact_recall']['missed']}")
    else:
        print(f"    None! All answerable questions scored >= 50%")
    
    # Hallucination check
    print(f"\n  HALLUCINATION CHECK:")
    unanswerable = [r for r in results if not r["is_answerable"]]
    for u in unanswerable:
        status = u["details"]["hallucination"]["actual_behavior"]
        icon = "PASS" if u["scores"]["hallucination"] == 1.0 else "FAIL"
        print(f"    [{icon}] {u['id']:8s} | {u['question'][:45]:45s} | {status}")


if __name__ == "__main__":
    import os
    os.makedirs("eval", exist_ok=True)
    run_evaluation(use_llm_judge=True)
