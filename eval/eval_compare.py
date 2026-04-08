"""
eval_compare.py — Compare baseline (recursive) vs parent-child chunking

Runs the full golden dataset against both approaches and prints
a side-by-side comparison.
"""

import json
import time
import os
from dotenv import load_dotenv

load_dotenv()


def run_comparison():
    from eval_dataset import get_dataset
    from ingestion import load_documents, chunk_recursive
    from retrieval import get_embedding_model, SimpleVectorStore
    from generation import generate_answer
    from parent_child import build_parent_child_store, search_parent_child
    from eval_runner import score_key_fact_recall, score_hallucination, score_retrieval, score_with_llm_judge
    
    docs = load_documents("data")
    embeddings = get_embedding_model()
    dataset = get_dataset()
    
    # ── Setup baseline (recursive chunking) ──
    print("\nSetting up BASELINE (recursive chunking)...")
    baseline_chunks = chunk_recursive(docs, chunk_size=1200, chunk_overlap=300)
    baseline_store = SimpleVectorStore(embeddings)
    baseline_store.add_documents(baseline_chunks)
    
    # ── Setup parent-child ──
    print("\nSetting up PARENT-CHILD chunking...")
    pc_store, pc_parents = build_parent_child_store(docs, embeddings, parent_size=1500, child_size=300)
    
    print(f"\nRunning {len(dataset)} test cases against both approaches...\n")
    
    baseline_results = []
    pc_results = []
    
    for i, test_case in enumerate(dataset):
        qid = test_case["id"]
        question = test_case["question"]
        print(f"[{i+1}/{len(dataset)}] {qid}: {question[:50]}...")
        
        # ── Baseline ──
        b_retrieved = baseline_store.search(question, top_k=5)
        b_rag = generate_answer(question, b_retrieved)
        b_answer = b_rag["answer"]
        
        b_fact = score_key_fact_recall(b_answer, test_case["key_facts"])
        b_hall = score_hallucination(b_answer, test_case["is_answerable"])
        
        baseline_results.append({
            "id": qid,
            "category": test_case["category"],
            "difficulty": test_case["difficulty"],
            "is_answerable": test_case["is_answerable"],
            "key_fact_recall": b_fact["score"],
            "hallucination": b_hall["score"],
            "missed_facts": b_fact["missed"],
        })
        
        # ── Parent-child ──
        pc_retrieved = search_parent_child(pc_store, pc_parents, question, top_k=8, return_parents=3)
        pc_rag = generate_answer(question, pc_retrieved)
        pc_answer = pc_rag["answer"]
        
        pc_fact = score_key_fact_recall(pc_answer, test_case["key_facts"])
        pc_hall = score_hallucination(pc_answer, test_case["is_answerable"])
        
        pc_results.append({
            "id": qid,
            "category": test_case["category"],
            "difficulty": test_case["difficulty"],
            "is_answerable": test_case["is_answerable"],
            "key_fact_recall": pc_fact["score"],
            "hallucination": pc_hall["score"],
            "missed_facts": pc_fact["missed"],
        })
    
    # ── Print comparison ──
    print(f"\n\n{'=' * 70}")
    print(f"  BASELINE vs PARENT-CHILD COMPARISON")
    print(f"{'=' * 70}")
    
    # Overall
    b_avg_fact = sum(r["key_fact_recall"] for r in baseline_results) / len(baseline_results)
    pc_avg_fact = sum(r["key_fact_recall"] for r in pc_results) / len(pc_results)
    b_avg_hall = sum(r["hallucination"] for r in baseline_results) / len(baseline_results)
    pc_avg_hall = sum(r["hallucination"] for r in pc_results) / len(pc_results)
    
    delta_fact = pc_avg_fact - b_avg_fact
    delta_hall = pc_avg_hall - b_avg_hall
    
    print(f"\n  OVERALL:")
    print(f"    {'Metric':<25s} {'Baseline':>10s} {'Parent-Child':>14s} {'Delta':>10s}")
    print(f"    {'key_fact_recall':<25s} {b_avg_fact:>10.2%} {pc_avg_fact:>14.2%} {delta_fact:>+10.2%}")
    print(f"    {'hallucination':<25s} {b_avg_hall:>10.2%} {pc_avg_hall:>14.2%} {delta_hall:>+10.2%}")
    
    # By category
    print(f"\n  BY CATEGORY (key_fact_recall):")
    categories = sorted(set(r["category"] for r in baseline_results))
    print(f"    {'Category':<25s} {'Baseline':>10s} {'Parent-Child':>14s} {'Delta':>10s}")
    for cat in categories:
        b_cat = [r for r in baseline_results if r["category"] == cat]
        pc_cat = [r for r in pc_results if r["category"] == cat]
        b_avg = sum(r["key_fact_recall"] for r in b_cat) / len(b_cat)
        pc_avg = sum(r["key_fact_recall"] for r in pc_cat) / len(pc_cat)
        delta = pc_avg - b_avg
        marker = " <<<" if abs(delta) > 0.1 else ""
        print(f"    {cat:<25s} {b_avg:>10.2%} {pc_avg:>14.2%} {delta:>+10.2%}{marker}")
    
    # By difficulty
    print(f"\n  BY DIFFICULTY (key_fact_recall):")
    print(f"    {'Difficulty':<25s} {'Baseline':>10s} {'Parent-Child':>14s} {'Delta':>10s}")
    for diff in ["easy", "medium", "hard"]:
        b_diff = [r for r in baseline_results if r["difficulty"] == diff]
        pc_diff = [r for r in pc_results if r["difficulty"] == diff]
        if b_diff:
            b_avg = sum(r["key_fact_recall"] for r in b_diff) / len(b_diff)
            pc_avg = sum(r["key_fact_recall"] for r in pc_diff) / len(pc_diff)
            delta = pc_avg - b_avg
            print(f"    {diff:<25s} {b_avg:>10.2%} {pc_avg:>14.2%} {delta:>+10.2%}")
    
    # Per-question comparison for previously failing queries
    print(f"\n  PREVIOUSLY FAILING QUERIES:")
    print(f"    {'ID':<10s} {'Question':<40s} {'Base':>6s} {'PC':>6s} {'Fixed?':>8s}")
    failing_ids = ["CA-002", "CA-003", "CA-005", "IN-004", "CC-001", "CC-003"]
    for qid in failing_ids:
        b = next(r for r in baseline_results if r["id"] == qid)
        pc = next(r for r in pc_results if r["id"] == qid)
        question = next(t["question"] for t in dataset if t["id"] == qid)
        fixed = "YES" if pc["key_fact_recall"] > b["key_fact_recall"] else ("SAME" if pc["key_fact_recall"] == b["key_fact_recall"] else "WORSE")
        print(f"    {qid:<10s} {question[:38]:<40s} {b['key_fact_recall']:>5.0%} {pc['key_fact_recall']:>5.0%} {fixed:>8s}")
        if pc["missed_facts"]:
            print(f"              Still missing: {pc['missed_facts']}")
    
    # Regression check
    print(f"\n  REGRESSION CHECK (did anything get WORSE?):")
    regressions = []
    for b, pc in zip(baseline_results, pc_results):
        if pc["key_fact_recall"] < b["key_fact_recall"]:
            regressions.append((b["id"], b["key_fact_recall"], pc["key_fact_recall"]))
    
    if regressions:
        for qid, b_score, pc_score in regressions:
            print(f"    REGRESSION: {qid} dropped from {b_score:.0%} to {pc_score:.0%}")
    else:
        print(f"    No regressions! Parent-child is safe to ship.")
    
    # Save results
    comparison = {
        "baseline": baseline_results,
        "parent_child": pc_results,
        "summary": {
            "baseline_fact_recall": b_avg_fact,
            "parent_child_fact_recall": pc_avg_fact,
            "delta": delta_fact,
        }
    }
    with open("eval/comparison_results.json", "w") as f:
        json.dump(comparison, f, indent=2)
    print(f"\n  Results saved to eval/comparison_results.json")


if __name__ == "__main__":
    os.makedirs("eval", exist_ok=True)
    run_comparison()
