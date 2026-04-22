# AI-Powered RAG Application with Evaluation Framework

A Retrieval-Augmented Generation system built from scratch with a comprehensive evaluation framework. This project demonstrates end-to-end RAG development — from document ingestion and chunking to retrieval optimization and automated quality testing.

## What This Project Demonstrates

- **RAG Pipeline Architecture:** Document loading → chunking → embedding → retrieval → generation with Claude
- **Chunking Strategy Analysis:** Tested 3 strategies (fixed-size, recursive, section-based) with quantified comparison
- **Retrieval Optimization:** Evaluated semantic search, hybrid search, HyDE, parent-child chunking, and reranking — with data on what worked and what regressed
- **Evaluation Framework:** 30-question golden dataset with automated metrics (fact recall, hallucination detection, retrieval precision, LLM-as-judge)
- **Cost Modeling:** Full cost projections across model tiers (Haiku vs Sonnet)

## Architecture

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   Document   │ ──→ │   Chunking   │ ──→ │  Embedding   │ ──→ │  Generation  │
│   Loading    │     │  (3 strategies)│    │  + Retrieval │     │  (Claude)    │
└──────────────┘     └──────────────┘     └──────────────┘     └──────────────┘
                                                                      │
                                                               ┌──────┴──────┐
                                                               │  Evaluation │
                                                               │  Framework  │
                                                               └─────────────┘
```

## Key Findings

### Chunking Strategy Comparison

| Strategy | Chunks | Best For | Weakness |
|----------|--------|----------|----------|
| Fixed-size (500 chars) | 28 | Baseline only | Cuts mid-sentence, highest false confidence on wrong answers |
| Recursive (1200 chars) | 14 | General-purpose (shipped as default) | Can merge unrelated content in adjacent paragraphs |
| Section-based (7 sections) | 7 | Simple lookups | Large sections dilute embeddings, fails on specific entity queries |

### Retrieval Approaches Tested

| Approach | Result | Shipped? |
|----------|--------|----------|
| Semantic search (cosine similarity) | 77% fact recall baseline | Yes (default) |
| Hybrid search (semantic + BM25) | Fixed 2 queries, broke 2 others — net negative | No |
| HyDE (hypothetical question embeddings) | Higher scores but wrong chunks still ranked top | No |
| Parent-child chunking | +17% education, regressed career queries | No |
| LLM reranking | Improved ordering when correct chunk was in top K | Tested |

### The Root Cause Discovery

Through systematic debugging, I identified that the remaining retrieval failures share one root cause: **vocabulary mismatch between user queries and document content.** The embedding model cannot connect "current role" with "Sep 2025 – Present" because this requires temporal reasoning, not semantic similarity. This class of failure cannot be fixed by chunking or retrieval architecture alone — it requires either keyword fallback, knowledge graphs, or query preprocessing.

### Evaluation Results (Baseline)

```
Overall Scores:
  key_fact_recall     : 77.17%
  hallucination       : 90.00%  (100% pass on unanswerable questions)
  retrieval_precision : 82.78%
  llm_judge           : 78.00%

By Difficulty:
  easy   : 95% fact recall
  medium : 79% fact recall
  hard   : 25% fact recall
```

### Cost Model

| Metric | Value |
|--------|-------|
| Avg cost per query (Haiku) | $0.000382 |
| 1,000 queries/day (Haiku) | $0.38/day (~$11/month) |
| 1,000 queries/day (Sonnet) | ~$3.80/day (~$114/month) |

## Project Structure

```
├── src/
│   ├── ingestion.py      # Document loading + 3 chunking strategies
│   ├── retrieval.py       # Embedding + semantic search (numpy-based)
│   ├── generation.py      # Claude integration with reranking
│   ├── config.py          # API keys, model settings, thresholds
│   ├── hyde.py            # HyDE implementation (hypothetical question embeddings)
│   └── parent_child.py    # Parent-child chunking experiment
├── eval/
│   ├── eval_dataset.py    # 30-question golden dataset across 8 categories
│   ├── eval_runner.py     # Automated evaluation with 4 metrics
│   └── eval_compare.py    # Side-by-side comparison runner
├── docs/
│   ├── PRD.md             # Product Requirements Document
│   └── chunking_analysis.md
├── data/                  # Source documents
└── .env                   # API keys (not committed)
```

## How to Run

### Prerequisites
- Python 3.10+
- OpenAI API key (for embeddings)
- Anthropic API key (for generation)

### Setup
```bash
git clone https://github.com/YOUR_USERNAME/ai-pm-rag-evaluator.git
cd ai-pm-rag-evaluator
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Add your API keys to .env
```

### Run the RAG pipeline
```bash
PYTHONPATH=src python src/generation.py
```

### Run the evaluation
```bash
PYTHONPATH=src:eval python eval/eval_runner.py
```

### Compare chunking strategies
```bash
PYTHONPATH=src python src/ingestion.py
```

## Debugging Framework

Every RAG failure traces to one of four layers. Diagnose top-down:

```
1. DATA        → Is the answer in the source document?
2. CHUNKING    → Is it intact in one chunk, or split across boundaries?
3. RETRIEVAL   → Does the right chunk appear in top K results?
4. GENERATION  → Right chunks retrieved, but LLM gave wrong answer?
```

Fix at the highest layer possible. Data fixes are cheapest and most reliable. Generation fixes (prompt tuning) are the last resort and most fragile.

## Lessons Learned

1. **Retrieval quality matters more than prompt engineering.** Every answer failure traced to retrieval, not generation. The system prompt worked from v1.
2. **Improving one metric can degrade others.** Hybrid search fixed 2 query types but broke 2 others. Never ship without automated regression testing.
3. **Fix at the highest layer possible.** Data restructuring is free. Engineering fixes add complexity and risk.
4. **Embedding dilution is the silent killer.** A chunk with 5 topics scores medium for all and high for none.
5. **Unanswerable queries are as important as answerable ones.** The similarity gap between answerable (0.52) and unanswerable (0.46) was dangerously small.

## Tech Stack

- **Embeddings:** OpenAI text-embedding-3-small
- **Generation:** Claude Haiku (claude-haiku-4-5-20251001)
- **Vector Search:** Custom numpy-based cosine similarity
- **Framework:** LangChain (document loading only), custom code for retrieval and generation
- **Evaluation:** Custom framework with LLM-as-judge

## What I'd Build Next (V2 Roadmap)

1. **Conditional keyword fallback** — trigger BM25 only when semantic confidence is low, avoiding hybrid search regressions
2. **Knowledge graph extraction** — structured entity-relationship store for factual lookups that bypass embeddings entirely
3. **Query routing** — classify query type and route to the best retrieval strategy per type
4. **Parent-child with HyDE combined** — small focused children with generated questions, large parents for LLM context
