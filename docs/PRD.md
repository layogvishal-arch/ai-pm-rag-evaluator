# Product Requirements Document: AI-Powered Personal Knowledge Base (RAG)

**Author:** Vishal Goyal  
**Date:** April 2026  
**Status:** V1 Shipped, V2 In Planning  
**Repo:** ai-pm-rag-evaluator

---

## 1. Problem Statement

Knowledge workers spend significant time searching through personal documents, resumes, and career records to answer recurring questions — from recruiters, colleagues, or even themselves. Existing solutions (Ctrl+F, manual scanning) fail when the answer spans multiple sections, uses different vocabulary than the query, or requires synthesis across documents.

This project builds a Retrieval-Augmented Generation (RAG) system that lets users ask natural language questions about a personal knowledge base and receive accurate, cited answers.

**Target user:** Professionals who need quick, accurate retrieval from personal or organizational documents.

---

## 2. Architecture Overview

The system follows a four-layer pipeline. Every failure in the system can be traced to one of these layers, and fixes should be applied at the highest layer possible (cheapest, most reliable).

```
[Source Documents] → [Chunking] → [Embedding + Retrieval] → [Generation]
     Layer 1            Layer 2         Layer 3                Layer 4
     Data             Ingestion        Retrieval             Generation
```

**Technology stack:**
- Embedding model: OpenAI text-embedding-3-small (1536 dimensions, $0.02/1M tokens)
- Generation model: Claude Haiku (claude-haiku-4-5-20251001)
- Vector store: Numpy-based cosine similarity (no external DB dependencies)
- Framework: LangChain for document loading, custom code for retrieval and generation

---

## 3. Chunking Strategy Analysis

Three chunking strategies were implemented and tested against 6 queries (4 answerable, 1 cross-cutting, 1 unanswerable).

### Strategies tested

**Fixed-size (500 chars):** Splits every N characters regardless of content. Produced 28 chunks. Cuts mid-sentence, loses context at boundaries. Resulted in the worst retrieval quality — retrieved Deloitte content when asked about HireQuotient (high confidence on wrong answer, similarity 0.5569).

**Recursive character (800 chars, 100 overlap):** Splits by paragraph, then sentence, then word boundaries. Produced 20 chunks. Best all-round performer. Correctly retrieved HireQuotient as top result. Gave the lowest false confidence on unanswerable queries (0.4626 for salary question).

**Section-based (7 sections):** Splits on document section headers. Produced 7 chunks. Best for simple lookups (birthplace scored 0.5476 vs 0.5257 for fixed-size). Failed on specific entity queries due to embedding dilution — the Career section contained 5 companies in one chunk, so "HireQuotient" signal was diluted.

### Comparative results

| Query | Fixed-size | Recursive | Section-based | Winner |
|-------|-----------|-----------|---------------|--------|
| Where was Vishal born? | 0.5257 (correct) | 0.5190 (correct) | 0.5476 (correct) | Section |
| What did Vishal do at HireQuotient? | 0.5569 (WRONG - Deloitte) | 0.5361 (correct) | 0.5260 (WRONG - Education) | Recursive |
| Who is Sejal? | 0.4506 (WRONG - ServiceNow) | 0.3783 (correct) | 0.4234 (partial) | Recursive |
| What is Vishal's salary? (unanswerable) | 0.5038 (too high) | 0.4626 (lowest) | 0.4805 (mid) | Recursive |
| Leadership experiences? | 0.5597 (partial) | 0.6108 (partial) | 0.6278 (partial) | All partial |

### Recommendation

Recursive chunking as the primary strategy (1200 chars, 300 overlap after tuning). Section-based is superior for structured lookups but fails on specific entity queries. Fixed-size should never be used in production.

### Key finding: embedding dilution

When a chunk contains multiple topics (e.g., HireQuotient CRM + Partnerships + Tech Mahindra in one chunk), the embedding represents the average meaning rather than any specific topic. This causes the chunk to score low for specific queries about any individual topic. Solution: ensure each chunk covers one coherent topic, or use parent-child chunking in V2.

---

## 4. Retrieval Analysis

### Semantic-only search

Uses cosine similarity between query embedding and chunk embeddings. Strengths: captures meaning ("born" matches "hometown"). Weakness: fails on proper nouns ("Sejal" scored 0.3783) and vocabulary mismatches ("current role" doesn't match "Sep 2025 – Present").

### Hybrid search (tested, not shipped)

Combined semantic similarity (60%) with BM25 keyword matching (40%). Fixed the "Sejal" and "programming languages" retrieval failures. However, introduced regressions on previously working queries — "HireQuotient" and "Where was Vishal born?" both degraded because common keywords like "Vishal" appear in every chunk, boosting irrelevant results.

**Decision: Not shipped in V1.** Net impact was negative (fixed 2 queries, broke 2, worsened 1). This validates the need for an automated evaluation framework before tuning retrieval parameters — manual testing misses regressions.

### Reranking

Implemented LLM-based reranking using Claude Haiku. After semantic retrieval returns top 5 chunks, Claude re-scores them by relevance to the specific query. Cost: ~$0.0002 per query additional. Latency: ~500ms additional.

Reranking improved result ordering when the correct chunk was in the top 5. However, it cannot fix cases where the correct chunk is not retrieved at all (the "current role" problem where Tech Mahindra ranked #9 out of 14).

### The "current role" failure — root cause analysis

**Problem:** "What is Vishal's current role?" never retrieved the Tech Mahindra chunk.

**Layer-by-layer diagnosis:**
1. Data layer: Information exists in the document. Not a data problem.
2. Chunking layer: Recursive splitting merged Tech Mahindra content with HireQuotient CRM and Partnerships content. The chunk is dominated by HireQuotient topics. This IS the root cause.
3. Retrieval layer: The diluted embedding scored 0.3044, ranking it #9 out of 14. Too low for top-5 retrieval. Consequence of layer 2.
4. Generation layer: Never received the right chunk, so correctly said "I don't know." Generation is working as designed.

**Fix options evaluated (in order of PM preference):**

| Fix | Layer | Effort | Impact | Risk | Recommendation |
|-----|-------|--------|--------|------|----------------|
| Add "Current role" field to Personal Identity section | Data | 5 min | Fixes this query | None | Do immediately |
| Custom chunking: split at job title patterns | Chunking | 2 hours | Fixes all role queries | May over-split | V1.1 |
| Parent-child chunking | Chunking | 4 hours | Systemic fix for dilution | More complex indexing | V2 |
| HyDE (hypothetical question embeddings) | Retrieval | 6 hours | Fixes vocabulary mismatch | LLM cost at indexing | V2 |
| Hybrid search with per-query weight tuning | Retrieval | 8 hours | Broad improvement | Regression risk, needs eval framework | V2 (after eval framework) |

---

## 5. Generation Layer

### System prompt design

The system prompt enforces three behaviors:
1. Context-only answers: prevents hallucination by restricting to retrieved content
2. Graceful refusal: explicitly says "I don't have enough information" rather than guessing
3. Source citation: cites which section the answer came from

### Validation results

- Accurately answered 4/5 answerable queries with correct citations
- Correctly refused the unanswerable query ("salary") with no hallucination
- "Current role" was a retrieval failure, not a generation failure — the model correctly said "I don't know" when the right chunk was not provided
- Multi-turn conversation support via conversation history in message array

### Cost model

| Metric | Value |
|--------|-------|
| Avg input tokens per query | 842 |
| Avg output tokens per query | 135 |
| Avg cost per query (Haiku) | $0.000382 |
| Cost per 1,000 queries/day | $0.38/day (~$11/month) |
| Cost per 1,000 queries/day (Sonnet) | ~$3.80/day (~$114/month) |
| Reranking overhead | +$0.0002/query |

---

## 6. Chunking Strategies Reference

Through research and testing, 10 chunking strategies were identified, ordered by complexity:

**Text-level (basic):** Fixed-size, recursive character, sentence-level. Generic, work on any document. Recursive is the best default.

**Structure-aware (intermediate):** Header/section-based, HTML/code-aware, table-aware. Require knowing document format. Best for structured content like documentation and wikis.

**Semantic (advanced):** Embedding-based topic detection, LLM-powered chunking. Understand meaning. Best for long unstructured narratives. Higher cost.

**Production patterns (combined):** Parent-child chunking (small chunks for retrieval, large chunks for context), HyDE (store hypothetical questions as embeddings instead of raw text). These solve the fundamental trade-off between retrieval precision and context completeness.

**PM decision framework:** Start with recursive. If retrieval quality is below target, move up one level. Each level adds cost and complexity. Stop when quality meets threshold.

---

## 7. Known Limitations and V2 Roadmap

### Current limitations
- Pure semantic search fails on proper noun lookups (names, company names)
- Large chunks cause embedding dilution for multi-topic content
- "Current" / "Present" vocabulary mismatch not resolved
- No automated evaluation framework (manual testing only)
- Single document only — no multi-document support

### V2 roadmap (prioritized)

1. **Evaluation framework** (Days 3-4): Golden dataset of 30-50 Q&A pairs. Automated metrics for retrieval precision, answer faithfulness, hallucination rate. Regression testing before any retrieval changes.

2. **Parent-child chunking:** Small chunks (200 chars) indexed for retrieval, parent chunks (1200 chars) sent to LLM. Fixes embedding dilution without losing context.

3. **Hybrid search with eval-gated deployment:** BM25 + semantic, but only ship after evaluation framework confirms no regressions.

4. **HyDE indexing:** Generate 3-5 hypothetical questions per chunk at indexing time. Fixes vocabulary mismatch between user queries and document content.

5. **Multi-document support:** Ingest resume, project docs, and meeting notes. Cross-document queries.

---

## 8. Lessons Learned

1. **Retrieval quality > prompt engineering.** The system prompt worked well from V1. Every answer failure traced back to retrieval, not generation. Teams that spend weeks tuning prompts while ignoring retrieval are optimizing the wrong layer.

2. **Improving one metric can degrade others.** Hybrid search fixed 2 query types but broke 2 others. Never ship retrieval changes without automated regression testing.

3. **Fix at the highest layer possible.** Data fixes (restructuring the source document) are free and reliable. Engineering fixes (hybrid search, reranking) add complexity and risk. Always ask "can I fix this with better data?" first.

4. **Embedding dilution is the silent killer.** A chunk with 5 topics will score medium for all of them and high for none. One topic per chunk is the goal.

5. **Unanswerable queries are as important as answerable ones.** A system that says "I know" when it doesn't is more dangerous than one that says "I don't know" too often. The similarity gap between answerable (0.52) and unanswerable (0.46) queries was dangerously small — only 0.06 difference.
