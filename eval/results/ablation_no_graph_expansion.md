# ContextForge evaluation results

Generated 2026-10-03T08:40:08+00:00 by `scripts/run_eval.py` from the working tree committed together with this file. All numbers below are produced by the evaluation harness; nothing is hand-entered.

* Storage backend: **postgres+pgvector (bm25)**; embeddings: **all-MiniLM-L6-v2 (onnx)**; entity extraction: **rules+spacy**
* Corpus: 29 documents, 110 chunks, 115 entities, 254 relationship rows
* Dataset `contextforge-sample-v1` v1: 100 queries (82 answerable, 18 unanswerable); categories: entity=22, lexical=20, relational=15, semantic=25, unanswerable=18
* Candidate pool 30 per retriever, fusion `weighted`, normalization `minmax`, confidence threshold 0.45
* Hybrid weights {'vector': 0.5, 'lexical': 0.5, 'entity': 0.0}; hybrid+entity weights {'vector': 0.4, 'lexical': 0.4, 'entity': 0.2}
* Environment: Python 3.11.15, 4 CPUs, Linux-6.18.44-fc-v64-x86_64-with-glibc2.39

## Retrieval quality (answerable queries)

| Method | Recall@1 | Recall@5 | Recall@10 | Hit@5 | MRR | nDCG@10 | Source acc@1 | p50 latency | p95 latency |
|---|---|---|---|---|---|---|---|---|---|
| Lexical (BM25) | 0.594 | 0.811 | 0.908 | 0.829 | 0.731 | 0.765 | 0.744 | 3.2 ms | 4.8 ms |
| Vector (MiniLM) | 0.490 | 0.760 | 0.868 | 0.793 | 0.653 | 0.692 | 0.646 | 3.5 ms | 3.9 ms |
| Hybrid | 0.581 | 0.902 | 0.957 | 0.927 | 0.751 | 0.796 | 0.756 | 6.2 ms | 7.8 ms |
| Hybrid + Entity | 0.594 | 0.872 | 0.933 | 0.890 | 0.746 | 0.785 | 0.756 | 10.2 ms | 11.9 ms |

Latency is end-to-end `engine.search` time per query (query embedding, candidate generation, scoring, reranking and confidence estimation), measured after 3 warm-up queries per mode.

## MRR by query category

| Category | n | Lexical (BM25) | Vector (MiniLM) | Hybrid | Hybrid + Entity |
|---|---|---|---|---|---|
| entity | 22 | 0.807 | 0.670 | 0.849 | 0.871 |
| lexical | 20 | 1.000 | 0.738 | 0.942 | 0.942 |
| relational | 15 | 0.464 | 0.276 | 0.349 | 0.295 |
| semantic | 25 | 0.609 | 0.796 | 0.752 | 0.749 |

## Hit@5 by query category

| Category | n | Lexical (BM25) | Vector (MiniLM) | Hybrid | Hybrid + Entity |
|---|---|---|---|---|---|
| entity | 22 | 0.909 | 0.864 | 0.955 | 0.955 |
| lexical | 20 | 1.000 | 0.850 | 1.000 | 1.000 |
| relational | 15 | 0.733 | 0.400 | 0.800 | 0.600 |
| semantic | 25 | 0.680 | 0.920 | 0.920 | 0.920 |

## Abstention (all queries; positive class = unanswerable)

| Method | Abstain precision | Abstain recall | False-abstention rate | Hit@5 on answered | Mean conf. (answerable) | Mean conf. (unanswerable) |
|---|---|---|---|---|---|---|
| Lexical (BM25) | 0.294 | 0.556 | 0.293 | 0.948 | 0.592 | 0.391 |
| Vector (MiniLM) | 0.269 | 0.389 | 0.232 | 0.857 | 0.613 | 0.523 |
| Hybrid | 0.462 | 0.333 | 0.085 | 0.973 | 0.624 | 0.491 |
| Hybrid + Entity | 0.688 | 0.611 | 0.061 | 0.922 | 0.649 | 0.525 |

## Dev / test split

| Split | Method | Recall@5 | MRR | Abstain P | Abstain R |
|---|---|---|---|---|---|
| dev | Lexical (BM25) | 0.852 | 0.739 | 0.300 | 0.500 |
| dev | Vector (MiniLM) | 0.852 | 0.718 | 0.300 | 0.500 |
| dev | Hybrid | 0.889 | 0.771 | 0.600 | 0.500 |
| dev | Hybrid + Entity | 0.833 | 0.758 | 0.800 | 0.667 |
| test | Lexical (BM25) | 0.791 | 0.727 | 0.292 | 0.583 |
| test | Vector (MiniLM) | 0.715 | 0.621 | 0.250 | 0.333 |
| test | Hybrid | 0.909 | 0.741 | 0.375 | 0.250 |
| test | Hybrid + Entity | 0.891 | 0.740 | 0.636 | 0.583 |

## Entity resolution

| Method | Pairwise precision | Pairwise recall | Pairwise F1 | Predicted clusters | Gold clusters |
|---|---|---|---|---|---|
| Exact string match (baseline) | 0.993 | 0.600 | 0.748 | 42 | 21 |
| Normalized name match (baseline) | 0.994 | 0.734 | 0.844 | 37 | 21 |
| ContextForge resolver | 1.000 | 1.000 | 1.000 | 21 | 21 |

Gold mentions: 144; found among extracted mentions: 144 (coverage 1.000). Pairwise scores are computed over matched mentions only.
