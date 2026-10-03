# ContextForge architecture

This document explains how the pieces fit together and why they are built the way they are. The README covers usage and results.

## Module map

```
app/
  core/config.py        Settings: every tunable (weights, pools, thresholds, ER weights), env-overridable (CF_*)
  core/container.py     Dependency container: builds repo, embedder, extractor, resolver, engine, generator
  core/text.py          Text normalization, lexical analyzer (Snowball stems, Postgres stop list), sentence splitter
  ingestion/            parsers.py (TXT/MD/PDF/JSON -> blocks with offsets), chunker.py, pipeline.py
  embeddings/           Embedder protocol; onnx_minilm.py (default), sentence_transformer.py, hashing.py (tests)
  db/                   Repository protocol; postgres.py (+ schema.sql); memory.py
  entities/             extraction, normalization, resolution, relationships, linking, graph, service
  retrieval/            bm25.py, normalization.py, entity_scorer.py, reranker.py, confidence.py, engine.py
  generation/           optional: extractive baseline, Ollama client, citation validation
  evaluation/           dataset + relevance resolution, metrics, runner, report, entity-resolution eval, CLI
  api/                  FastAPI routers; schemas/ holds the Pydantic request/response models
```

Every consumer (API, CLI scripts, evaluation harness, tests) builds the system through `Container`. The benchmark therefore exercises exactly the wiring that serves requests.

## Data model (PostgreSQL)

| Table | Key columns | Notes |
|---|---|---|
| `documents` | `id`, `filename`, `source_type`, `content_hash`, `title`, `text`, `metadata`, `created_at` | `UNIQUE(filename, content_hash)` makes ingestion idempotent; `text` is the normalized full text that chunk offsets index into |
| `chunks` | `id`, `document_id`, `chunk_index`, `text`, `search_text`, `char_start`, `char_end`, `metadata`, `embedding vector(384)`, `tsv` | `tsv` is a generated `to_tsvector('english', search_text)` column with a GIN index; `embedding` has an HNSW (`vector_cosine_ops`) index |
| `entity_mentions` | `id`, `chunk_id`, `document_id`, `surface`, `entity_type`, `char_start`, `char_end`, `extractor`, `confidence`, `entity_id` | Extractor `rules`/`spacy` at ingest; `known_name`/`header` are derived at each entity rebuild |
| `entities` | `id`, `canonical_name`, `entity_type`, `aliases[]`, `mention_count`, `document_count` | IDs are hashes of type + canonical name, stable across rebuilds |
| `entity_relationships` | `source_entity_id`, `target_entity_id`, `relationship_type`, `confidence`, `supporting_chunk_id`, `evidence`, `method` | One row per (pair, relation, supporting chunk); aggregated at query time |
| `resolution_decisions` | `left_surface`, `right_surface`, `score`, `decision`, `features jsonb`, `reason` | The inspectable ER log |
| `lexeme_stats` (materialized view) | `lexeme`, `df` | `ts_stat` over all chunks; BM25 document frequencies |
| `corpus_stats` (materialized view) | `n_chunks`, `avgdl` | BM25 corpus statistics |

Document IDs are `doc_` + a hash of (filename, content hash), and chunk IDs are `<doc_id>:<index>`. The same corpus always produces the same IDs, which makes results diffable across runs.

## Ingestion flow

1. **Parse** (`ingestion/parsers.py`). Each format becomes a normalized text plus *blocks* with character offsets and structural metadata.
   * Markdown: heading path; fenced code is kept intact.
   * PDF (pypdf): page numbers; hyphenated line breaks are repaired.
   * JSON: the largest list of objects becomes the records, with a JSON pointer per record; scalar top-level fields form a header block.
2. **Chunk** (`ingestion/chunker.py`). Blocks are packed into chunks of at most about 180 approximate tokens (MiniLM truncates at 256 word pieces).
   * Top-level sections, PDF pages and JSON records always start a new chunk.
   * Oversized blocks are split on sentence boundaries with one sentence of overlap, and a short lead-in such as a section heading is folded into the first piece.
   * `search_text = "<title> > <section path>\n" + text` is what gets embedded and indexed; `text` stays an exact slice of the document.
3. **Embed** the `search_text` in batches (ONNX Runtime, mean pooling, L2 normalization).
4. **Extract mentions** from each chunk's `text` (rules + spaCy run per paragraph, with headings excluded from the NER input).
5. **Store** the document, chunks and mentions in one transaction.
6. After the batch: refresh the BM25 statistics (`REFRESH MATERIALIZED VIEW`) and **rebuild the entity layer** (below).

### Entity rebuild (`entities/service.py`)

```
stored mentions ──► known-name pass (bare "Halcyon", "Marta" once "Project Halcyon", "Marta Kowalski" are known)
                └─► resolver: type harmonization → grouping → within-doc coreference
                              → blocked pairwise log-linear scoring → constrained union-find
                └─► header pass: entities named in each chunk's title/section header (exact, unambiguous)
                └─► relationship extraction per sentence / JSON record (+ header context entities)
                └─► persist atomically; build the in-process EntityGraph (alias index, chunk↔entity maps, noisy-OR adjacency)
```

The rebuild is global: about 60 ms for the sample corpus, O(corpus) in general. That keeps the logic simple and makes every result reproducible. Incremental resolution is listed as future work.

## Query flow (`retrieval/engine.py`)

| Stage | What happens | Typical time (sample corpus, Postgres) |
|---|---|---|
| lexical | BM25 SQL: GIN-prefiltered OR `tsquery`, TF via `unnest(tsvector)`, IDF from `lexeme_stats` (computed once per query in a materialized CTE) | ~2–3 ms |
| embed | ONNX MiniLM query embedding | ~3–5 ms |
| vector | `ORDER BY embedding <=> $q LIMIT 30` on HNSW (`ef_search=100`) | ~1 ms |
| entity | link query spans → plan (direct + typed 1-hop expansion) → entity candidates | ~3–10 ms |
| scoring | fill in every active signal for every candidate in the union (`lexical_scores`, `vector_scores`) | ~2–3 ms |
| rerank | normalize, weighted blend (or RRF), sort | < 0.1 ms |
| confidence | IDF-weighted term coverage of the top chunks, agreement, support, gates (incl. unknown-entity lookup) | ~2 ms |

Every response includes `timings_ms` with this breakdown.

## Design decisions and trade-offs

| Decision | Alternative | Why |
|---|---|---|
| all-MiniLM-L6-v2 through **ONNX Runtime** | `sentence-transformers` + PyTorch | Same weights, no 1–2 GB torch dependency, fast CPU inference. Parity checked against the published reference similarity (0.7553). The `sentence-transformers` backend still exists behind the same `Embedder` protocol. |
| **pgvector inside PostgreSQL** | Dedicated vector DB | One transactional store for text, vectors, entities and provenance; no dual-write consistency problem. Lower peak vector throughput is acceptable at this scale. |
| **BM25 computed in SQL** | `ts_rank_cd`; Elasticsearch/OpenSearch | `ts_rank_cd` has no IDF, measured at 0.669 vs 0.731 lexical MRR. An external engine would add a second system to keep in sync. |
| **Score every candidate on every signal** | Treat a candidate missing from a retriever's top-k as 0 | Truncation artefacts otherwise dominate fusion; the extra cost is two indexed lookups over ≤ 90 IDs. |
| **Per-query min-max** for lexical/vector; **absolute** entity score | Global calibration; RRF | BM25 is unbounded and cosine ranges vary per query, so min-max makes them commensurable. The entity score is meaningful in absolute terms (1.0 = mentions every queried entity). RRF was measured and is worse here (hybrid MRR 0.684 vs 0.751) because it discards score gaps. |
| **Raw signals for confidence** | Use the fused score | Normalized scores are relative by construction (the top result is always near 1.0), so they cannot express "everything is weak". Confidence uses raw cosine and IDF-weighted term coverage instead. |
| **Hand-set log-linear ER model** | Learned classifier; LLM-based resolution | No labelled pairs at the start; every weight is inspectable and every decision is logged with its features. The model form allows fitting the weights later by logistic regression. |
| **Constrained clustering** | Plain transitive closure | Prevents "Quillon" merging with both "Quillon Labs" and "Quillon Health" and so chaining them together. |
| **In-process entity graph** | Graph database; recursive SQL | The graph is small, and 1-hop lookups from a dict are microseconds. The source of truth stays in PostgreSQL tables, so recursive CTEs are the scale-out path. |
| **Evidence-string relevance labels** | Hard-coded chunk IDs | Labels survive chunking changes, and the harness refuses to run on a stale label. |
| **Deterministic, idempotent ingestion** | Random UUIDs | Stable IDs across rebuilds; re-uploading identical content is a no-op reported as `duplicate`. |

## Extension points

* `Embedder.embed(texts) -> np.ndarray`: add a model (e.g. bge-small) without touching retrieval.
* `Repository`: storage backends (both existing backends pass the same engine tests).
* `EntityExtractor.extract_many(texts)`: e.g. a GLiNER or LLM-based extractor alongside the rules.
* `Reranker.rerank(query, candidates, chunks, weights)`: a cross-encoder reranker.
* `Generator.generate(query, evidence)`: any local or remote model; citations are validated outside the generator.

## Concurrency

Reads (search, entity lookups) are lock-free. Writes (ingest, delete, entity rebuild) are serialized by a re-entrant lock in `Container`. The entity graph is swapped atomically after a rebuild. The PostgreSQL backend uses a `psycopg_pool` connection pool with pgvector types registered per connection.
