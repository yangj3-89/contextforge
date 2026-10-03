# ContextForge

ContextForge is a provenance-aware hybrid retrieval engine that combines lexical search, semantic similarity, and entity relationships to retrieve evidence from heterogeneous document collections.

Unlike vector-only RAG systems, ContextForge evaluates multiple retrieval signals, preserves source-level provenance, resolves recurring entities across documents, and abstains when available evidence is insufficient.

**Stack:** Python 3.11 · FastAPI · Pydantic · PostgreSQL 16 + pgvector (HNSW) · PostgreSQL full-text search with BM25 computed in SQL · all-MiniLM-L6-v2 (ONNX Runtime) · spaCy + rules for NER · React + TypeScript · pytest. No paid APIs; generation is optional and local (Ollama).

![Search with entity expansion](docs/images/search_success.png)

---

## Contents

1. [Why ContextForge exists](#why-contextforge-exists)
2. [Results at a glance](#results-at-a-glance)
3. [Architecture](#architecture)
4. [Retrieval pipeline](#retrieval-pipeline)
5. [Entity resolution](#entity-resolution)
6. [Relationships and entity-aware retrieval](#relationships-and-entity-aware-retrieval)
7. [Confidence and abstention](#confidence-and-abstention)
8. [Evaluation methodology and results](#evaluation-methodology-and-results)
9. [API examples](#api-examples)
10. [Screenshots](#screenshots)
11. [Installation and running](#installation-and-running)
12. [Repository layout](#repository-layout)
13. [Limitations](#limitations)
14. [Future work](#future-work)

---

## Why ContextForge exists

Most "chat with your documents" systems embed chunks, take the top-k by cosine similarity, and hand them to an LLM. That design has three blind spots:

* **Exact identifiers.** Error codes, ticket keys and config names (`QL-5031`, `HAL-157`, `retrieval.hybrid_alpha`) are poorly represented by sentence embeddings. A lexical retriever finds them trivially.
* **Entities and relationships.** "Which database does *the project led by Alice Chen* use?" requires knowing that Alice Chen leads Project Halcyon. The answer passage mentions Halcyon and PostgreSQL but never Alice. Neither cosine similarity nor keyword overlap encodes that hop. The problem is compounded when the same entity appears as "Alice Chen", "A. Chen" and "Dr. Chen".
* **No notion of "I don't know".** Top-k always returns *something*. Without an explicit confidence model, weak or irrelevant evidence is passed downstream as if it were an answer.

The engineering question this project answers with measurements rather than claims:

> Can combining lexical retrieval, vector similarity, and entity relationships improve retrieval quality over vector-only retrieval while preserving provenance and abstaining when evidence is weak?

## Results at a glance

These are measured on the bundled synthetic benchmark: 29 documents, 110 chunks, 100 labelled queries, PostgreSQL + pgvector backend. Full report: [`eval/results/latest.md`](eval/results/latest.md). Every number is produced by `scripts/run_eval.py`; none are hand-entered.

| Method | Recall@1 | Recall@5 | Recall@10 | MRR | p50 latency | p95 latency |
|---|---|---|---|---|---|---|
| Lexical (BM25) | 0.594 | 0.811 | 0.908 | 0.731 | 3.2 ms | 4.7 ms |
| Vector (MiniLM) | 0.490 | 0.760 | 0.868 | 0.653 | 3.4 ms | 3.9 ms |
| Hybrid | 0.581 | 0.902 | 0.957 | 0.751 | 6.1 ms | 7.2 ms |
| Hybrid + Entity | 0.581 | **0.927** | 0.957 | 0.751 | 10.5 ms | 12.5 ms |

What the numbers say, including where they don't flatter the system:

* **Hybrid beats vector-only clearly**: Recall@5 0.760 → 0.902, MRR 0.653 → 0.751. On this corpus lexical-only is a strong baseline too (MRR 0.731). Hybrid's gain over lexical is mostly in recall (0.811 → 0.902), not first-rank precision.
* **The entity layer improves recall, not ranking precision.** Hybrid + Entity raises Recall@5 from 0.902 to 0.927, and relational Hit@5 from 0.800 to 0.933. Recall@1 and MRR are unchanged. Relational queries remain the hardest category for every method (best MRR 0.464, lexical).
* **Abstention is where the entity layer matters most.** Hybrid + Entity abstains on 61% of unanswerable queries with 69% precision and a 6.1% false-abstention rate, vs 33% / 46% / 8.5% for plain hybrid. It catches all 10 out-of-domain and unknown-entity questions, but only 1 of 8 in-domain "near miss" questions (see [Limitations](#limitations)).
* **Entity resolution**: pairwise F1 1.000 vs 0.748 (exact string match) and 0.844 (normalized names) on 144 hand-labelled mentions. This is a small, synthetic gold set, so treat it as a correctness check, not a general accuracy claim.

## Architecture

```mermaid
flowchart TD
    A[Documents: TXT / MD / PDF / JSON] --> B[Ingestion: parse + normalize]
    B --> C[Structure-aware chunking<br/>exact char offsets + contextual header]

    C --> D[Embeddings<br/>all-MiniLM-L6-v2, ONNX]
    C --> E[Entity extraction<br/>rules + spaCy]
    C --> G[(tsvector + GIN<br/>BM25 stats)]

    D --> F[(pgvector HNSW)]
    E --> R[Entity resolution<br/>blocking + log-linear matcher<br/>+ constrained clustering]
    R --> H[(entities / mentions /<br/>relationships)]

    F --> I[Vector retrieval]
    G --> J[Lexical retrieval]
    H --> K[Entity linking +<br/>1-hop graph expansion]

    I --> L[Candidate union + score every signal<br/>normalize + weighted rerank]
    J --> L
    K --> L

    L --> M[Confidence gate]
    M -->|sufficient| N[Ranked evidence + provenance<br/>+ component scores]
    M -->|insufficient| O[Abstain with reasons]
    N -.optional.-> P[Cited answer<br/>extractive or Ollama]
```

Everything lives in PostgreSQL: documents, chunks, embeddings (`vector(384)` with an HNSW index), the full-text index (generated `tsvector` column with a GIN index), entities, mentions, relationships and the entity-resolution decision log. No separate vector database or graph database is needed. A zero-infrastructure **in-memory backend** (numpy cosine + a Python BM25) implements the same `Repository` interface. Tests use it, and you can run it locally without Postgres; the benchmark is reported on both backends.

More detail: [`docs/architecture.md`](docs/architecture.md).

## Retrieval pipeline

```
query
  ├─ lexical retriever: BM25 over tsvector (GIN prefilter, IDF from lexeme stats)  → top 30
  ├─ vector retriever: cosine over pgvector HNSW (ef_search = 100)                  → top 30
  └─ entity retriever (hybrid_entity): linked entities + 1-hop relationships        → top 30
        │
  union of candidates; EVERY active signal is computed for EVERY candidate
  (no "missing = 0" artefacts from one retriever's top-k cutoff)
        │
  per-query min-max normalization of vector and lexical scores
  (entity score is already absolute in [0, 1])
        │
  reranker:  final = α·vector + β·lexical + γ·entity      (or weighted RRF)
        │
  top-k → confidence gate → results + provenance | abstention
```

| Mode | α (vector) | β (lexical) | γ (entity) |
|---|---|---|---|
| `lexical_only` | 0 | 1 | 0 |
| `vector_only` | 1 | 0 | 0 |
| `hybrid` | 0.5 | 0.5 | 0 |
| `hybrid_entity` | 0.4 | 0.4 | 0.2 |

The weights were chosen *a priori* (equal lexical/vector split plus a minority entity share) and were not tuned on the benchmark. They live in [`app/core/config.py`](backend/app/core/config.py) with every other tunable (pool size, normalization, fusion method, thresholds, resolution weights). All of them can be overridden through environment variables (`CF_RETRIEVAL__CANDIDATE_POOL=50`), and the weights can also be overridden per request.

Implementation notes worth knowing:

* **BM25 inside PostgreSQL.** Native `ts_rank_cd` has no IDF term. ContextForge computes Okapi BM25 in SQL instead. A GIN-indexed OR `tsquery` prefilters candidates, term frequencies come from `unnest(tsvector)`, and document frequencies come from a `lexeme_stats` materialized view (`ts_stat`) refreshed after each ingestion batch. Measured: lexical MRR 0.731 with BM25 vs 0.669 with `ts_rank_cd` ([ablation](#ablations)).
* **Contextual chunk headers.** Each chunk is embedded and indexed as `"<document title> > <section path>\n<text>"`, because paragraphs rarely repeat their subject. Provenance still points at the exact original slice: `document.text[char_start:char_end] == chunk.text`, and a test checks this.
* **Structure-aware chunking.** Markdown sections, PDF pages and JSON records start new chunks. Long sections are split on sentence boundaries with one sentence of overlap. JSON chunks keep their JSON pointer (`/issues/3`) and PDF chunks their page numbers.
* **Reranker interface.** The default reranker is deterministic weighted scoring. A cross-encoder would implement the same `Reranker.rerank()` protocol and either replace or blend `final_score`, with component scores still attached to every result.

## Entity resolution

Mentions are extracted by a composite extractor. Precise rules (a technology lexicon; "Project X" codenames; titled or initialled names like "Dr. Chen" and "S. Okafor"; organization names with corporate head words; dates; events; key/value fields in JSON records) take priority over spaCy `en_core_web_sm` NER on overlapping spans. A corpus-level *known-name pass* then finds bare mentions of names discovered elsewhere, such as "Halcyon" once "Project Halcyon" has been seen.

Resolution ([`app/entities/resolution.py`](backend/app/entities/resolution.py)) is deterministic and logs every decision:

1. **Type harmonization.** If extractors disagree on a surface form's type (spaCy: ORG, rule: PROJECT), the type with the highest summed extractor confidence wins.
2. **Normalization and grouping.** Text is case-folded and ASCII-folded, punctuation is stripped, and type-specific affixes are removed (Inc./Corp./LLC; Dr./Ms.; "Project"). So "OpenAI Inc." and "OpenAI" form one surface group.
3. **Within-document coreference.** A partial person mention ("Alice", "Dr. Chen", "A. Chen") attaches to a full name *in the same document* only when exactly one compatible full name occurs there.
4. **Cross-document pairwise matching.** Candidate pairs come from blocking keys (shared tokens, prefixes, surnames, known alias groups) and are scored with a Fellegi–Sunter-style log-linear model with hand-set weights:

   `score = sigmoid(bias + Σ wᵢ·fᵢ)`. The features fᵢ are compact-form equality (`Open AI` ≡ `OpenAI`), known technology alias (`k8s` ≡ `Kubernetes`), Jaro-Winkler similarity, surname match with compatible first name/initial, token subset, context-embedding similarity, document co-occurrence, uniqueness or ambiguity of a short-form completion, and hard conflicts (different surnames or numbers).
5. **Constrained clustering.** Pairs scoring ≥ 0.85 merge in descending order with union-find. A merge that would place two conflicting groups in one cluster (a cannot-link constraint) is refused, which prevents `A~B~C` chaining. Pairs between 0.55 and 0.85 are recorded as *ambiguous* and kept separate.

Ambiguity is preserved rather than guessed away. In the sample corpus, "Alice" in the meeting notes resolves to Alice Chen (the only full-name Alice in that document), "Alice" in the trip report resolves to Alice Moreno, and a bare "Alice" with no disambiguating context would remain its own unresolved entity. `POST /entities/resolve` can also be called with ad-hoc mentions to inspect exactly how a set of names would be clustered, and why.

## Relationships and entity-aware retrieval

Relationships (`works_on`, `affiliated_with`, `uses`, `owns`, `located_in`, `partners_with`, `participates_in`, plus a weak `related_to` co-occurrence) are inferred per sentence or per JSON record from entity-type pairs and trigger phrases. Confidence is graded by how direct the evidence is: trigger between the mentions 0.85; JSON field 0.75; trigger elsewhere in the sentence 0.65; header-context relation 0.52; plain co-occurrence 0.35. Every row stores its `supporting_chunk_id` and evidence sentence. Edges are aggregated with noisy-OR into an in-process adjacency map.

At query time (`hybrid_entity`):

1. **Entity linking**: exact and compact alias matches, fuzzy matches for capitalized spans (typos), and partial names ("Okafor"). An alias shared by several entities splits its confidence between them.
2. **Entity score** per chunk: `1 − (1 − direct_coverage)·(1 − expansion)`. Direct coverage is the confidence-weighted fraction of query entities the chunk mentions. Expansion credits chunks that mention a 1-hop neighbor, weighted by `link_conf × edge_weight × decay`.
3. **Type-hinted expansion.** If the query names the neighbor's type ("the *project* led by…", "the *company* … runs"), the decay is 1.0 instead of 0.6, because that neighbor is the bridge the question is about.
4. **Header context.** Entities named in a chunk's title or section header count as context entities of that chunk. The "Implementation" section of the Lodestar RFC is about Lodestar even though the name never appears in the paragraph.

## Confidence and abstention

Confidence is an *engineered* score in [0, 1], not a calibrated probability. It averages interpretable signals, renormalized over the ones available in each mode:

| Signal | Meaning |
|---|---|
| relevance | Absolute evidence strength: raw cosine mapped from [0.20, 0.60] to [0, 1], and/or IDF-weighted query-term coverage of the best of the top-3 chunks |
| margin | How clearly the top result separates from the runner-up |
| agreement | Overlap of the independent lexical and vector top-5 lists (hybrid modes) |
| support | Number of distinct source documents among the strong results |
| entity_consistency | Fraction of linked query entities present in the top results (`hybrid_entity`) |

**Hard gates** abstain regardless of the aggregate:

* *weak evidence*: no chunk has cosine ≥ 0.25 **and** none covers ≥ 34% of the query's informative terms.
* *unknown entity* (`hybrid_entity`): the query names a person, organization, project, event or location whose terms occur nowhere in the corpus ("What did *Bob Smith* present…", "…*Project Nimbus*…").

An abstention returns `status: "insufficient_evidence"`, the fixed message, the confidence breakdown, and human-readable reasons. The rejected candidates are included only with `include_evidence_on_abstain: true`. The optional answer generator (`POST /answer`) is **never called** when the gate fails, and every citation marker it emits is validated against the evidence it was given.

The threshold (0.45) was set a priori. [`eval/results/confidence_sweep.md`](eval/results/confidence_sweep.md) sweeps it on the dev split and reports the test split. With only 6 unanswerable dev queries, the dev-optimal threshold is noise-prone, so the default was not changed.

## Evaluation methodology and results

**Dataset** ([`eval/datasets/queries.json`](eval/datasets/queries.json)): 100 hand-labelled queries over the bundled corpus, stratified by what they probe:

| Category | n | Example |
|---|---|---|
| lexical | 20 | "What does error QL-5031 mean?" |
| semantic | 25 | "How long do new hires wait before they start carrying the pager?" (paraphrase of "join the on-call rotation after their sixth week") |
| entity | 22 | "What is A. Chen responsible for…", "What is QuillonLabs' net revenue retention?" |
| relational | 15 | "What solver library does Samuel Okafor's project use?" |
| unanswerable | 18 | 8 in-domain near misses, 5 unknown entities, 5 out-of-domain |

Relevance is labelled with **evidence strings**, not chunk IDs. A chunk is relevant if it belongs to a listed source file and contains an evidence string. This keeps labels valid when chunking changes, and the harness **fails loudly** if an evidence string matches no chunk, so a stale label never counts silently as a miss. Every third query is in the `dev` split; results are reported per split.

**Metrics**: Recall@k (fraction of relevant chunks in the top k), Hit@k, MRR, nDCG@10, source accuracy@1, mean/p50/p95 end-to-end latency after warm-up, and abstention precision/recall/false-abstention rate. Entity resolution is scored with pairwise precision/recall/F1 against [`eval/datasets/entity_resolution_gold.json`](eval/datasets/entity_resolution_gold.json) and compared with two string-matching baselines.

### MRR / Hit@5 by category (PostgreSQL backend)

| Category | n | Lexical | Vector | Hybrid | Hybrid + Entity |
|---|---|---|---|---|---|
| lexical | 20 | 1.000 / 1.000 | 0.738 / 0.850 | 0.942 / 1.000 | 0.942 / 1.000 |
| semantic | 25 | 0.609 / 0.680 | 0.796 / 0.920 | 0.752 / 0.920 | 0.749 / 0.920 |
| entity | 22 | 0.807 / 0.909 | 0.670 / 0.864 | 0.849 / 0.955 | 0.849 / 0.955 |
| relational | 15 | 0.464 / 0.733 | 0.276 / 0.400 | 0.349 / 0.800 | 0.357 / 0.933 |

Each signal wins where you would expect: lexical on identifiers, vector on paraphrases, and the fused modes are the most robust overall. Fusion costs a little first-rank precision on the categories where one signal is perfect (lexical 1.000 → 0.942 MRR).

### Abstention (all 100 queries; positive class = unanswerable)

| Method | Abstain precision | Abstain recall | False-abstention rate |
|---|---|---|---|
| Lexical | 0.294 | 0.556 | 0.293 |
| Vector | 0.269 | 0.389 | 0.232 |
| Hybrid | 0.462 | 0.333 | 0.085 |
| Hybrid + Entity | 0.688 | 0.611 | 0.061 |

### Entity resolution (144 gold mentions, 21 gold entities)

| Method | Precision | Recall | F1 | Predicted clusters |
|---|---|---|---|---|
| Exact string match | 0.993 | 0.600 | 0.748 | 42 |
| Normalized name match | 0.994 | 0.734 | 0.844 | 37 |
| ContextForge resolver | 1.000 | 1.000 | 1.000 | 21 |

### Ablations

All ablations reindex and rerun the full benchmark with one setting changed (`scripts/reproduce_results.sh`; table generated into [`eval/results/comparison.md`](eval/results/comparison.md)). Hybrid + Entity rows unless noted:

| Configuration | Recall@5 | MRR | Relational Hit@5 |
|---|---|---|---|
| Default | 0.927 | 0.751 | 0.933 |
| No graph expansion (`MAX_HOPS=0`, direct entity matches only) | 0.872 | 0.746 | 0.600 |
| No header-context entities | 0.890 | 0.744 | 0.733 |
| No header context *and* no type hints (≈ first development run) | 0.884 | 0.743 | 0.667 |
| No type-hinted expansion | 0.896 | 0.746 | 0.733 |
| Weighted RRF instead of score blending | 0.884 | 0.685 | 0.733 |
| `ts_rank_cd` instead of BM25 (lexical-only row: MRR 0.669 vs 0.731) | 0.908 | 0.740 | 0.933 |
| In-memory backend (Python BM25 + numpy) | 0.927 | 0.756 | 0.933 |

The most important finding: **entity matching alone hurts.** Without relationship expansion, Hybrid + Entity (0.872) falls *below* plain hybrid (0.902), because chunks that merely mention the queried person crowd out the answer. The relationship graph is what makes the entity signal useful. Header context and type hints were added after error analysis on the first full benchmark run, which showed exactly this regression. That history, and the bugs the evaluation caught, are written up in [`docs/evaluation.md`](docs/evaluation.md). Because these components were designed with this dataset in view, the per-split table in [`latest.md`](eval/results/latest.md) is the fairer read: test-split Recall@5 is 0.927 for Hybrid + Entity vs 0.909 for Hybrid.

## API examples

Interactive docs at `http://localhost:8000/docs`. Endpoints:

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Backend, model, extractor, index counts |
| POST | `/documents/upload` | Multipart upload of TXT/MD/PDF/JSON (idempotent per file content) |
| GET | `/documents`, `/documents/{id}` | List documents / document with chunks and offsets |
| DELETE | `/documents/{id}` | Remove a document and rebuild the entity graph |
| POST | `/search` | Retrieval in one of the four modes, with confidence gate |
| POST | `/answer` | Optional cited answer (extractive or Ollama); skipped on abstention |
| GET | `/entities`, `/entities/{id}`, `/entities/graph` | Resolved entities, mentions with provenance, relationships with evidence |
| POST | `/entities/resolve` | Rebuild resolution, or resolve ad-hoc mentions and return the decision log |
| POST | `/evaluate` | Run a dataset from `eval/datasets/` against the current index |

```bash
curl -s -X POST localhost:8000/search -H 'Content-Type: application/json' \
  -d '{"query": "Which runtime does Samantha Reyes'"'"'s project use on Android tablets?", "mode": "hybrid_entity", "top_k": 3}'
```

```jsonc
{
  "query": "Which runtime does Samantha Reyes's project use on Android tablets?",
  "mode": "hybrid_entity",
  "status": "success",
  "confidence": 0.6275,
  "results": [
    {
      "rank": 1,
      "text": "run_id: WB-032\ndevice: Reference mid-range Android tablet\nruntime: ONNX Runtime\nquantization: int4 ...",
      "source": "wren_benchmarks.json",
      "document_id": "doc_6a9238b0b2f75f28",
      "chunk_id": "doc_6a9238b0b2f75f28:0002",
      "location": {"json_path": "/runs/1", "record_index": 1},
      "retrieval_mode": "hybrid_entity",
      "vector_score": 1.0, "lexical_score": 0.9487, "entity_score": 0.964, "final_score": 0.9723,
      "vector_similarity": 0.5513,
      "retrieved_by": ["entity", "lexical", "vector"],
      "matched_entities": ["Android", "Project Wren (via Samantha Reyes)", "ONNX Runtime (via Android)"]
    }
  ],
  "confidence_detail": {"score": 0.6275, "threshold": 0.45, "reasons": [], "gates_triggered": [],
    "signals": {"vector_similarity": 0.5513, "lexical_coverage": 0.5068, "relevance": 0.6925, "margin": 0.0089,
                "agreement": 0.7, "support": 1.0, "entity_consistency": 0.5}},
  "query_entities": [{"canonical_name": "Samantha Reyes", "method": "exact", "...": "..."}, {"canonical_name": "Android", "method": "exact", "...": "..."}],
  "timings_ms": {"lexical_ms": 2.5, "embed_ms": 20.8, "vector_ms": 1.4, "entity_ms": 9.7, "scoring_ms": 2.9, "rerank_ms": 0.1, "confidence_ms": 1.8, "total_ms": 39.3}
}
```

An abstention (`"query": "What did Bob Smith present at the Quillon Offsite?"`):

```jsonc
{
  "status": "insufficient_evidence",
  "message": "ContextForge could not find sufficiently strong supporting evidence.",
  "confidence": 0.5264,
  "results": [],
  "confidence_detail": {
    "gates_triggered": ["unknown_entity"],
    "reasons": ["query names entities that do not occur anywhere in the corpus: Bob Smith"]
  },
  "unknown_entities": ["Bob Smith"]
}
```

Ad-hoc entity resolution:

```bash
curl -s -X POST localhost:8000/entities/resolve -H 'Content-Type: application/json' -d '{"mentions": [
  {"surface": "Sam Altman", "entity_type": "PERSON", "document": "a"}, {"surface": "S. Altman", "entity_type": "PERSON", "document": "b"},
  {"surface": "Sam", "entity_type": "PERSON", "document": "a"}, {"surface": "OpenAI", "entity_type": "ORGANIZATION", "document": "a"},
  {"surface": "Open AI", "entity_type": "ORGANIZATION", "document": "b"}, {"surface": "OpenAI Inc.", "entity_type": "ORGANIZATION", "document": "c"}]}'
# clusters: [Sam Altman, S. Altman, Sam], [OpenAI, Open AI, OpenAI Inc.]
# decisions: "Sam ~ Sam Altman" within-document coreference; "Open AI ~ OpenAI" 0.994; "S. Altman ~ Sam Altman" 0.970 ...
```

(Example outputs were captured from the running API on the sample corpus and abridged with `...`.)

## Screenshots

| Hybrid + Entity search with score breakdown | Abstention with reasons | Entity view (aliases, relationships, evidence) |
|---|---|---|
| ![search](docs/images/search_success.png) | ![abstain](docs/images/search_abstain.png) | ![entity](docs/images/entity_view.png) |

The screenshots are illustrative; the benchmark above is the evidence of quality.

## Installation and running

Requirements: Python 3.11+, Node 18+ (UI only), and optionally PostgreSQL 16 with pgvector.

```bash
# 1. Backend dependencies (from the repo root)
python -m venv .venv && source .venv/bin/activate
pip install -e "backend[nlp,dev]"
pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl   # optional; rules-only otherwise
python scripts/download_model.py        # all-MiniLM-L6-v2 ONNX export (~90 MB, checksum-verified); also auto-downloads on first use
```

**Quickest start (no database):** the in-memory backend indexes `sample_data/` at startup.

```bash
cd backend
CF_STORAGE_BACKEND=memory CF_BOOTSTRAP_DIR=../sample_data uvicorn app.main:app --port 8000
```

**PostgreSQL + pgvector:**

```bash
docker compose up -d db                                    # or: sudo scripts/start_local_postgres.sh (Debian/Ubuntu, no Docker)
export CF_STORAGE_BACKEND=postgres CF_DATABASE_URL=postgresql://contextforge:contextforge@localhost:5432/contextforge
python scripts/ingest_sample_data.py --reset               # index sample_data/ into Postgres
cd backend && uvicorn app.main:app --port 8000
```

**Frontend:**

```bash
cd frontend && npm install && npm run dev                  # http://localhost:5173 (API URL via VITE_API_URL, default http://localhost:8000)
```

**Everything in Docker:** `docker compose up --build` (Postgres + API + UI). The compose file and Dockerfiles are included but were **not executed** in the development environment, which had no Docker daemon. Every component they run was exercised natively.

**Tests:**

```bash
cd backend
pytest                                                              # unit + API tests (hashing embedder, rule extractor; no network)
CF_TEST_DATABASE_URL=postgresql://contextforge:contextforge@localhost:5432/contextforge_test pytest -m postgres   # pgvector backend
pytest -m model                                                     # real MiniLM ONNX model checks (skipped if not downloaded)
```

**Benchmark:**

```bash
python scripts/run_eval.py --backend postgres      # rebuilds the index, writes eval/results/latest.{json,md}
python scripts/run_eval.py --backend memory --out eval/results/memory_backend.json
scripts/reproduce_results.sh                       # all runs + ablations + threshold sweep + comparison table
```

**Optional local generation** (retrieval never depends on it): run Ollama, then set `CF_GENERATION__BACKEND=ollama` and `CF_GENERATION__OLLAMA_MODEL=llama3.2:3b`, and use `POST /answer`. Without it, `/answer` uses an extractive generator that only copies cited evidence sentences.

Configuration reference: [`.env.example`](.env.example) and [`backend/app/core/config.py`](backend/app/core/config.py).

## Repository layout

```
backend/
  app/
    api/            FastAPI routers (health, documents, search/answer, entities, evaluate)
    core/           config (all tunables), text analysis, dependency container
    db/             Repository protocol; PostgreSQL+pgvector and in-memory implementations; schema.sql
    embeddings/     ONNX MiniLM (default), sentence-transformers (optional), hashing (tests)
    entities/       extraction, normalization, resolution, relationships, linking, graph, service
    evaluation/     dataset + relevance resolution, metrics, runner, report, ER evaluation, CLI
    generation/     optional answer generation: extractive baseline, Ollama, citation checks
    ingestion/      parsers (TXT/MD/PDF/JSON), structure-aware chunker, ingestion pipeline
    models/         domain dataclasses
    retrieval/      BM25, normalization, entity scorer, reranker, confidence, engine
    schemas/        Pydantic API models
  tests/            111 pytest tests (unit, API, PostgreSQL, real-model)
frontend/           React + TypeScript (Vite) UI
eval/datasets/      queries.json (100 queries), entity_resolution_gold.json
eval/results/       latest.{json,md}, memory_backend, ablation_*, confidence_sweep.md, comparison.md
sample_data/        29 synthetic documents (MD, TXT, PDF, JSON)
scripts/            download model, ingest, run/compare/sweep evaluations, regenerate PDFs, local Postgres
docs/               architecture.md, evaluation.md, screenshots
```

## Limitations

* **Small, synthetic, self-authored benchmark.** 29 documents and 100 queries written for this project, including the queries. The categories are deliberately stratified to exercise each signal, so the *mix* determines the averages; read the per-category tables. Results will not transfer as-is to real corpora, and nothing here has been run on a public benchmark (e.g. BEIR).
* **Some components were designed after error analysis on this dataset** (header context, type-hinted expansion). Dev/test numbers are reported separately, but the test split is not truly held out from the design process.
* **Relational queries remain weak on ranking.** Entity expansion pulls answers into the top 5 (Hit@5 0.933) but rarely to rank 1 (MRR 0.357). Chunks about the queried person still outrank the chunk about their project. For example, for "Which database does the project led by Alice Chen use?" the storage paragraph is retrieved but ranks below Alice Chen's bio chunks.
* **Abstention only catches evidence-level failures.** It reliably abstains on out-of-domain and unknown-entity questions. It misses 7 of 8 in-domain near misses (e.g. "What was revenue in Q4 2025?" when only Q3 is reported); the one it catches names a location that appears nowhere in the corpus. Retrieval signals cannot tell that the topic is present but the specific fact is not; that needs answer-level verification.
* **Confidence is engineered, not calibrated.** The signal weights and thresholds are hand-set and interpretable, not probabilities.
* **Heuristic entity resolution and noisy relationships.** Hand-set match weights; trigger-phrase relation typing; spaCy's small model mislabels some spans (e.g. "Operations Research" as an organization). The perfect ER score is on a small gold set built from the same synthetic corpus.
* **Global re-resolution on every ingest.** Entity resolution and relationship extraction rerun over the whole corpus after each upload batch (about 60 ms here; O(corpus) in general). Incremental resolution is future work. The entity graph is cached per API process, so a process that did not perform an out-of-band reindex needs a restart or `POST /entities/resolve`.
* **Small-scale latency numbers.** Measured on a 110-chunk corpus on a 4-vCPU container; they show relative overheads, not production capacity.
* **Not exercised here:** the `sentence-transformers` backend (HuggingFace was unreachable from the build environment; the ONNX backend uses the same weights, and its parity was checked against the published reference similarity, 0.7553), live Ollama generation (tested against a mocked HTTP API), and Docker Compose.

## Future work

* Cross-encoder reranker behind the existing `Reranker` protocol, compared on the same harness.
* Query decomposition for relational questions: score the residual query ("which database does … use") separately from the bridge entity.
* Answer-level verification (NLI or LLM judge) so abstention can catch in-domain near misses.
* Learn the entity-resolution weights by logistic regression from labelled pairs, and the confidence model by calibration (e.g. isotonic) on a larger dev set.
* Incremental entity resolution; graph queries in SQL (recursive CTEs) for multi-hop expansion at scale.
* Evaluate on public retrieval benchmarks (BEIR subsets) and a real document collection.

## License

MIT, see [LICENSE](LICENSE).
