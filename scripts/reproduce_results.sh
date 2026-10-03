#!/usr/bin/env bash
# Regenerate every file in eval/results/ from scratch.
# Requires PostgreSQL + pgvector at CF_DATABASE_URL (default: postgresql://contextforge:contextforge@localhost:5432/contextforge)
# and the ONNX embedding model (downloaded automatically on first use).
set -euo pipefail
cd "$(dirname "$0")/.."
R=eval/results
mkdir -p "$R"

run() {  # run <output-name> [extra args...]
  local out=$1; shift
  echo ">> $out"
  python scripts/run_eval.py "$@" --out "$R/$out.json" > /dev/null
}

# Ablations (summary only): each one reindexes with the given settings.
CF_RETRIEVAL__LEXICAL_SCORER=ts_rank    run ablation_ts_rank             --backend postgres --summary-only
CF_RETRIEVAL__FUSION=rrf                run ablation_rrf_fusion          --backend postgres --summary-only
CF_ENTITY_RETRIEVAL__HINTED_HOP_DECAY=0.6 run ablation_no_type_hints     --backend postgres --summary-only
CF_ENTITY_RETRIEVAL__HEADER_CONTEXT=false run ablation_no_header_context --backend postgres --summary-only
CF_ENTITY_RETRIEVAL__HEADER_CONTEXT=false CF_ENTITY_RETRIEVAL__HINTED_HOP_DECAY=0.6 \
                                        run ablation_no_header_no_hints  --backend postgres --summary-only
CF_ENTITY_RETRIEVAL__MAX_HOPS=0         run ablation_no_graph_expansion  --backend postgres --summary-only

# Same configuration on the in-memory backend (BM25 + numpy cosine).
run memory_backend --backend memory --summary-only

# Primary run last, so the database is left indexed with the default configuration.
run latest --backend postgres

python scripts/sweep_confidence.py "$R/latest.json" --out "$R/confidence_sweep.md" > /dev/null
python scripts/compare_results.py "$R/latest.json" "$R/memory_backend.json" "$R"/ablation_*.json > "$R/comparison.md"
echo "Done: $(ls "$R" | wc -l) files in $R"
