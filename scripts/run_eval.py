"""Rebuild the index from sample_data/ and run the retrieval benchmark.

    python scripts/run_eval.py --backend postgres            # writes eval/results/latest.{json,md}
    python scripts/run_eval.py --backend memory --out eval/results/memory_backend.json

See ``python scripts/run_eval.py --help`` for all options.
"""

from app.evaluation.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
