"""Index a directory of documents (default: sample_data/) using the configured backend.

    python scripts/ingest_sample_data.py [--data DIR] [--reset]

With the memory backend, set CF_MEMORY_SNAPSHOT_PATH so the API server can load
the resulting index; with the postgres backend the index lives in the database.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from app.core.config import Settings
from app.core.container import Container

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "sample_data")
    parser.add_argument("--reset", action="store_true", help="delete the existing index first")
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING)

    container = Container(Settings())
    if args.reset:
        container.reset()
    results = container.ingest_directory(args.data)
    for r in results:
        print(f"{r.status:9s} {r.filename:45s} chunks={r.chunks:3d} mentions={r.mentions:3d} {r.error or ''}")
    print(f"\nBackend: {container.repo.backend_name}")
    print(f"Index:   {container.repo.counts()}")
    if container.last_rebuild:
        print(f"Entities: {container.last_rebuild.stats}")


if __name__ == "__main__":
    main()
