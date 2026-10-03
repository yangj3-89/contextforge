"""Download the all-MiniLM-L6-v2 ONNX export used by the default embedding backend.

    python scripts/download_model.py [--model-dir PATH]

The archive is verified against a pinned SHA-256 before it is unpacked.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from app.core.config import Settings
from app.embeddings.onnx_minilm import MODEL_URL, download_model, model_files_present


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=Settings().model_dir)
    args = parser.parse_args()
    if model_files_present(args.model_dir):
        print(f"Model already present in {args.model_dir}")
        return
    print(f"Downloading {MODEL_URL} -> {args.model_dir}")
    download_model(args.model_dir)
    print("Done.")


if __name__ == "__main__":
    main()
