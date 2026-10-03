"""all-MiniLM-L6-v2 via ONNX Runtime.

This reproduces the sentence-transformers inference pipeline for this model
(WordPiece tokenization -> BERT encoder -> attention-masked mean pooling ->
L2 normalization) without requiring PyTorch. The ONNX export is the one
published for Chroma (same weights as ``sentence-transformers/all-MiniLM-L6-v2``)
and is downloaded once into ``Settings.model_dir``.
"""

from __future__ import annotations

import hashlib
import logging
import shutil
import tarfile
import tempfile
import threading
import urllib.request
from pathlib import Path

import numpy as np

from app.embeddings.base import l2_normalize

logger = logging.getLogger(__name__)

MODEL_URL = "https://chroma-onnx-models.s3.amazonaws.com/all-MiniLM-L6-v2/onnx.tar.gz"
MODEL_SHA256 = "913d7300ceae3b2dbc2c50d1de4baacab4be7b9380491c27fab7418616a16ec3"
REQUIRED_FILES = ("model.onnx", "tokenizer.json")
MAX_SEQ_LENGTH = 256  # sentence-transformers' max_seq_length for this model


class ModelNotAvailableError(RuntimeError):
    pass


def model_files_present(model_dir: Path) -> bool:
    return all((model_dir / name).exists() for name in REQUIRED_FILES)


def download_model(model_dir: Path, url: str = MODEL_URL, sha256: str = MODEL_SHA256) -> Path:
    """Download and verify the ONNX export, then unpack it into ``model_dir``."""
    model_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        archive = Path(tmp) / "onnx.tar.gz"
        logger.info("Downloading embedding model from %s", url)
        with urllib.request.urlopen(url, timeout=120) as response, archive.open("wb") as fh:
            shutil.copyfileobj(response, fh)
        digest = hashlib.sha256(archive.read_bytes()).hexdigest()
        if digest != sha256:
            raise ModelNotAvailableError(f"Checksum mismatch for model archive: {digest}")
        with tarfile.open(archive) as tar:
            for member in tar.getmembers():
                name = Path(member.name).name
                if member.isfile() and name:
                    extracted = tar.extractfile(member)
                    if extracted is not None:
                        (model_dir / name).write_bytes(extracted.read())
    if not model_files_present(model_dir):
        raise ModelNotAvailableError(f"Model archive did not contain {REQUIRED_FILES}")
    return model_dir


class OnnxMiniLMEmbedder:
    dim = 384

    def __init__(self, model_dir: Path, auto_download: bool = True, batch_size: int = 32) -> None:
        if not model_files_present(model_dir):
            if not auto_download:
                raise ModelNotAvailableError(
                    f"ONNX model not found in {model_dir}. Run `python scripts/download_model.py`."
                )
            download_model(model_dir)

        import onnxruntime as ort
        from tokenizers import Tokenizer

        self.name = "all-MiniLM-L6-v2 (onnx)"
        self.batch_size = batch_size
        self._tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
        self._tokenizer.enable_truncation(max_length=MAX_SEQ_LENGTH)
        self._tokenizer.enable_padding(pad_id=0, pad_token="[PAD]")
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self._session = ort.InferenceSession(
            str(model_dir / "model.onnx"), options, providers=["CPUExecutionProvider"]
        )
        self._input_names = {i.name for i in self._session.get_inputs()}
        self._lock = threading.Lock()

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        outputs = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            with self._lock:  # tokenizer padding state is shared
                encodings = self._tokenizer.encode_batch(batch)
            input_ids = np.array([e.ids for e in encodings], dtype=np.int64)
            attention = np.array([e.attention_mask for e in encodings], dtype=np.int64)
            feeds = {"input_ids": input_ids, "attention_mask": attention}
            if "token_type_ids" in self._input_names:
                feeds["token_type_ids"] = np.zeros_like(input_ids)
            hidden = self._session.run(None, feeds)[0]  # (batch, seq, 384)
            mask = attention[..., None].astype(np.float32)
            pooled = (hidden * mask).sum(axis=1) / np.clip(mask.sum(axis=1), 1e-9, None)
            outputs.append(pooled)
        return l2_normalize(np.vstack(outputs))
