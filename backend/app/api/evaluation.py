from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException

from app.api.deps import get_container
from app.core.container import Container
from app.evaluation.dataset import DatasetIntegrityError, load_dataset
from app.evaluation.report import render_markdown
from app.evaluation.runner import run_evaluation
from app.schemas.api import EvaluateRequest, EvaluateResponse

router = APIRouter(tags=["evaluation"])

DATASET_DIR = Path(__file__).resolve().parents[3] / "eval" / "datasets"


@router.post("/evaluate", response_model=EvaluateResponse)
def evaluate(req: EvaluateRequest, container: Container = Depends(get_container)) -> EvaluateResponse:
    """Run the benchmark against the *current* index (it must contain the dataset's sources)."""
    path = (DATASET_DIR / req.dataset).resolve()
    if path.parent != DATASET_DIR.resolve() or not path.exists():
        raise HTTPException(404, f"dataset '{req.dataset}' not found in eval/datasets/")
    try:
        report = run_evaluation(container, load_dataset(path), req.modes, req.k_values)
    except DatasetIntegrityError as exc:
        raise HTTPException(409, str(exc)) from exc
    return EvaluateResponse(
        dataset=req.dataset,
        num_queries=report["meta"]["dataset"]["queries"],
        summary=report["summary"],
        markdown=render_markdown(report),
    )
