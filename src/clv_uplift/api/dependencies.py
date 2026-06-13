# src/clv_uplift/api/dependencies.py
#
# This module owns the model loading logic.
# Endpoints never import the model directly - they declare a dependency
# on get_model() / get_bundle(), and FastAPI injects the result automatically.

import math
import pickle
import logging
from functools import lru_cache

from clv_uplift.config import MODEL_PATH

logger = logging.getLogger(__name__)


class DummyModel:
    """
    A stand-in for a simple engagement-scoring model (Chapters 2-3 scaffold).
    Kept intact and serving /api/v1/score. The real causal pipeline is served
    separately via get_bundle() + /api/v1/predict + /api/v1/explain.
    """

    version = "dummy-0.1"

    def predict(self, recency_days: int, frequency: int, monetary_value: float) -> dict:
        recency_score = 1.0 / max(recency_days, 1)
        frequency_score = math.log1p(frequency) / math.log1p(1000)
        composite_score = 0.6 * recency_score + 0.4 * frequency_score
        return {
            "recency_score":   round(recency_score,   6),
            "frequency_score": round(frequency_score, 4),
            "composite_score": round(composite_score, 4),
        }


@lru_cache(maxsize=1)
def get_model() -> DummyModel:
    """Load the dummy scaffold model once and cache it (Chapters 2-3). Untouched."""
    logger.info("Loading DummyModel into memory (once per process)...")
    return DummyModel()


@lru_cache(maxsize=1)
def get_bundle():
    """
    Load the real ServingBundle (Phase 2 causal pipeline) once and cache it.

    Independent of get_model(): the dummy scaffold and the causal model coexist.
    Raises RuntimeError if MODEL_PATH does not exist yet (training not run); the
    lifespan warmup catches this so the service still starts and serves /score, and
    the /predict + /explain routes convert it to a 503 at request time.

    Importing ServingBundle here (not at module top) keeps the import cost off the
    /score path and avoids importing heavy modeling deps unless the bundle is used.
    """
    from clv_uplift.models.serving import ServingBundle  # noqa: F401 (needed for unpickling)

    if not MODEL_PATH.exists():
        raise RuntimeError(
            f"No trained bundle at {MODEL_PATH}. "
            f"Run notebooks/02_uplift_training.py first to create it."
        )
    logger.info("Loading ServingBundle from %s ...", MODEL_PATH)
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


def require_bundle():
    """
    FastAPI dependency for routes that need the real bundle. Converts a missing
    artifact into a clean 503 (service up, model not trained yet) rather than a 500.
    """
    from fastapi import HTTPException
    try:
        return get_bundle()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))