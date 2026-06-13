# src/clv_uplift/models/serving.py
"""
ServingBundle - the minimal, pickle-stable contract between training and the API.

Deliberately separate from Phase 2.1's UpliftModelBundle (a rich in-memory TRAINING
artifact). ServingBundle holds only what /predict and /explain need. The training script
(notebooks/02_uplift_training.py) extracts these fields and pickles a ServingBundle to
MODEL_PATH; the API's get_bundle() loads it.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ServingBundle:
    """Minimal serving contract. Only what /predict and /explain need."""
    # Identity
    model_name: str
    version: str

    # The one model the API calls at serving time (fitted CausalForestDML).
    causal_forest: object

    # Preprocessing - fitted on TRAINING data, travels with the model so raw RFM input
    # is scored server-side with the exact training-distribution mapping (retrain-safe).
    rfm_binner: object

    # Feature contract - order matters for inference.
    feature_cols: list

    # Certification flags (Phase 2.2 / 2.3 gates).
    rate_test_passed: bool          # False in the current run
    surrogate_valid: bool           # False in the current run
    surrogate_r2: float             # 0.8542
    surrogate_r2_threshold: float   # 0.90

    # SHAP surrogate - present even when surrogate_valid=False so /explain can report R^2.
    shap_surrogate: object

    # Population-level results for /predict context.
    ate_point: float                # +0.105
    ate_ci_lower: float
    ate_ci_upper: float
    e_value_point: float            # 1.838
    e_value_ci: float               # 1.000

    metadata: dict = field(default_factory=dict)
import pickle
from pathlib import Path
from typing import Optional


def load_bundle(path: Optional[Path] = None) -> "ServingBundle":
    """
    Load a pickled ServingBundle from MODEL_PATH (or an explicit path). Raises RuntimeError
    if the artifact is missing, with the same message the API uses. ServingBundle is defined
    in this module, so unpickling resolves the class correctly.
    """
    from clv_uplift.config import MODEL_PATH
    target = Path(path) if path is not None else MODEL_PATH
    if not target.exists():
        raise RuntimeError(
            f"No bundle at {target}. Run notebooks/02_uplift_training.py first."
        )
    with open(target, "rb") as f:
        return pickle.load(f)