# src/clv_uplift/api/routers/info.py
#
# GET /api/v1/model-info - population-level causal findings + certification status.
# Read-only; powers the Streamlit "model context" panel so individual CATEs are framed
# against the population ATE and the tiered findings.

from typing import Optional

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from ..dependencies import require_bundle

router = APIRouter()


def _as_float(value, default: float) -> float:
    """
    Coerce a metadata value to float. Training writes some metadata as display strings
    (e.g. confounding_correction = '+0.229'), so accept str or number and fall back to
    the default rather than 422-ing on a formatting quirk.
    """
    if value is None:
        return default
    try:
        return float(str(value).strip().lstrip("+"))
    except (TypeError, ValueError):
        return default


class ModelInfoResponse(BaseModel):
    model_name: str
    version: str
    ate_point: float
    ate_ci_lower: float
    ate_ci_upper: float
    e_value_point: float
    e_value_ci: float
    rate_test_passed: bool
    surrogate_valid: bool
    surrogate_r2: float
    surrogate_r2_threshold: float
    tier_ate: int
    tier_heterogeneity: int
    confounding_correction: float
    naive_association: float
    positivity_strain: str
    metadata: dict


@router.get("/model-info", response_model=ModelInfoResponse, summary="Population causal findings")
async def get_model_info(bundle=Depends(require_bundle)):
    """Population-level causal findings and certification status (503 if no bundle)."""
    md = bundle.metadata or {}
    return ModelInfoResponse(
        model_name=bundle.model_name,
        version=bundle.version,
        ate_point=float(bundle.ate_point),
        ate_ci_lower=float(bundle.ate_ci_lower),
        ate_ci_upper=float(bundle.ate_ci_upper),
        e_value_point=float(bundle.e_value_point),
        e_value_ci=float(bundle.e_value_ci),
        rate_test_passed=bool(bundle.rate_test_passed),
        surrogate_valid=bool(bundle.surrogate_valid),
        surrogate_r2=float(bundle.surrogate_r2),
        surrogate_r2_threshold=float(bundle.surrogate_r2_threshold),
        tier_ate=int(md.get("tier_ate", 2)),
        tier_heterogeneity=int(md.get("tier_heterogeneity", 3)),
        confounding_correction=_as_float(md.get("confounding_correction"), 0.229),
        naive_association=_as_float(md.get("naive_association"), 0.334),
        positivity_strain=str(md.get("positivity_strain", "24.4% extreme propensity")),
        metadata=md,
    )