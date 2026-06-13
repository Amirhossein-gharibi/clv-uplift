# src/clv_uplift/api/routers/predict.py
#
#   POST /api/v1/score   - DummyModel engagement score (UNCHANGED; frozen tests).
#   POST /api/v1/predict - real CausalForestDML CATE from raw RFM (Phase 3 / 3.5).

import logging

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException

from ..schemas import CustomerFeatures, CATEFeatures, ScoreResponse, PredictResponse
from ..dependencies import get_model, DummyModel, require_bundle

logger = logging.getLogger(__name__)
router = APIRouter()


EXAMPLE_INPUT = CustomerFeatures(
    customer_id="12345",
    recency_days=14,
    frequency=23,
    monetary_value=1842.50,
    clv_segment="Champions",
)


def _features_vector(features: CATEFeatures, bundle) -> np.ndarray:
    """
    Convert raw CATEFeatures to the seven-feature model vector via the bundle's fitted
    RFMBinner (training-distribution modal lookup). Returns a (1, 7) array in FEATURE_COLS
    order. Quartile derivation lives inside the bundle - correct and retrain-safe; no
    caller knowledge of cutpoints, no hardcoded thresholds in the API.
    """
    raw = pd.DataFrame([{
        "recency_days": features.recency_days,
        "frequency": features.frequency,
        "monetary_value": features.monetary_value,
        "cancel_rate": features.cancel_rate,
    }])
    return bundle.rfm_binner.transform(raw).values


@router.post(
    "/score",
    response_model=ScoreResponse,
    summary="Score a customer (scaffold engagement score)",
    description="Accepts a customer's RFM features and returns an engagement score "
                "(DummyModel scaffold).",
)
async def score_customer(
    features: CustomerFeatures,
    model: DummyModel = Depends(get_model),
):
    """Original scaffold endpoint - unchanged."""
    try:
        scores = model.predict(
            recency_days=features.recency_days,
            frequency=features.frequency,
            monetary_value=features.monetary_value,
        )
    except Exception as exc:
        logger.exception("Prediction failed for customer %s", features.customer_id)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(exc)}")

    return ScoreResponse(
        customer_id=features.customer_id,
        model_version=model.version,
        **scores,
    )


@router.post(
    "/predict",
    response_model=PredictResponse,
    summary="Estimate a customer's CATE (CausalForestDML)",
    description="Returns the individual CATE and 95% CI from the trained causal model. "
                "When heterogeneity was not certified by the RATE test, the estimate is "
                "returned but flagged not-for-targeting.",
)
async def predict_cate(
    features: CATEFeatures,
    bundle=Depends(require_bundle),
):
    """Real causal CATE endpoint. 503 if no trained bundle exists yet."""
    try:
        X = _features_vector(features, bundle)
        cate = float(np.asarray(bundle.causal_forest.effect(X)).reshape(-1)[0])
        lb, ub = bundle.causal_forest.effect_interval(X, alpha=0.05)
        cate_lo = float(np.asarray(lb).reshape(-1)[0])
        cate_hi = float(np.asarray(ub).reshape(-1)[0])
    except Exception as exc:
        logger.exception("CATE prediction failed for customer %s", features.customer_id)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(exc)}")

    if bundle.rate_test_passed:
        targeting_certified = True
        confidence = "high"
        recommended_action = (
            "Heterogeneity certified (RATE test passed). CATE may inform targeting."
        )
    else:
        targeting_certified = False
        confidence = "not_certified"
        recommended_action = (
            "Individual CATE estimate available but heterogeneity not statistically "
            "confirmed (RATE Tier-3 null). Do not use this score for targeting decisions. "
            f"Population ATE = {bundle.ate_point:+.3f} (E-value {bundle.e_value_point:.2f}) "
            "is structurally valid but individual-level ranking is unreliable given "
            "positivity constraints."
        )

    return PredictResponse(
        customer_id=features.customer_id,
        cate_estimate=round(cate, 6),
        cate_ci_lower=round(cate_lo, 6),
        cate_ci_upper=round(cate_hi, 6),
        targeting_certified=targeting_certified,
        confidence=confidence,
        recommended_action=recommended_action,
        model_version=bundle.version,
    )


@router.get(
    "/docs-demo",
    response_model=CustomerFeatures,
    summary="Get an example request payload (scaffold /score)",
)
async def get_example_input():
    return EXAMPLE_INPUT