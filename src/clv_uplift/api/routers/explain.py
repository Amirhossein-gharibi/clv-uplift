# src/clv_uplift/api/routers/explain.py
#
# POST /api/v1/explain - SHAP feature attributions for a customer's CATE.
# When the surrogate fidelity gate fired (surrogate_valid=False), returns HTTP 200 with a
# structured degraded response (NOT 4xx/5xx): the gate firing is expected behavior.

import logging

import numpy as np
from fastapi import APIRouter, Depends, HTTPException

from ..schemas import CATEFeatures, ExplainResponse, SHAPFeatureContribution
from ..dependencies import require_bundle
from .predict import _features_vector

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post(
    "/explain",
    response_model=ExplainResponse,
    summary="Explain a customer's CATE (surrogate SHAP)",
    description="Returns SHAP feature attributions for the customer's CATE. If the SHAP "
                "surrogate did not meet the fidelity gate (R^2 > 0.90), returns HTTP 200 "
                "with explanation_available=False and the specific reason.",
)
async def explain_cate(
    features: CATEFeatures,
    bundle=Depends(require_bundle),
):
    """SHAP explanation endpoint. 503 if no bundle; 200-degraded if surrogate invalid."""
    try:
        X = _features_vector(features, bundle)
        cate = float(np.asarray(bundle.causal_forest.effect(X)).reshape(-1)[0])
    except Exception as exc:
        logger.exception("Explain failed for customer %s", features.customer_id)
        raise HTTPException(status_code=500, detail=f"Explanation failed: {str(exc)}")

    # Degraded path (the live path this run): surrogate below fidelity gate.
    if not bundle.surrogate_valid:
        return ExplainResponse(
            customer_id=features.customer_id,
            cate_estimate=round(cate, 6),
            explanation_available=False,
            explanation_unavailable_reason=(
                f"SHAP surrogate fidelity R^2={bundle.surrogate_r2:.4f} is below the "
                f"required threshold of {bundle.surrogate_r2_threshold:.2f}. Feature "
                f"attributions withheld - reporting SHAP values from an unfaithful "
                f"surrogate would misrepresent the model's CATE function. This is a "
                f"consequence of the positivity constraints in the training data."
            ),
            surrogate_r2=bundle.surrogate_r2,
            surrogate_r2_threshold=bundle.surrogate_r2_threshold,
            top_features=[],
            baseline_cate=None,
            model_version=bundle.version,
        )

    # Valid path: interventional SHAP on the surrogate for this customer.
    try:
        import shap
        explainer = shap.TreeExplainer(
            bundle.shap_surrogate, feature_perturbation="interventional"
        )
        sv = np.asarray(explainer.shap_values(X)).reshape(-1)
        baseline = float(np.asarray(explainer.expected_value).reshape(-1)[0])
        contribs = sorted(
            (SHAPFeatureContribution(feature=f, shap_value=round(float(v), 6))
             for f, v in zip(bundle.feature_cols, sv)),
            key=lambda c: abs(c.shap_value), reverse=True,
        )
    except Exception as exc:
        logger.exception("SHAP computation failed for customer %s", features.customer_id)
        raise HTTPException(status_code=500, detail=f"Explanation failed: {str(exc)}")

    return ExplainResponse(
        customer_id=features.customer_id,
        cate_estimate=round(cate, 6),
        explanation_available=True,
        explanation_unavailable_reason=None,
        surrogate_r2=bundle.surrogate_r2,
        surrogate_r2_threshold=bundle.surrogate_r2_threshold,
        top_features=contribs,
        baseline_cate=round(baseline, 6),
        model_version=bundle.version,
    )