# src/clv_uplift/api/schemas.py
#
# Pydantic v2 models define the API contract: valid inputs and output shapes.

from pydantic import BaseModel, Field
from typing import Annotated, Literal, Optional


VALID_SEGMENTS = Literal["Champions", "Loyal", "At-Risk", "Lost"]


class CustomerFeatures(BaseModel):
    """Input schema for the /score scaffold endpoint. FROZEN - the five original tests
    depend on this exact shape."""

    customer_id: str = Field(
        ...,
        description="Unique customer identifier",
        json_schema_extra={"example": "12345"},
    )
    recency_days: Annotated[int, Field(
        ge=0, le=3650, description="Days since the customer's last purchase",
    )]
    frequency: Annotated[int, Field(
        ge=1, description="Number of distinct invoices (purchases)",
    )]
    monetary_value: Annotated[float, Field(
        ge=0.0, description="Total spend in GBP across all purchases",
    )]
    clv_segment: VALID_SEGMENTS


class CATEFeatures(BaseModel):
    """
    Input schema for the CATE endpoints (/predict, /explain). Accepts RAW RFM values;
    r/f/m quartile scoring is handled server-side by the bundle's fitted RFMBinner, so
    callers never need the training-distribution cutpoints. cancel_rate is passed through
    directly; clv_segment is reporting context only and is NOT a model feature.
    """

    customer_id: str = Field(..., json_schema_extra={"example": "C_002"})
    recency_days: Annotated[int, Field(ge=0, le=3650)]
    frequency: Annotated[int, Field(ge=1)]
    monetary_value: Annotated[float, Field(ge=0.0)]
    cancel_rate: Annotated[float, Field(ge=0.0, le=1.0)] = 0.0
    clv_segment: VALID_SEGMENTS = "Loyal"


class ScoreResponse(BaseModel):
    """Output schema for the (scaffold) /score endpoint. Unchanged."""

    customer_id: str
    recency_score: float = Field(description="Recency signal. Range [0, 1].")
    frequency_score: float = Field(description="Frequency signal, log-normalized. Range [0, 1].")
    composite_score: float = Field(description="Weighted recency+frequency. Range [0, 1].")
    model_version: str = Field(description="Version of the model that produced this score.")


class PredictResponse(BaseModel):
    """Output schema for /api/v1/predict (real CausalForestDML CATE)."""

    customer_id: str
    cate_estimate: float = Field(
        description="Individual CATE (incremental conversion probability) from CausalForestDML."
    )
    cate_ci_lower: float = Field(description="Lower bound, 95% CI (effect_interval).")
    cate_ci_upper: float = Field(description="Upper bound, 95% CI (effect_interval).")
    targeting_certified: bool = Field(
        description="True only if the RATE test confirmed heterogeneity at 95%. "
                    "False means the score is NOT endorsed for targeting."
    )
    confidence: Literal["high", "medium", "low", "not_certified"]
    recommended_action: str
    model_version: str

    model_config = {
        "json_schema_extra": {
            "example": {
                "customer_id": "C_002",
                "cate_estimate": 0.1264,
                "cate_ci_lower": -0.0612,
                "cate_ci_upper": 0.3145,
                "targeting_certified": False,
                "confidence": "not_certified",
                "recommended_action": "Individual CATE available but heterogeneity not "
                                      "statistically confirmed (RATE Tier-3 null). Do not "
                                      "use for targeting.",
                "model_version": "0.1.0",
            }
        }
    }


class SHAPFeatureContribution(BaseModel):
    """One feature's SHAP contribution to a customer's CATE (populated only when valid)."""
    feature: str
    shap_value: float


class ExplainResponse(BaseModel):
    """Output schema for /api/v1/explain. HTTP 200 even when explanation is withheld."""

    customer_id: str
    cate_estimate: float
    explanation_available: bool
    explanation_unavailable_reason: Optional[str] = None
    surrogate_r2: float
    surrogate_r2_threshold: float
    top_features: list[SHAPFeatureContribution] = Field(default_factory=list)
    baseline_cate: Optional[float] = None
    model_version: str


class HealthResponse(BaseModel):
    """Output schema for GET /health. Unchanged."""

    status: Literal["ok", "degraded"]
    model_loaded: bool
    version: str