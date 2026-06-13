# src/clv_uplift/models/__init__.py
"""Models sub-package."""
from clv_uplift.models.diagnostics import run_diagnostics, DiagnosticsResult
from clv_uplift.models.uplift import train_uplift_models, UpliftModelBundle
from clv_uplift.models.validation import run_validation, ValidationResult
from clv_uplift.models.explain import run_surrogate_shap, SurrogateExplanation
from clv_uplift.models.policy import run_policy_and_fairness, PolicyFairnessResult
from clv_uplift.models.serving import ServingBundle

__all__ = [
    "run_diagnostics",
    "DiagnosticsResult",
    "train_uplift_models",
    "UpliftModelBundle",
    "run_validation",
    "ValidationResult",
    "run_surrogate_shap",
    "SurrogateExplanation",
    "run_policy_and_fairness",
    "PolicyFairnessResult",
    "ServingBundle",
]