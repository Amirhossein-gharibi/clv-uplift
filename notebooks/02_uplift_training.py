# notebooks/02_uplift_training.py
# %% [markdown]
# # Phase 2 training + persistence
# Runs the full causal pipeline once and pickles a ServingBundle (including the fitted
# RFMBinner) to MODEL_PATH for the API to load. Cell markers (# %%) run interactively.

# %%
import pickle
import logging

from clv_uplift.config import FEATURE_COLS, RFM_QUANTILES, MODEL_PATH
from clv_uplift.data.audit import load_raw
from clv_uplift.data.loader import clean_transactions
from clv_uplift.features.rfm import (
    build_rfm, assign_clv_segment, create_synthetic_treatment, split_data, RFMBinner,
)
from clv_uplift.models.diagnostics import run_diagnostics
from clv_uplift.models.uplift import train_uplift_models
from clv_uplift.models.validation import run_validation
from clv_uplift.models.explain import run_surrogate_shap
from clv_uplift.models.policy import run_policy_and_fairness
from clv_uplift.models.serving import ServingBundle

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("train")

# %%
# --- Full Phase 2 pipeline --------------------------------------------------------------
raw = load_raw()
clean = clean_transactions(raw)
rfm = build_rfm(clean)
rfm = assign_clv_segment(rfm)
rfm = create_synthetic_treatment(rfm)
rfm_train, rfm_test = split_data(rfm)

diag = run_diagnostics(rfm_train, rfm_test)
bundle = train_uplift_models(rfm_train, rfm_test, diag)
validation = run_validation(rfm_train, rfm_test, bundle)
explanation = run_surrogate_shap(rfm_train, rfm_test, bundle)
run_policy_and_fairness(rfm_train, rfm_test, bundle, clean)

# %%
# --- Fit the serving binner on TRAINING data (modal lookup; logs mismatch per feature) --
binner = RFMBinner(n_quantiles=RFM_QUANTILES).fit(
    X_raw=rfm_train[["recency_days", "frequency", "monetary_value"]],
    scores_train=rfm_train[["r_score", "f_score", "m_score"]],
)

# %%
# --- Assemble the ServingBundle from the correct source objects -------------------------
serving = ServingBundle(
    model_name="CausalForestDML",
    version="0.1.0",
    causal_forest=bundle.causal_forest,
    rfm_binner=binner,
    feature_cols=list(FEATURE_COLS),
    rate_test_passed=bool(validation.rate_test_passed),
    surrogate_valid=bool(explanation.surrogate_valid),
    surrogate_r2=float(explanation.surrogate_r2),
    surrogate_r2_threshold=0.90,
    shap_surrogate=explanation.surrogate,
    ate_point=float(bundle.ate_cf),
    ate_ci_lower=float(bundle.ate_cf_ci[0]),
    ate_ci_upper=float(bundle.ate_cf_ci[1]),
    e_value_point=float(validation.e_value_point),
    e_value_ci=float(validation.e_value_ci),
    metadata={
        "python": "3.11.9",
        "econml": "0.16.0",
        "dowhy": "0.14",
        "positivity_strain": "24.4% extreme propensity",
        "confounding_correction": "+0.229",
        "tier_ate": 2,
        "tier_heterogeneity": 3,
        "naive_association": float(bundle.naive_diff_test),
        "autoc_est": float(validation.autoc_est),
        "cate_cal_r2": float(validation.cate_cal_r2),
        "binner_mismatch": binner.mismatch_rates_,
    },
)

# %%
# --- Persist -----------------------------------------------------------------------------
with open(MODEL_PATH, "wb") as f:
    pickle.dump(serving, f, protocol=pickle.HIGHEST_PROTOCOL)
logger.info("ServingBundle saved to %s", MODEL_PATH)
print(f"ServingBundle saved to {MODEL_PATH}")
print(f"  rate_test_passed={serving.rate_test_passed}  surrogate_valid={serving.surrogate_valid}  "
      f"surrogate_r2={serving.surrogate_r2:.4f}")
print(f"  ATE={serving.ate_point:+.4f} CI [{serving.ate_ci_lower:+.4f}, {serving.ate_ci_upper:+.4f}]")
print(f"  binner mismatch: {{ {', '.join(f'{k}: {v*100:.1f}%' for k, v in binner.mismatch_rates_.items())} }}")