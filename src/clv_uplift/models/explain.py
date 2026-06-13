# src/clv_uplift/models/explain.py
"""
Phase 2.3 - Surrogate SHAP explanation of the forest's CATE function.

A GradientBoostingRegressor surrogate is fitted to the CausalForestDML CATE estimates,
its fidelity to the forest is gated (R^2 > 0.90), and SHAP values are computed on the
surrogate with interventional perturbation. Verified against shap 0.48 / econml 0.16.

FRAMING (because the Phase 2.2 RATE test was a Tier-3 null): these SHAP values explain
what drives the FOREST'S CATE ESTIMATES - model transparency - NOT certified, actionable
heterogeneity. The forest learned a CATE surface with structure; SHAP describes that
surface faithfully (to within the surrogate R^2). RATE separately told us that structure
does not rise above noise at 95% for targeting. Both are true and non-contradictory.

R^2 GATE (graceful, not a crash): if surrogate R^2 < 0.90 the surrogate is approximating
the forest poorly, so SHAP would explain surrogate ERROR rather than the CATE function.
In that case SHAP is NOT computed or reported; this is documented as a fourth consequence
of the Phase 2.0 positivity strain (alongside the wide ATE CI, the CI-bound E-value of
1.0, and the RATE null). The surrogate is NEVER tuned to force the gate to pass.

Run standalone (full pipeline -> ... -> validation -> surrogate SHAP):
    python -m clv_uplift.models.explain
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import shap
from sklearn.ensemble import GradientBoostingRegressor

from clv_uplift.config import FEATURE_COLS, FIGURES_DIR, RANDOM_SEED
from clv_uplift.data.audit import load_raw
from clv_uplift.data.loader import clean_transactions
from clv_uplift.features.rfm import (
    build_rfm,
    assign_clv_segment,
    create_synthetic_treatment,
    split_data,
)
from clv_uplift.models.diagnostics import run_diagnostics
from clv_uplift.models.uplift import train_uplift_models, UpliftModelBundle
from clv_uplift.models.validation import run_validation

warnings.filterwarnings("ignore", message="X does not have valid feature names", category=UserWarning)

SURROGATE_R2_GATE = 0.90
SURROGATE_PARAMS  = dict(n_estimators=300, random_state=RANDOM_SEED)
SHAP_BACKGROUND_N = 200   # interventional background sample size (cost scales with this)


@dataclass
class SurrogateExplanation:
    """Surrogate model + SHAP results. Stored live on the bundle for the /explain endpoint."""
    surrogate: object
    surrogate_r2: float
    surrogate_valid: bool
    expected_value: float
    background: object                       # ndarray sample for rebuilding the explainer
    feature_importance: dict                 # feature -> mean|shap| (test); {} if not valid
    feature_direction: dict                  # feature -> sign of corr(feature, shap); {} if invalid
    shap_values_test: object = None          # (n_test, n_features) ndarray or None
    metadata: dict = field(default_factory=dict)


def _rule(char: str = "=", width: int = 70) -> str:
    return char * width


def run_surrogate_shap(rfm_train: pd.DataFrame, rfm_test: pd.DataFrame,
                       bundle: UpliftModelBundle) -> SurrogateExplanation:
    """Fit the surrogate, apply the R^2 gate, and (if it passes) compute interventional SHAP."""
    print("\n" + _rule())
    print("PHASE 2.3  SURROGATE SHAP")
    print(_rule())

    Xtr = rfm_train[FEATURE_COLS].values
    Xte = rfm_test[FEATURE_COLS].values

    # --- Step 1: fit GBR surrogate to the forest CATE on the TRAIN set ----------------
    cate_train = np.asarray(bundle.causal_forest.effect(Xtr)).reshape(-1)
    surrogate = GradientBoostingRegressor(**SURROGATE_PARAMS)
    surrogate.fit(Xtr, cate_train)

    # --- Step 2: R^2 fidelity gate (graceful) -----------------------------------------
    surrogate_r2 = float(surrogate.score(Xtr, cate_train))
    print(f"Surrogate fidelity R^2 (train): {surrogate_r2:.4f}  (gate > {SURROGATE_R2_GATE})")
    surrogate_valid = surrogate_r2 > SURROGATE_R2_GATE

    if not surrogate_valid:
        print("\n  GATE NOT MET: surrogate R^2 <= 0.90. SHAP would explain the surrogate's "
              "approximation error, not the forest's CATE function. SHAP is NOT computed.")
        print("  This is documented as a FOURTH consequence of the Phase 2.0 positivity strain "
              "(with the wide ATE CI, the CI-bound E-value of 1.0, and the RATE null).")
        print("  The surrogate is NOT adjusted to force the gate.")
        result = SurrogateExplanation(
            surrogate=surrogate, surrogate_r2=surrogate_r2, surrogate_valid=False,
            expected_value=float("nan"), background=None,
            feature_importance={}, feature_direction={}, shap_values_test=None,
            metadata={"random_seed": RANDOM_SEED, "reason": "surrogate_r2_below_gate"},
        )
        bundle.surrogate = surrogate
        bundle.surrogate_r2 = surrogate_r2
        bundle.surrogate_valid = False
        bundle.explanation = result
        print("\n" + _rule())
        print("PHASE 2.3 COMPLETE (SHAP withheld - gate not met) - paste this output back.")
        print(_rule())
        return result

    print("  GATE MET: the surrogate faithfully approximates the forest CATE surface.")

    # --- Step 3: interventional TreeExplainer -----------------------------------------
    rng = np.random.default_rng(RANDOM_SEED)
    if len(Xtr) > SHAP_BACKGROUND_N:
        bg_idx = rng.choice(len(Xtr), size=SHAP_BACKGROUND_N, replace=False)
        background = Xtr[bg_idx]
    else:
        background = Xtr
    print(f"  SHAP background: {len(background)} sampled training rows (interventional).")

    explainer = shap.TreeExplainer(
        surrogate, data=background, feature_perturbation="interventional"
    )
    shap_values_test = np.asarray(explainer.shap_values(Xte))
    expected_value = float(np.asarray(explainer.expected_value).reshape(-1)[0])

    # Additivity sanity check: expected_value + sum(shap) ~ surrogate prediction.
    recon = expected_value + shap_values_test.sum(axis=1)
    pred = surrogate.predict(Xte)
    max_add_err = float(np.abs(recon - pred).max())
    print(f"  SHAP additivity check: max |expected + sum(shap) - pred| = {max_add_err:.2e}")

    # --- Step 4: global importance + direction ----------------------------------------
    mean_abs = np.abs(shap_values_test).mean(axis=0)
    importance = {f: float(v) for f, v in zip(FEATURE_COLS, mean_abs)}
    order = sorted(importance, key=importance.get, reverse=True)
    direction = {}
    for j, f in enumerate(FEATURE_COLS):
        col = Xte[:, j]
        sv = shap_values_test[:, j]
        if col.std() > 0 and sv.std() > 0:
            direction[f] = float(np.sign(np.corrcoef(col, sv)[0, 1]))
        else:
            direction[f] = 0.0

    print("\n  Global SHAP importance (mean |SHAP| on test, descending):")
    for f in order:
        d = direction[f]
        arrow = "higher feature -> higher CATE" if d > 0 else \
                ("higher feature -> lower CATE" if d < 0 else "flat")
        print(f"    {f:16s}: {importance[f]:.5f}   ({arrow})")

    # --- Step 5: figures --------------------------------------------------------------
    # Robust bar chart (always works).
    fig, ax = plt.subplots(figsize=(6, 4))
    vals = [importance[f] for f in order]
    ax.barh(range(len(order)), vals)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order)
    ax.invert_yaxis()
    ax.set_xlabel("mean |SHAP value|  (impact on predicted CATE)")
    ax.set_title("Surrogate SHAP - global CATE feature importance")
    fig.tight_layout()
    bar_path = FIGURES_DIR / "shap_importance.png"
    fig.savefig(bar_path, dpi=120)
    plt.close(fig)
    print(f"\n  Saved importance bar -> {bar_path}")

    # Beeswarm summary (best-effort; shap plotting can be finicky under Agg).
    try:
        plt.figure()
        shap.summary_plot(shap_values_test, Xte, feature_names=FEATURE_COLS, show=False)
        bee_path = FIGURES_DIR / "shap_summary.png"
        plt.tight_layout()
        plt.savefig(bee_path, dpi=120, bbox_inches="tight")
        plt.close()
        print(f"  Saved SHAP summary  -> {bee_path}")
    except Exception as exc:
        print(f"  (SHAP summary plot skipped: {type(exc).__name__}: {exc})")

    result = SurrogateExplanation(
        surrogate=surrogate, surrogate_r2=surrogate_r2, surrogate_valid=True,
        expected_value=expected_value, background=background,
        feature_importance=importance, feature_direction=direction,
        shap_values_test=shap_values_test,
        metadata={
            "random_seed": RANDOM_SEED,
            "background_n": int(len(background)),
            "importance_order": order,
            "additivity_max_err": max_add_err,
        },
    )
    bundle.surrogate = surrogate
    bundle.surrogate_r2 = surrogate_r2
    bundle.surrogate_valid = True
    bundle.explanation = result

    print("\n" + _rule())
    print("PHASE 2.3 COMPLETE - paste this output back.")
    print(_rule())
    return result


if __name__ == "__main__":
    raw = load_raw()
    clean = clean_transactions(raw)
    rfm = build_rfm(clean)
    rfm = assign_clv_segment(rfm)
    rfm = create_synthetic_treatment(rfm)
    rfm_train, rfm_test = split_data(rfm)
    diag = run_diagnostics(rfm_train, rfm_test)
    bundle = train_uplift_models(rfm_train, rfm_test, diag)
    run_validation(rfm_train, rfm_test, bundle)
    run_surrogate_shap(rfm_train, rfm_test, bundle)
    print("\nFEATURE PIPELINE + DIAGNOSTICS + UPLIFT + VALIDATION + SHAP COMPLETE.")