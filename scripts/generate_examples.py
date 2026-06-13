# scripts/generate_examples.py
"""
Author-side artifact builder. Run ONCE after training to generate:
  - examples/sample_predict_response.json
  - examples/sample_explain_response.json
  - artifacts/figures/cate_distribution.png
  - artifacts/figures/qini_curve.png

Requires: artifacts/uplift_model.pkl AND data/raw/online_retail_II.xlsx
          (the data file is needed to reconstruct the test-set feature matrix for the
          full CATE array behind the two figures).

Outputs are committed to the repo. GitHub visitors see the committed results without
running this script or having the data file.

Usage (from project root, venv active):  python scripts/generate_examples.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from clv_uplift.config import FEATURE_COLS, FIGURES_DIR, BREAKEVEN_CATE
from clv_uplift.data.audit import load_raw
from clv_uplift.data.loader import clean_transactions
from clv_uplift.features.rfm import (
    build_rfm, assign_clv_segment, create_synthetic_treatment, split_data,
)
from clv_uplift.models.serving import load_bundle

ROOT = Path(__file__).resolve().parent.parent
EXAMPLES_DIR = ROOT / "examples"
EXAMPLES_DIR.mkdir(parents=True, exist_ok=True)


def _flat(a) -> np.ndarray:
    return np.asarray(a).reshape(-1)


def reconstruct_test_set():
    """Rebuild the exact seed-42 test split to recover the test-set CATE array."""
    raw = load_raw()
    clean = clean_transactions(raw)
    rfm = build_rfm(clean)
    rfm = assign_clv_segment(rfm)
    rfm = create_synthetic_treatment(rfm)
    _, rfm_test = split_data(rfm)
    return rfm_test


def make_cate_distribution(cate: np.ndarray, ate_point: float):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(cate, bins=40, alpha=0.8, color="#1565c0")
    ax.axvline(0.0, color="#555", linestyle="-", linewidth=1, label="no effect")
    ax.axvline(BREAKEVEN_CATE, color="#888", linestyle="--", linewidth=1,
               label=f"break-even {BREAKEVEN_CATE}")
    ax.axvline(ate_point, color="#c62828", linestyle="-", linewidth=2,
               label=f"population ATE {ate_point:+.3f}")
    above = float((cate > BREAKEVEN_CATE).mean()) * 100
    ax.set_xlabel("CATE (incremental conversion probability)")
    ax.set_ylabel("Count (test-set customers)")
    ax.set_title(f"Test-set CATE distribution  ({above:.1f}% above break-even)")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    path = FIGURES_DIR / "cate_distribution.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"  saved -> {path}")


def make_qini_curve(cate: np.ndarray, treatment: np.ndarray, outcome: np.ndarray):
    """
    Qini-style cumulative-uplift curve: rank customers by predicted CATE (desc), then at
    each prefix compute the empirical uplift (treated mean - control mean) scaled by the
    number targeted, vs a random-targeting diagonal. Descriptive (RATE Tier-3), included
    as a standard uplift diagnostic.
    """
    order = np.argsort(-cate)
    t, y = treatment[order], outcome[order]
    n = len(cate)

    qini = np.zeros(n)
    for i in range(1, n + 1):
        tt, yy = t[:i], y[:i]
        n_t = tt.sum()
        n_c = i - n_t
        if n_t > 0 and n_c > 0:
            uplift = yy[tt == 1].mean() - yy[tt == 0].mean()
        else:
            uplift = 0.0
        qini[i - 1] = uplift * i

    x = np.arange(1, n + 1) / n
    random_line = np.linspace(0, qini[-1], n)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(x, qini, color="#1565c0", linewidth=2, label="model (CATE-ranked)")
    ax.plot(x, random_line, color="#888", linestyle="--", linewidth=1, label="random targeting")
    ax.set_xlabel("Fraction of population targeted (ranked by predicted CATE)")
    ax.set_ylabel("Cumulative uplift (scaled)")
    ax.set_title("Qini curve - model vs random targeting (descriptive)")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    path = FIGURES_DIR / "qini_curve.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"  saved -> {path}")


def predict_one(bundle, cust: dict) -> dict:
    raw = pd.DataFrame([{
        "recency_days": cust["recency_days"],
        "frequency": cust["frequency"],
        "monetary_value": cust["monetary_value"],
        "cancel_rate": cust["cancel_rate"],
    }])
    X = bundle.rfm_binner.transform(raw).values
    cate = float(_flat(bundle.causal_forest.effect(X))[0])
    lb, ub = bundle.causal_forest.effect_interval(X, alpha=0.05)
    lo, hi = float(_flat(lb)[0]), float(_flat(ub)[0])
    return {
        "customer_id": cust["customer_id"],
        "cate_estimate": round(cate, 6),
        "cate_ci_lower": round(lo, 6),
        "cate_ci_upper": round(hi, 6),
        "targeting_certified": bool(bundle.rate_test_passed),
        "confidence": "high" if bundle.rate_test_passed else "not_certified",
        "model_version": bundle.version,
    }


def main():
    print("Loading bundle ...")
    bundle = load_bundle()

    print("Reconstructing test set (seed 42) for the figure CATE array ...")
    rfm_test = reconstruct_test_set()
    Xte = rfm_test[FEATURE_COLS].values
    cate = _flat(bundle.causal_forest.effect(Xte))
    treatment = rfm_test["treatment"].values
    outcome = rfm_test["outcome"].values

    print("Generating figures ...")
    make_cate_distribution(cate, float(bundle.ate_point))
    make_qini_curve(cate, treatment, outcome)

    print("Generating sample API responses for the 5 example customers ...")
    examples = json.load(open(EXAMPLES_DIR / "sample_customers.json", encoding="utf-8"))
    predict_responses = [predict_one(bundle, c) for c in examples]

    explain_response = {
        "customer_id": examples[1]["customer_id"],
        "cate_estimate": predict_responses[1]["cate_estimate"],
        "explanation_available": bool(bundle.surrogate_valid),
        "explanation_unavailable_reason": (
            None if bundle.surrogate_valid else
            f"SHAP surrogate fidelity R^2={bundle.surrogate_r2:.4f} is below the required "
            f"threshold of {bundle.surrogate_r2_threshold:.2f}. Feature attributions withheld."
        ),
        "surrogate_r2": round(float(bundle.surrogate_r2), 6),
        "surrogate_r2_threshold": float(bundle.surrogate_r2_threshold),
        "top_features": [],
        "baseline_cate": None,
        "model_version": bundle.version,
    }

    with open(EXAMPLES_DIR / "sample_predict_response.json", "w", encoding="utf-8") as f:
        json.dump(predict_responses, f, indent=2)
    print(f"  saved -> {EXAMPLES_DIR / 'sample_predict_response.json'}")
    with open(EXAMPLES_DIR / "sample_explain_response.json", "w", encoding="utf-8") as f:
        json.dump(explain_response, f, indent=2)
    print(f"  saved -> {EXAMPLES_DIR / 'sample_explain_response.json'}")

    print("\nDone. Commit examples/*.json and artifacts/figures/*.png.")


if __name__ == "__main__":
    main()