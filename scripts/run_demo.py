# scripts/run_demo.py
"""
Zero-friction CLI demo. Loads the trained ServingBundle and predicts CATE + 95% CI for the
five example customers - NO data file, NO Docker, NO running API required. Prints the
population context and the tiered findings so a reader sees the whole story from one command.

Usage (from project root, venv active, after training):  python scripts/run_demo.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from clv_uplift.config import BREAKEVEN_CATE
from clv_uplift.models.serving import load_bundle

ROOT = Path(__file__).resolve().parent.parent
EXAMPLES = ROOT / "examples" / "sample_customers.json"


def _flat(a) -> np.ndarray:
    return np.asarray(a).reshape(-1)


def main():
    bundle = load_bundle()

    print("=" * 78)
    print("CLV UPLIFT - DEMO")
    print("=" * 78)
    print(f"Model            : {bundle.model_name} v{bundle.version}")
    print(f"Population ATE    : {bundle.ate_point:+.4f}  "
          f"95% CI [{bundle.ate_ci_lower:+.4f}, {bundle.ate_ci_upper:+.4f}]")
    md = bundle.metadata or {}
    print(f"Naive association: {md.get('naive_association', float('nan')):+.4f}  "
          f"(confounding correction {md.get('confounding_correction', 'n/a')})")
    print(f"E-value (point)  : {bundle.e_value_point:.2f}   "
          f"E-value (CI bound): {bundle.e_value_ci:.2f}")
    print(f"RATE certified   : {bundle.rate_test_passed}  "
          f"(Tier {md.get('tier_ate', 2)} ATE / Tier {md.get('tier_heterogeneity', 3)} heterogeneity)")
    print(f"Surrogate R^2    : {bundle.surrogate_r2:.4f} "
          f"(gate {bundle.surrogate_r2_threshold:.2f}; valid={bundle.surrogate_valid})")
    print("=" * 78)

    examples = json.load(open(EXAMPLES, encoding="utf-8"))
    print(f"\n{'Customer':<8} {'Segment':<11} {'CATE':>9} {'CI_low':>9} {'CI_high':>9}  Certified")
    print("-" * 78)
    for c in examples:
        raw = pd.DataFrame([{
            "recency_days": c["recency_days"],
            "frequency": c["frequency"],
            "monetary_value": c["monetary_value"],
            "cancel_rate": c["cancel_rate"],
        }])
        X = bundle.rfm_binner.transform(raw).values
        cate = float(_flat(bundle.causal_forest.effect(X))[0])
        lb, ub = bundle.causal_forest.effect_interval(X, alpha=0.05)
        lo, hi = float(_flat(lb)[0]), float(_flat(ub)[0])
        certified = "yes" if bundle.rate_test_passed else "no"
        print(f"{c['customer_id']:<8} {c['clv_segment']:<11} "
              f"{cate:>+9.4f} {lo:>+9.4f} {hi:>+9.4f}  {certified}")

    print("-" * 78)
    print(f"Break-even CATE for targeting: {BREAKEVEN_CATE}")
    if not bundle.rate_test_passed:
        print("\nNOTE: RATE test did not certify heterogeneity (Tier-3 null). The population")
        print("ATE is real and robust, but individual CATEs are NOT certified for targeting.")
    print("=" * 78)


if __name__ == "__main__":
    main()