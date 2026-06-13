# src/clv_uplift/models/policy.py
"""
Phase 2.4 - GATE, policy gate, and (geographic) fairness audit.

Three components, shaped by the Phase 2.2 RATE Tier-3 null (rate_test_passed=False):

  PolicyTree   SKIPPED ENTIRELY - not built with a caveat. The pre-specified hard gate
               (rate_test_passed) forbids constructing a targeting policy on unconfirmed
               heterogeneity. No policy learner is imported or called.

  GATE         Group Average Treatment Effects, two groupings with DIFFERENT standing:
                 - by clv_segment: CERTIFIED findings. A pre-registered grouping (defined
                   from RFM rules before any model). Group effect + honest CI come from
                   forest.ate(X_group) / forest.ate_interval(X_group) - NOT from CATE
                   sorting - so the RATE null does not invalidate them.
                 - by CATE quintile: DESCRIPTIVE ONLY, explicitly labelled. Shows the
                   pattern the forest learned (what it WOULD suggest) without certifying it
                   for targeting.

  Fairness     GEOGRAPHIC EQUITY audit (the dataset has no legally protected attributes).
               Country is reintroduced purely as a post-hoc grouping variable - never in
               FEATURE_COLS, never in the model/treatment. Reframed as PROSPECTIVE
               documentation (no policy is deployed): what a future policy WOULD do across
               geographies/segments if positivity were restored.

Run standalone (full pipeline -> ... -> SHAP -> GATE/policy/fairness):
    python -m clv_uplift.models.policy
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from clv_uplift.config import FEATURE_COLS, CLV_SEGMENTS, FIGURES_DIR, BREAKEVEN_CATE
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
from clv_uplift.models.explain import run_surrogate_shap

warnings.filterwarnings("ignore", message="X does not have valid feature names", category=UserWarning)

FAIRNESS_TOP_N_COUNTRIES = 5
MIN_GROUP_N = 20   # below this a geographic group is too small to report reliably


@dataclass
class PolicyFairnessResult:
    policy_tree_built: bool
    gate_by_segment: dict        # segment -> {ate, ci_low, ci_high, n}
    gate_by_quintile: dict       # quintile -> {mean_pred_cate, n, emp_treated, emp_control, emp_diff}
    fairness_by_country: dict    # country -> {mean_cate, std, n, targeting_rate}
    fairness_by_segment: dict    # segment -> {targeting_rate, n}
    overall_targeting_rate: float
    spd_country: float           # statistical parity difference across countries
    breakeven: float
    metadata: dict = field(default_factory=dict)


def _rule(char: str = "=", width: int = 70) -> str:
    return char * width


def _scalar(x) -> float:
    return float(np.asarray(x).reshape(-1)[0])


def _country_by_customer(clean_df: pd.DataFrame) -> pd.Series:
    """Most frequent Country per CustomerID (grouping variable only - never a feature)."""
    return clean_df.groupby("CustomerID")["Country"].agg(
        lambda s: s.mode().iloc[0] if len(s.mode()) else s.iloc[0]
    )


def run_policy_and_fairness(rfm_train: pd.DataFrame, rfm_test: pd.DataFrame,
                            bundle: UpliftModelBundle,
                            clean_df: pd.DataFrame) -> PolicyFairnessResult:
    print("\n" + _rule())
    print("PHASE 2.4  GATE / POLICY / FAIRNESS")
    print(_rule())

    forest = bundle.causal_forest
    Xte = rfm_test[FEATURE_COLS].values
    cate = np.asarray(bundle.cate_test["causal_forest"]).reshape(-1)
    treatment = rfm_test["treatment"].values
    outcome = rfm_test["outcome"].values
    rate_passed = bool(bundle.validation["rate_test_passed"])

    # --- POLICY GATE ------------------------------------------------------------------
    print("\n" + _rule("-"))
    print("POLICY (PolicyTree)")
    print(_rule("-"))
    if rate_passed:
        # Not reached in this run; left explicit so the gate logic is visible.
        print("  rate_test_passed=True -> a targeting policy could be constructed here.")
        policy_tree_built = False  # construction deferred to a dedicated step if ever enabled
    else:
        print("  rate_test_passed=False -> PolicyTree is WITHHELD entirely (hard gate).")
        print("  No targeting policy is constructed. Per the pre-specified gate, shipping a")
        print("  policy built on unconfirmed heterogeneity is worse than shipping none.")
        policy_tree_built = False

    # --- GATE by clv_segment (CERTIFIED) ----------------------------------------------
    print("\n" + _rule("-"))
    print("GATE by clv_segment  (CERTIFIED findings; forest ate/ate_interval per group)")
    print(_rule("-"))
    gate_by_segment = {}
    seg_series = rfm_test["clv_segment"].astype(str).values
    for seg in CLV_SEGMENTS:
        mask = seg_series == seg
        n = int(mask.sum())
        if n < 2:
            print(f"  {seg:10s}: n={n} (too small; skipped)")
            continue
        X_seg = Xte[mask]
        ate_seg = _scalar(forest.ate(X_seg))
        lo, hi = forest.ate_interval(X_seg, alpha=0.05)
        lo, hi = _scalar(lo), _scalar(hi)
        gate_by_segment[seg] = {"ate": ate_seg, "ci_low": lo, "ci_high": hi, "n": n}
        sig = "" if lo <= 0 <= hi else "  (CI excludes 0)"
        print(f"  {seg:10s}: ATE {ate_seg:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  n={n}{sig}")

    # --- GATE by CATE quintile (DESCRIPTIVE ONLY) -------------------------------------
    print("\n" + _rule("-"))
    print("GATE by CATE quintile  (DESCRIPTIVE ONLY - not certified for targeting, RATE Tier-3)")
    print(_rule("-"))
    quint = pd.qcut(pd.Series(cate).rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int).values
    gate_by_quintile = {}
    for q in [1, 2, 3, 4, 5]:
        m = quint == q
        n = int(m.sum())
        mean_pred = float(cate[m].mean())
        t_mask = m & (treatment == 1)
        c_mask = m & (treatment == 0)
        emp_t = float(outcome[t_mask].mean()) if t_mask.sum() else float("nan")
        emp_c = float(outcome[c_mask].mean()) if c_mask.sum() else float("nan")
        emp_diff = emp_t - emp_c
        gate_by_quintile[int(q)] = {
            "mean_pred_cate": mean_pred, "n": n,
            "emp_treated": emp_t, "emp_control": emp_c, "emp_diff": emp_diff,
        }
        print(f"  Q{q}: predicted CATE {mean_pred:+.4f} | empirical (treated-control) "
              f"{emp_diff:+.4f} (nt={int(t_mask.sum())}, nc={int(c_mask.sum())}) | n={n}")
    print("  NOTE: 'empirical (treated-control)' is confounded WITHIN quintile and is")
    print("        descriptive only - it is NOT a causal group effect.")

    # --- FAIRNESS: geographic equity (PROSPECTIVE) ------------------------------------
    print("\n" + _rule("-"))
    print("FAIRNESS - geographic equity (PROSPECTIVE; no protected attributes in data)")
    print(_rule("-"))
    country = rfm_test.index.map(_country_by_customer(clean_df))
    fair_df = pd.DataFrame({
        "Country": np.asarray(country, dtype=object),
        "cate": cate,
        "clv_segment": seg_series,
    })
    overall_targeting_rate = float((fair_df["cate"] > BREAKEVEN_CATE).mean())
    print(f"  Break-even CATE threshold: {BREAKEVEN_CATE} (cost/AOV = 0.50/20)")
    print(f"  Overall prospective targeting rate (CATE > {BREAKEVEN_CATE}): "
          f"{overall_targeting_rate * 100:.1f}%")

    top_countries = fair_df["Country"].value_counts().head(FAIRNESS_TOP_N_COUNTRIES).index.tolist()
    print(f"\n  By Country (top {FAIRNESS_TOP_N_COUNTRIES} by customer count):")
    fairness_by_country = {}
    rates = []
    for c in top_countries:
        sub = fair_df[fair_df["Country"] == c]
        n = len(sub)
        rate = float((sub["cate"] > BREAKEVEN_CATE).mean())
        rates.append(rate)
        fairness_by_country[c] = {
            "mean_cate": float(sub["cate"].mean()),
            "std": float(sub["cate"].std()),
            "n": n,
            "targeting_rate": rate,
        }
        warn = "  (< 20; interpret cautiously)" if n < MIN_GROUP_N else ""
        print(f"    {c:18s}: mean CATE {sub['cate'].mean():+.4f} | targeting rate "
              f"{rate * 100:5.1f}% | n={n}{warn}")
    spd_country = float(max(rates) - min(rates)) if rates else float("nan")
    print(f"\n  Statistical Parity Difference across countries (max - min targeting rate): "
          f"{spd_country * 100:.1f} pp")

    print(f"\n  By clv_segment (prospective targeting rate):")
    fairness_by_segment = {}
    for seg in CLV_SEGMENTS:
        sub = fair_df[fair_df["clv_segment"] == seg]
        n = len(sub)
        if n == 0:
            continue
        rate = float((sub["cate"] > BREAKEVEN_CATE).mean())
        fairness_by_segment[seg] = {"targeting_rate": rate, "n": n}
        print(f"    {seg:10s}: targeting rate {rate * 100:5.1f}%  n={n}")

    print("\n  PROSPECTIVE FRAMING: no targeting policy was deployed (RATE Tier-3 null).")
    print("  The figures above document what a future policy WOULD do across geographies")
    print("  and segments if the dataset were augmented to restore positivity - so any")
    print("  disparity could be reviewed BEFORE deployment.")

    # --- Figures ----------------------------------------------------------------------
    # GATE by segment (with CI error bars).
    if gate_by_segment:
        segs = list(gate_by_segment.keys())
        ates = [gate_by_segment[s]["ate"] for s in segs]
        los = [gate_by_segment[s]["ate"] - gate_by_segment[s]["ci_low"] for s in segs]
        his = [gate_by_segment[s]["ci_high"] - gate_by_segment[s]["ate"] for s in segs]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(range(len(segs)), ates, yerr=[los, his], capsize=4, alpha=0.8)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xticks(range(len(segs)))
        ax.set_xticklabels(segs)
        ax.set_ylabel("Group ATE (forest)")
        ax.set_title("GATE by CLV segment (95% CI)")
        fig.tight_layout()
        p = FIGURES_DIR / "gate_by_segment.png"
        fig.savefig(p, dpi=120); plt.close(fig)
        print(f"\n  Saved -> {p}")

    # Fairness targeting rate by country.
    if fairness_by_country:
        cs = list(fairness_by_country.keys())
        rs = [fairness_by_country[c]["targeting_rate"] * 100 for c in cs]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(range(len(cs)), rs, alpha=0.8)
        ax.axhline(overall_targeting_rate * 100, color="red", linestyle="--", linewidth=1,
                   label=f"overall {overall_targeting_rate * 100:.1f}%")
        ax.set_xticks(range(len(cs)))
        ax.set_xticklabels(cs, rotation=30, ha="right")
        ax.set_ylabel("Prospective targeting rate (%)")
        ax.set_title("Geographic equity: prospective targeting rate by country")
        ax.legend(loc="best", fontsize=8)
        fig.tight_layout()
        p = FIGURES_DIR / "fairness_by_country.png"
        fig.savefig(p, dpi=120); plt.close(fig)
        print(f"  Saved -> {p}")

    result = PolicyFairnessResult(
        policy_tree_built=policy_tree_built,
        gate_by_segment=gate_by_segment,
        gate_by_quintile=gate_by_quintile,
        fairness_by_country=fairness_by_country,
        fairness_by_segment=fairness_by_segment,
        overall_targeting_rate=overall_targeting_rate,
        spd_country=spd_country,
        breakeven=BREAKEVEN_CATE,
        metadata={"rate_test_passed": rate_passed,
                  "top_countries": top_countries},
    )
    bundle.policy_fairness = result

    print("\n" + _rule())
    print("PHASE 2.4 COMPLETE - paste this output back.")
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
    run_policy_and_fairness(rfm_train, rfm_test, bundle, clean)
    print("\nFULL PHASE 2 PIPELINE COMPLETE.")