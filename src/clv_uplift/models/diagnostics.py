# src/clv_uplift/models/diagnostics.py
"""
Phase 2.0 - Pre-estimation diagnostics (four required checks).

These run BEFORE any uplift model is fitted. All four must pass or be explicitly logged
and explained before proceeding to Phase 2.1.

  2.0.1  Propensity score calibration  - isotonic calibration; ECE before/after (BOTH
                                         out-of-fold for a fair comparison); reliability
                                         diagram.
  2.0.2  Overlap test                  - fraction of extreme propensity scores; overlap
                                         histograms saved to figures/overlap_propensity.png.
  2.0.3  SMD / balance check           - standardised mean differences (verbatim formula);
                                         flag |SMD| > 0.1 but DO NOT drop - the imbalance
                                         IS the confounding the model must disentangle.
  2.0.4  Outcome residual diagnostics  - cross_val_predict OOF residuals from a CALIBRATED
                                         outcome model (isotonic, num_leaves=15); four checks.

Calibrated propensity (model + ps_train/ps_test from the FULL fit) and overlap weights
are returned in a DiagnosticsResult for consumption by Phase 2.1 (X-Learner receives the
pre-calibrated propensity rather than re-estimating it internally). NOTE the separation:
the 2.0.1 ECE diagnostic uses OUT-OF-FOLD propensities for an honest before/after read,
while the ps_train/ps_test arrays fed downstream come from the full-data fit (as required
for serving).

Run standalone (rebuilds the feature pipeline, then runs all four checks):
    python -m clv_uplift.models.diagnostics
"""
from __future__ import annotations

import platform
from dataclasses import dataclass

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # headless, non-interactive - safe on Windows and in Docker
import matplotlib.pyplot as plt

from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import cross_val_predict
from lightgbm import LGBMClassifier

from clv_uplift.config import FEATURE_COLS, FIGURES_DIR, RANDOM_SEED
from clv_uplift.data.audit import load_raw
from clv_uplift.data.loader import clean_transactions
from clv_uplift.features.rfm import (
    build_rfm,
    assign_clv_segment,
    create_synthetic_treatment,
    split_data,
)

# Nuisance LGBM configs. Propensity matches the CausalForestDML model_t (200 trees).
# Outcome matches model_y (300 trees) but adds num_leaves=15 to curb tail overconfidence
# (teaching-chat Decision 2); it is additionally isotonic-calibrated in 2.0.4.
PROPENSITY_LGBM = dict(n_estimators=200, random_state=RANDOM_SEED, verbose=-1, n_jobs=1)
OUTCOME_LGBM    = dict(n_estimators=300, num_leaves=15, random_state=RANDOM_SEED,
                       verbose=-1, n_jobs=1)

# Thresholds (from the handoff).
EXTREME_PS_LOW       = 0.05
EXTREME_PS_HIGH      = 0.95
EXTREME_FRAC_LIMIT   = 0.05   # overlap test: switch to overlap weights if exceeded
SMD_FLAG             = 0.10   # |SMD| above this is flagged (not dropped)
RESID_QUINTILE_FLAG  = 0.05   # mean residual in a fitted-value quintile above this is flagged
N_CALIB_BINS         = 10


@dataclass
class DiagnosticsResult:
    """Artifacts Phase 2.1 needs, plus the diagnostic summary."""
    calibrated_propensity: object      # fitted CalibratedClassifierCV (full-data fit)
    ps_train: np.ndarray               # full-fit calibrated propensity (downstream use)
    ps_test: np.ndarray
    overlap_weights_train: np.ndarray
    extreme_fraction: float
    use_overlap_weights: bool
    ece_before: float                  # OOF raw
    ece_after: float                   # OOF calibrated (fair comparison)
    smd: pd.Series
    max_abs_smd: float
    residual_quintile_flags: list


def _rule(char: str = "=", width: int = 70) -> str:
    return char * width


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray,
                               n_bins: int = N_CALIB_BINS) -> float:
    """Equal-width-bin ECE: sample-weighted mean |accuracy - confidence| across bins."""
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.digitize(y_prob, edges[1:-1])  # bin indices 0..n_bins-1
    n = len(y_true)
    ece = 0.0
    for b in range(n_bins):
        mask = idx == b
        if not mask.any():
            continue
        conf = y_prob[mask].mean()
        acc = y_true[mask].mean()
        ece += (mask.sum() / n) * abs(acc - conf)
    return float(ece)


def _reliability_points(y_true: np.ndarray, y_prob: np.ndarray,
                        n_bins: int = N_CALIB_BINS):
    """Per-bin (mean predicted prob, observed fraction positive) for a reliability plot."""
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.digitize(y_prob, edges[1:-1])
    xs, ys = [], []
    for b in range(n_bins):
        mask = idx == b
        if not mask.any():
            continue
        xs.append(y_prob[mask].mean())
        ys.append(y_true[mask].mean())
    return np.array(xs), np.array(ys)


def _check_propensity_calibration(Xtr, Xte, t_train):
    """
    2.0.1 - isotonic calibration; ECE before/after; reliability diagram.

    Both ECE numbers are OUT-OF-FOLD so the comparison is fair:
      before = raw LGBM propensity via cross_val_predict
      after  = isotonic-calibrated propensity via cross_val_predict over the
               CalibratedClassifierCV (nested CV)
    The ps_train / ps_test arrays returned for DOWNSTREAM use come from the full-data
    fit (required for serving), which is a separate object from the OOF diagnostic.
    """
    print(_rule())
    print("2.0.1  PROPENSITY SCORE CALIBRATION")
    print(_rule())

    # "Before": honest out-of-fold raw LGBM propensity.
    raw = LGBMClassifier(**PROPENSITY_LGBM)
    ps_raw_oof = cross_val_predict(raw, Xtr, t_train, cv=5, method="predict_proba")[:, 1]
    ece_before = expected_calibration_error(t_train.values, ps_raw_oof)

    # "After" (diagnostic): out-of-fold isotonic-calibrated propensity (nested CV) so it
    # is directly comparable to 'before'. Estimator passed POSITIONALLY for cross-version
    # safety (named 'base_estimator' pre-sklearn-1.2, 'estimator' from 1.2+).
    cal_template = CalibratedClassifierCV(
        LGBMClassifier(**PROPENSITY_LGBM), method="isotonic", cv=5
    )
    ps_cal_oof = cross_val_predict(
        cal_template, Xtr, t_train, cv=5, method="predict_proba"
    )[:, 1]
    ece_after = expected_calibration_error(t_train.values, ps_cal_oof)

    # Full-data fit -> the calibrated propensities actually fed downstream.
    calibrated = CalibratedClassifierCV(
        LGBMClassifier(**PROPENSITY_LGBM), method="isotonic", cv=5
    )
    calibrated.fit(Xtr, t_train)
    ps_train = calibrated.predict_proba(Xtr)[:, 1]
    ps_test = calibrated.predict_proba(Xte)[:, 1]

    print(f"ECE before (raw, OOF)            : {ece_before:.4f}")
    print(f"ECE after  (isotonic, OOF)       : {ece_after:.4f}")
    improved = "improved" if ece_after < ece_before else "did NOT improve"
    print(f"Calibration {improved} (both OOF, lower ECE is better).")
    print("Note: both ECEs are out-of-fold for a fair comparison. The ps_train/ps_test "
          "fed downstream come from the full-data fit (separate from this diagnostic).")

    # Reliability diagram (before vs after, both OOF).
    bx, by = _reliability_points(t_train.values, ps_raw_oof)
    ax_, ay = _reliability_points(t_train.values, ps_cal_oof)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="perfect")
    ax.plot(bx, by, "o-", label=f"before OOF (ECE={ece_before:.3f})")
    ax.plot(ax_, ay, "s-", label=f"after OOF (ECE={ece_after:.3f})")
    ax.set_xlabel("Mean predicted propensity")
    ax.set_ylabel("Observed treated fraction")
    ax.set_title("Propensity reliability diagram (OOF)")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    calib_path = FIGURES_DIR / "propensity_calibration.png"
    fig.savefig(calib_path, dpi=120)
    plt.close(fig)
    print(f"Saved reliability diagram -> {calib_path}")

    return calibrated, ps_train, ps_test, ece_before, ece_after


def _check_overlap(ps_train, t_train):
    """2.0.2 - extreme propensity fraction; overlap histograms saved."""
    print("\n" + _rule())
    print("2.0.2  OVERLAP TEST")
    print(_rule())
    extreme_mask = (ps_train < EXTREME_PS_LOW) | (ps_train > EXTREME_PS_HIGH)
    extreme_fraction = float(extreme_mask.mean())
    use_overlap = extreme_fraction > EXTREME_FRAC_LIMIT

    print(f"Propensity range: [{ps_train.min():.4f}, {ps_train.max():.4f}]")
    print(f"Extreme fraction (ps<{EXTREME_PS_LOW} or ps>{EXTREME_PS_HIGH}): "
          f"{extreme_fraction * 100:.2f}%")
    if use_overlap:
        print(f"  FLAG: exceeds {EXTREME_FRAC_LIMIT * 100:.0f}% -> overlap weights "
              f"RECOMMENDED for Phase 2.1 (treated: 1-ps, control: ps).")
    else:
        print(f"  OK: at or below {EXTREME_FRAC_LIMIT * 100:.0f}% - overlap adequate.")

    overlap_weights = np.where(t_train.values == 1, 1.0 - ps_train, ps_train)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(ps_train[t_train.values == 0], bins=30, alpha=0.5, label="control", density=True)
    ax.hist(ps_train[t_train.values == 1], bins=30, alpha=0.5, label="treated", density=True)
    ax.axvline(EXTREME_PS_LOW, color="red", linestyle=":", linewidth=1)
    ax.axvline(EXTREME_PS_HIGH, color="red", linestyle=":", linewidth=1)
    ax.set_xlabel("Calibrated propensity score")
    ax.set_ylabel("Density")
    ax.set_title("Propensity overlap by treatment group")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    overlap_path = FIGURES_DIR / "overlap_propensity.png"
    fig.savefig(overlap_path, dpi=120)
    plt.close(fig)
    print(f"Saved overlap histogram -> {overlap_path}")

    return extreme_fraction, overlap_weights, use_overlap


def _check_balance(Xtr, t_train):
    """2.0.3 - standardised mean differences (verbatim formula); flag |SMD|>0.1."""
    print("\n" + _rule())
    print("2.0.3  SMD / BALANCE CHECK")
    print(_rule())

    def compute_smd(X, t):
        out = {}
        for col in X.columns:
            denom = X[col].std()
            num = X.loc[t == 1, col].mean() - X.loc[t == 0, col].mean()
            out[col] = (num / denom) if denom != 0 else 0.0
        return out

    smd = pd.Series(compute_smd(Xtr, t_train)).sort_values(key=np.abs, ascending=False)
    print("Standardised mean differences (treated - control) / pooled std:")
    flagged = []
    for col, val in smd.items():
        mark = "  FLAG" if abs(val) > SMD_FLAG else ""
        if abs(val) > SMD_FLAG:
            flagged.append(col)
        print(f"  {col:16s}: {val:+.4f}{mark}")
    print(f"\nFlagged |SMD| > {SMD_FLAG}: {flagged if flagged else 'none'}")
    print("Interpretation: flagged imbalance is NOT removed. Treatment correlates with "
          "recency_days and monetary_value by design, so residual imbalance on those "
          "(and their correlates) IS the confounding the uplift model disentangles.")
    return smd, float(smd.abs().max())


def _check_outcome_residuals(Xtr, y_train):
    """2.0.4 - OOF residual diagnostics from a CALIBRATED outcome model."""
    print("\n" + _rule())
    print("2.0.4  OUTCOME MODEL RESIDUAL DIAGNOSTICS")
    print(_rule())
    print("Outcome model: LGBM(n_estimators=300, num_leaves=15) + isotonic calibration "
          "(cv=5); residuals are nested-CV out-of-fold.")
    # Calibrated outcome model, estimator passed POSITIONALLY (cross-version safe).
    calibrated_outcome = CalibratedClassifierCV(
        LGBMClassifier(**OUTCOME_LGBM), method="isotonic", cv=5
    )
    y_hat = cross_val_predict(
        calibrated_outcome, Xtr, y_train, cv=5, method="predict_proba"
    )[:, 1]
    resid = y_train.values - y_hat

    # (a) residuals vs fitted values
    corr_fit = float(np.corrcoef(resid, y_hat)[0, 1])
    print(f"(a) mean residual: {resid.mean():+.4f} | corr(residual, fitted): {corr_fit:+.4f}")
    print("    (well-calibrated outcome model -> mean ~ 0 and low corr with fitted)")

    # (b) residuals vs each feature
    print("(b) corr(residual, feature):")
    for col in Xtr.columns:
        c = float(np.corrcoef(resid, Xtr[col].values)[0, 1])
        mark = "  FLAG" if abs(c) > 0.10 else ""
        print(f"    {col:16s}: {c:+.4f}{mark}")

    # (c) subgroup quintile means (quintiles of fitted value)
    print(f"(c) mean residual by fitted-value quintile (flag |mean| > {RESID_QUINTILE_FLAG}):")
    q = pd.qcut(pd.Series(y_hat).rank(method="first"), 5, labels=[1, 2, 3, 4, 5])
    flags = []
    for qi in [1, 2, 3, 4, 5]:
        m = (q == qi).values
        mr = float(resid[m].mean())
        mark = "  FLAG" if abs(mr) > RESID_QUINTILE_FLAG else ""
        if abs(mr) > RESID_QUINTILE_FLAG:
            flags.append(int(qi))
        print(f"    Q{qi}: mean residual {mr:+.4f}  (fitted~{y_hat[m].mean():.3f}){mark}")

    # (d) heteroskedasticity
    print("(d) residual std by fitted-value quintile:")
    for qi in [1, 2, 3, 4, 5]:
        m = (q == qi).values
        print(f"    Q{qi}: std {resid[m].std():.4f}")
    print("    Note: for a BINARY outcome, residual variance is mechanically ~p(1-p), so "
          "spread varying with fitted value is expected, not a model defect.")

    print(f"\nQuintiles flagged (|mean residual| > {RESID_QUINTILE_FLAG}): "
          f"{flags if flags else 'none'}")
    return flags


def run_diagnostics(rfm_train: pd.DataFrame, rfm_test: pd.DataFrame) -> DiagnosticsResult:
    """Run all four pre-estimation diagnostics and return artifacts for Phase 2.1."""
    print("\n" + _rule())
    print("PHASE 2.0  PRE-ESTIMATION DIAGNOSTICS")
    print(_rule())
    import sklearn
    import lightgbm
    import econml
    import dowhy
    print(f"env: python {platform.python_version()} | sklearn {sklearn.__version__} | "
          f"numpy {np.__version__} | lightgbm {lightgbm.__version__} | "
          f"econml {econml.__version__} | dowhy {dowhy.__version__}")

    Xtr = rfm_train[FEATURE_COLS]
    Xte = rfm_test[FEATURE_COLS]
    t_train = rfm_train["treatment"]
    y_train = rfm_train["outcome"]
    print(f"train: {len(Xtr):,} rows | test: {len(Xte):,} rows | "
          f"treated(train): {int(t_train.sum()):,} | positive(train): {int(y_train.sum()):,}")

    calibrated, ps_train, ps_test, ece_before, ece_after = _check_propensity_calibration(
        Xtr, Xte, t_train
    )
    extreme_fraction, overlap_weights, use_overlap = _check_overlap(ps_train, t_train)
    smd, max_abs_smd = _check_balance(Xtr, t_train)
    resid_flags = _check_outcome_residuals(Xtr, y_train)

    print("\n" + _rule())
    print("PHASE 2.0 COMPLETE - summary")
    print(_rule())
    print(f"  ECE before/after (OOF)  : {ece_before:.4f} -> {ece_after:.4f}")
    print(f"  Extreme propensity frac : {extreme_fraction * 100:.2f}%  "
          f"(overlap weights {'ON' if use_overlap else 'off'})")
    print(f"  Max |SMD|               : {max_abs_smd:.4f}")
    print(f"  Residual quintile flags : {resid_flags if resid_flags else 'none'}")
    print("Paste this output back.")
    print(_rule())

    return DiagnosticsResult(
        calibrated_propensity=calibrated,
        ps_train=ps_train,
        ps_test=ps_test,
        overlap_weights_train=overlap_weights,
        extreme_fraction=extreme_fraction,
        use_overlap_weights=use_overlap,
        ece_before=ece_before,
        ece_after=ece_after,
        smd=smd,
        max_abs_smd=max_abs_smd,
        residual_quintile_flags=resid_flags,
    )


if __name__ == "__main__":
    raw = load_raw()
    clean = clean_transactions(raw)
    rfm = build_rfm(clean)
    rfm = assign_clv_segment(rfm)
    rfm = create_synthetic_treatment(rfm)
    rfm_train, rfm_test = split_data(rfm)
    run_diagnostics(rfm_train, rfm_test)