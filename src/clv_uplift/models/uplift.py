# src/clv_uplift/models/uplift.py
"""
Phase 2.1 - Uplift estimators.

Fits four CATE estimators on the train split and evaluates CATE on the test split:

  CausalForestDML  PRIMARY. Unweighted (DML residualization handles overlap
                   intrinsically). Targets the ATE over the full population and is the
                   ONLY estimator here with valid individual-level confidence intervals
                   (honest=True, inference=True). discrete_treatment=True (binary T via
                   predict_proba on model_t) AND discrete_outcome=True (binary Y via
                   predict_proba on model_y - REQUIRED for a classifier model_y; without
                   it econml would call .predict() and residualize on hard 0/1 labels).

  S-Learner        Comparison baseline. Single LGBMRegressor with treatment as a
                   feature. ATE baseline; no overlap adjustment.
  T-Learner        Comparison baseline. Separate per-arm LGBMRegressors. ATE baseline;
                   no overlap adjustment.
  X-Learner        Overlap-aware baseline. Its combiner CATE = ps*tau0 + (1-ps)*tau1
                   concentrates weight on the common-support region through the
                   PRE-CALIBRATED propensity (Phase 2.0.1), passed via PreFittedPropensity
                   so econml reuses those exact scores without refitting.

Estimand labels (Option A - no sample_weight on any meta-learner):
  CausalForestDML -> ATE (full population), with CIs
  S/T-Learner     -> ATE baselines
  X-Learner       -> ATE/ATO-adjacent (propensity emphasis via its combiner)

CRITICAL (econml API): meta-learner OUTCOME models are REGRESSORS - econml metalearners
are conditional-mean estimators that call .predict() and difference the results, so a
classifier (returning hard labels) would yield a degenerate CATE in {-1,0,1}. The
propensity model stays a classifier (predict_proba).

Run standalone (full pipeline -> diagnostics -> fit -> comparison table):
    python -m clv_uplift.models.uplift
"""
from __future__ import annotations

import copy
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, ClassifierMixin
from lightgbm import LGBMClassifier, LGBMRegressor
from econml.dml import CausalForestDML
from econml.metalearners import SLearner, TLearner, XLearner

from clv_uplift.config import FEATURE_COLS, RANDOM_SEED
from clv_uplift.data.audit import load_raw
from clv_uplift.data.loader import clean_transactions
from clv_uplift.features.rfm import (
    build_rfm,
    assign_clv_segment,
    create_synthetic_treatment,
    split_data,
)
from clv_uplift.models.diagnostics import run_diagnostics, DiagnosticsResult

# Cosmetic only. (1) econml's cross-fitting mixes DataFrame-fitted and array-predicted
# calls, triggering sklearn's feature-NAME consistency notice (name metadata only, never
# values or column order - FEATURE_COLS fixes order throughout). (2) econml/sklearn pass
# the pre-1.6 'force_all_finite' kwarg, emitting a rename FutureWarning. Both are filtered
# precisely by message - this is NOT a blanket warnings suppression.
warnings.filterwarnings(
    "ignore", message="X does not have valid feature names", category=UserWarning,
)
warnings.filterwarnings(
    "ignore", message=".*force_all_finite.*", category=FutureWarning,
)

# --- Nuisance / base-learner configs ---------------------------------------------------
# Forest nuisances (handoff-pinned): model_t 200 trees, model_y 300 trees. Both are
# CLASSIFIERS because discrete_treatment/discrete_outcome are True (econml uses
# predict_proba on each).
FOREST_MODEL_T = dict(n_estimators=200, random_state=RANDOM_SEED, verbose=-1)
FOREST_MODEL_Y = dict(n_estimators=300, random_state=RANDOM_SEED, verbose=-1)

# Meta-learner outcome base learner: REGRESSOR (econml metalearners call .predict()).
# num_leaves=15 carried over from Phase 2.0.4 to curb tail overconfidence.
META_REGRESSOR = dict(n_estimators=300, num_leaves=15, random_state=RANDOM_SEED, verbose=-1)

# Forest structural hyperparameters NOT pinned by the handoff (confirm against the
# Phase 2.1 spec). n_estimators raised to 500 for stable honest-jackknife CIs; remaining
# structural params (min_samples_leaf, max_depth, max_samples, criterion) left at econml
# defaults.
FOREST_N_ESTIMATORS = 500

# Expected ATE band agreed with the teaching chat (true effect is ~0.4 on the logit near
# a 0.5 base rate -> ~0.06-0.12 on the probability scale). The naive diff (~0.33) is the
# confounded association; the gap to the corrected ATE is the confounding correction.
ATE_EXPECTED_BAND = (0.06, 0.12)


class PreFittedPropensity(BaseEstimator, ClassifierMixin):
    """
    Wraps a pre-fitted, calibrated propensity model (Phase 2.0.1) so econml's XLearner
    reuses those exact scores without refitting.

    BaseEstimator/ClassifierMixin make the wrapper survive econml's internal cloning.
    BUT the default sklearn clone() recurses into the inner CalibratedClassifierCV (itself
    an estimator) and returns it UNFITTED - so predict_proba later raises NotFittedError,
    because the no-op fit() intentionally does not refit. __sklearn_clone__ overrides this
    to deep-copy the wrapper, preserving the already-fitted inner model through cloning.
    classes_ is set in __init__ AND fit so it is present whenever econml reads it.
    """

    def __init__(self, fitted_model):
        self.fitted_model = fitted_model
        self.classes_ = np.array([0, 1])

    def __sklearn_clone__(self):
        # Preserve the fitted inner model through econml's clone() (default clone would
        # strip its fitted state). deepcopy copies learned attributes; the per-fold copies
        # all retain the calibrated Phase-2.0.1 propensity, which is the intent.
        return copy.deepcopy(self)

    def fit(self, X, T=None, **kwargs):
        # No-op: the model is already fitted and calibrated from Phase 2.0.1.
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        return self.fitted_model.predict_proba(X)

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


@dataclass
class UpliftModelBundle:
    """Fitted estimators + test-set CATEs/CIs. Extended in later phases (SHAP, DiCE)."""
    causal_forest: CausalForestDML
    s_learner: SLearner
    t_learner: TLearner
    x_learner: XLearner
    calibrated_propensity: object
    feature_cols: list
    ate_cf: float
    ate_cf_ci: tuple
    cate_test: dict                       # {'causal_forest','s','t','x'} -> np.ndarray
    cate_cf_ci: tuple                     # (lb_array, ub_array) per-unit, forest only
    naive_diff_test: float
    metadata: dict = field(default_factory=dict)


def _rule(char: str = "=", width: int = 70) -> str:
    return char * width


def _flat(a) -> np.ndarray:
    """Flatten an econml per-unit output to a 1-D array."""
    return np.asarray(a).reshape(-1)


def _scalar(x) -> float:
    """
    Extract a Python float from an econml population output. ate()/ate_interval() return
    arrays shaped (n_outcomes, n_treatments); under numpy 2.x, float() on a size-1
    non-0-d array raises TypeError, so we ravel and take the single element.
    """
    return float(np.asarray(x).reshape(-1)[0])


def train_uplift_models(rfm_train: pd.DataFrame, rfm_test: pd.DataFrame,
                        diag: DiagnosticsResult) -> UpliftModelBundle:
    """Fit all four estimators and evaluate CATE on the test split."""
    print("\n" + _rule())
    print("PHASE 2.1  UPLIFT ESTIMATORS")
    print(_rule())

    # Arrays at the econml boundary (avoids DataFrame feature-name warnings / quirks).
    Xtr = rfm_train[FEATURE_COLS].values
    Xte = rfm_test[FEATURE_COLS].values
    y_train = rfm_train["outcome"].values
    t_train = rfm_train["treatment"].values
    y_test = rfm_test["outcome"].values
    t_test = rfm_test["treatment"].values
    print(f"train: {len(Xtr):,} rows | test: {len(Xte):,} rows | features: {len(FEATURE_COLS)}")

    # --- CausalForestDML (primary, unweighted, ATE, individual CIs) -------------------
    print("\n[1/4] CausalForestDML (primary)")
    print("  discrete_treatment=True, discrete_outcome=True, honest=True, inference=True, cv=5")
    print(f"  model_t=LGBMClassifier(200), model_y=LGBMClassifier(300), "
          f"n_estimators={FOREST_N_ESTIMATORS}")
    causal_forest = CausalForestDML(
        model_t=LGBMClassifier(**FOREST_MODEL_T),
        model_y=LGBMClassifier(**FOREST_MODEL_Y),
        discrete_treatment=True,
        discrete_outcome=True,          # REQUIRED for the classifier model_y (see header)
        honest=True,
        inference=True,
        cv=5,
        n_estimators=FOREST_N_ESTIMATORS,
        random_state=RANDOM_SEED,
    )
    causal_forest.fit(y_train, t_train, X=Xtr, W=None)
    cate_cf = _flat(causal_forest.effect(Xte))
    lb_cf, ub_cf = causal_forest.effect_interval(Xte, alpha=0.05)
    lb_cf, ub_cf = _flat(lb_cf), _flat(ub_cf)
    ate_cf = _scalar(causal_forest.ate(Xte))
    ate_cf_lb, ate_cf_ub = causal_forest.ate_interval(Xte, alpha=0.05)
    ate_cf_ci = (_scalar(ate_cf_lb), _scalar(ate_cf_ub))
    print(f"  fitted. ATE={ate_cf:+.4f}  95% CI [{ate_cf_ci[0]:+.4f}, {ate_cf_ci[1]:+.4f}]")

    # --- S-Learner (ATE baseline) -----------------------------------------------------
    print("\n[2/4] S-Learner (ATE baseline)")
    s_learner = SLearner(overall_model=LGBMRegressor(**META_REGRESSOR))
    s_learner.fit(y_train, t_train, X=Xtr)
    cate_s = _flat(s_learner.effect(Xte))
    print(f"  fitted. ATE={cate_s.mean():+.4f}")

    # --- T-Learner (ATE baseline, no overlap weights - Option A) ----------------------
    print("\n[3/4] T-Learner (ATE baseline)")
    t_learner = TLearner(models=LGBMRegressor(**META_REGRESSOR))
    t_learner.fit(y_train, t_train, X=Xtr)
    cate_t = _flat(t_learner.effect(Xte))
    print(f"  fitted. ATE={cate_t.mean():+.4f}")

    # --- X-Learner (overlap-aware via pre-calibrated propensity) ----------------------
    print("\n[4/4] X-Learner (propensity-weighted combiner)")
    x_learner = XLearner(
        models=LGBMRegressor(**META_REGRESSOR),
        propensity_model=PreFittedPropensity(diag.calibrated_propensity),
    )
    x_learner.fit(y_train, t_train, X=Xtr)
    cate_x = _flat(x_learner.effect(Xte))
    print(f"  fitted. ATE={cate_x.mean():+.4f}")

    # --- Naive (confounded) association on the test split -----------------------------
    naive_diff = float(y_test[t_test == 1].mean() - y_test[t_test == 0].mean())

    cate_test = {"causal_forest": cate_cf, "s": cate_s, "t": cate_t, "x": cate_x}

    # --- Comparison table -------------------------------------------------------------
    print("\n" + _rule())
    print("ESTIMATOR COMPARISON (test set)")
    print(_rule())
    print(f"  Naive diff (confounded)        : {naive_diff:+.4f}   [association, NOT causal]")
    print(f"  S-Learner    (ATE baseline)    : {cate_s.mean():+.4f}")
    print(f"  T-Learner    (ATE baseline)    : {cate_t.mean():+.4f}")
    print(f"  X-Learner    (ATE/ATO-adjacent): {cate_x.mean():+.4f}")
    print(f"  CausalForestDML (ATE, PRIMARY) : {ate_cf:+.4f}   "
          f"95% CI [{ate_cf_ci[0]:+.4f}, {ate_cf_ci[1]:+.4f}]")
    print(f"\n  Confounding correction (naive - CF ATE): {naive_diff - ate_cf:+.4f}")

    # --- Forest CATE heterogeneity summary --------------------------------------------
    print("\nCausalForestDML CATE distribution (test):")
    print(f"  mean={cate_cf.mean():+.4f}  std={cate_cf.std():.4f}  "
          f"min={cate_cf.min():+.4f}  max={cate_cf.max():+.4f}")
    print(f"  share positive CATE: {(cate_cf > 0).mean() * 100:.1f}%")

    # --- Acceptance checks ------------------------------------------------------------
    print("\n" + _rule())
    print("ACCEPTANCE CHECKS")
    print(_rule())
    lo, hi = ATE_EXPECTED_BAND
    if lo <= ate_cf <= hi:
        print(f"  OK : CF-DML ATE {ate_cf:+.4f} within expected {lo:+.2f}..{hi:+.2f}.")
    else:
        print(f"  NOTE: CF-DML ATE {ate_cf:+.4f} OUTSIDE expected {lo:+.2f}..{hi:+.2f}. "
              f"Interpret before Phase 2.2.")
    if ate_cf < naive_diff:
        print(f"  OK : CF-DML ATE is below the naive diff (confounding corrected downward).")
    else:
        print(f"  NOTE: CF-DML ATE is NOT below naive - selection effect not corrected as "
              f"expected. Investigate.")
    if cate_cf.std() > 0:
        print(f"  OK : CATE shows heterogeneity (std={cate_cf.std():.4f} > 0).")
    else:
        print(f"  NOTE: CATE is constant (std=0) - no detected heterogeneity.")

    print("\n" + _rule())
    print("PHASE 2.1 COMPLETE - paste this output back.")
    print(_rule())

    return UpliftModelBundle(
        causal_forest=causal_forest,
        s_learner=s_learner,
        t_learner=t_learner,
        x_learner=x_learner,
        calibrated_propensity=diag.calibrated_propensity,
        feature_cols=list(FEATURE_COLS),
        ate_cf=ate_cf,
        ate_cf_ci=ate_cf_ci,
        cate_test=cate_test,
        cate_cf_ci=(lb_cf, ub_cf),
        naive_diff_test=naive_diff,
        metadata={
            "random_seed": RANDOM_SEED,
            "n_train": int(len(Xtr)),
            "n_test": int(len(Xte)),
            "forest_n_estimators": FOREST_N_ESTIMATORS,
            "estimands": {
                "causal_forest": "ATE (full population), individual CIs",
                "s": "ATE baseline",
                "t": "ATE baseline",
                "x": "ATE/ATO-adjacent (propensity combiner)",
            },
        },
    )


if __name__ == "__main__":
    raw = load_raw()
    clean = clean_transactions(raw)
    rfm = build_rfm(clean)
    rfm = assign_clv_segment(rfm)
    rfm = create_synthetic_treatment(rfm)
    rfm_train, rfm_test = split_data(rfm)
    diag = run_diagnostics(rfm_train, rfm_test)
    bundle = train_uplift_models(rfm_train, rfm_test, diag)
    print("\nFEATURE PIPELINE + DIAGNOSTICS + UPLIFT MODELS COMPLETE.")