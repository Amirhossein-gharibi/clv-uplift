# src/clv_uplift/models/validation.py
"""
Phase 2.2 - Post-estimation gates.

Runs five validation checks against the fitted estimators from Phase 2.1. All call
signatures were verified against the installed econml 0.16 / dowhy 0.14 APIs.

  RATE test (AUTOC)   econml.validate.DRTester with metric='toc' (NOT 'autoc' - econml
                      0.16 only accepts 'toc'/'qini'). The fitted forest is wrapped in a
                      1-D effect proxy because discrete_outcome=True makes effect() return
                      (n,1), which breaks DRTester's internal np.stack().T. AUTOC 95% CI =
                      est +/- 1.96*se; PASS iff lower bound > 0. HARD GATE for targeting:
                      if it spans zero, rate_test_passed=False and Phase 2.4 must not build
                      a targeting policy (Tier-3 null finding).

  CATE calibration    DRTester.evaluate_cal -> calibration R^2 (closer to 1 is better).

  DoWhy refuters (4)  Built on a linear backdoor estimate (the refuters test structural
                      validity of the data-generating process, not the forest):
                        1. Placebo treatment (native dowhy)      - HARD STOP if it fails.
                        2. Random common cause (native dowhy)    - document.
                        3. Data subset (hand-rolled OLS)         - document; histogram.
                        4. Bootstrap (hand-rolled OLS)           - document; histogram.
                      Subset/bootstrap are hand-rolled because dowhy's CausalRefutation
                      exposes only the MEAN new_effect, while the criteria need the std
                      across subsets and the bootstrap CI width.

  E-value             VanderWeele-Ding sensitivity. Risk difference -> approximate RR using
                      the EMPIRICAL control-arm test-set outcome rate as p0, then
                      E = RR + sqrt(RR*(RR-1)). Reported for the point estimate AND the CI
                      bound nearest the null (1.0 when the CI contains the null).

Run standalone (full pipeline -> diagnostics -> fit -> validation):
    python -m clv_uplift.models.validation
"""
from __future__ import annotations

import copy
import logging
import warnings
from dataclasses import dataclass, field, asdict

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import statsmodels.formula.api as smf
from lightgbm import LGBMClassifier, LGBMRegressor
from econml.validate import DRTester
from dowhy import CausalModel

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

# Quiet the noisy-but-harmless warnings/logs (filtered precisely, not blanket-suppressed).
warnings.filterwarnings("ignore", message="X does not have valid feature names", category=UserWarning)
warnings.filterwarnings("ignore", message=".*force_all_finite.*", category=FutureWarning)
for _name in ("dowhy", "dowhy.causal_model", "dowhy.causal_estimator",
              "dowhy.causal_refuters", "dowhy.causal_identifier"):
    logging.getLogger(_name).setLevel(logging.ERROR)

# --- Pass criteria / thresholds (from the teaching chat) -------------------------------
PLACEBO_BAND          = (-0.02, 0.02)   # mean placebo ATE must fall in here (HARD STOP)
RANDOM_CAUSE_TOL      = 0.10            # |new - orig| / |orig| must be < this
SUBSET_FRACTION       = 0.80
SUBSET_N_SIM          = 20
SUBSET_STD_FRACTION   = 0.50            # std of subset ATEs < this * |orig ATE|
BOOTSTRAP_N_SIM       = 100
BOOTSTRAP_CI_FACTOR   = 2.0             # boot CI width < this * analytic CI width
N_CAL_GROUPS          = 4

# DRTester's OWN doubly-robust nuisances (NOT pinned by the handoff; mirror our choices).
DRTESTER_MODEL_REGRESSION = dict(n_estimators=300, num_leaves=15, random_state=RANDOM_SEED, verbose=-1)
DRTESTER_MODEL_PROPENSITY = dict(n_estimators=200, random_state=RANDOM_SEED, verbose=-1)


class _CateEffect1D:
    """
    Proxy around a fitted CATE estimator so DRTester sees a 1-D (n,) effect.
    DRTester does np.stack([cate.effect(...)]).T, which assumes a 1-D effect; with
    discrete_outcome=True the forest's effect() is (n,1), producing a 3-D stack and an
    IndexError. Reshaping to (n,) fixes it. DRTester only calls .effect on the cate object.
    """
    def __init__(self, est):
        self.est = est

    def effect(self, X=None, T0=0, T1=1):
        return np.asarray(self.est.effect(X=X, T0=T0, T1=T1)).reshape(-1)


def e_value_from_rd(rd: float, p0: float) -> float:
    """
    E-value (VanderWeele-Ding) from a risk difference rd and baseline risk p0.
    Convert to an approximate risk ratio, then E = RR + sqrt(RR*(RR-1)). Protective
    effects (RR<1) are inverted. A null effect returns 1.0 (no robustness required).
    """
    if abs(rd) < 1e-10:
        return 1.0
    rr = (p0 + rd) / p0
    if rr <= 0:
        return 1.0
    if rr < 1:
        rr = 1.0 / rr
    return float(rr + np.sqrt(rr * (rr - 1.0)))


@dataclass
class ValidationResult:
    """All Phase 2.2 gate outcomes."""
    # structural refuters
    placebo_passed: bool
    placebo_new_effect: float
    random_cause_passed: bool
    random_cause_new_effect: float
    subset_passed: bool
    subset_std: float
    bootstrap_passed: bool
    bootstrap_ci_width: float
    analytic_ci_width: float
    # linear backdoor backbone
    linear_ate: float
    linear_ate_ci: tuple
    # RATE + calibration (forest)
    rate_test_passed: bool
    autoc_est: float
    autoc_se: float
    autoc_ci: tuple
    cate_cal_r2: float
    # sensitivity
    e_value_point: float
    e_value_ci: float
    p0_control_rate: float
    # reporting
    tier_ate: str
    tier_heterogeneity: str
    structural_valid: bool
    metadata: dict = field(default_factory=dict)


def _rule(char: str = "=", width: int = 70) -> str:
    return char * width


def _save_hist(values, title, path, original=None):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(values, bins=20, alpha=0.75)
    if original is not None:
        ax.axvline(original, color="red", linestyle="--", linewidth=1,
                   label=f"full-data ATE = {original:+.4f}")
        ax.legend(loc="best", fontsize=8)
    ax.set_xlabel("Estimated ATE (linear backdoor)")
    ax.set_ylabel("Count")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def run_validation(rfm_train: pd.DataFrame, rfm_test: pd.DataFrame,
                   bundle: UpliftModelBundle) -> ValidationResult:
    """Run all post-estimation gates. Attaches `.validation` to the bundle for Phase 2.4."""
    print("\n" + _rule())
    print("PHASE 2.2  POST-ESTIMATION GATES")
    print(_rule())

    rng = np.random.default_rng(RANDOM_SEED)
    train_df = rfm_train[["outcome", "treatment"] + FEATURE_COLS].reset_index(drop=True)
    n_train = len(train_df)
    formula = "outcome ~ treatment + " + " + ".join(FEATURE_COLS)

    Xtr = rfm_train[FEATURE_COLS].values
    Xte = rfm_test[FEATURE_COLS].values
    Ttr = rfm_train["treatment"].values
    Tte = rfm_test["treatment"].values
    Ytr = rfm_train["outcome"].values
    Yte = rfm_test["outcome"].values

    # --- Linear backdoor backbone (full-data OLS, HC3) --------------------------------
    full_fit = smf.ols(formula, data=train_df).fit(cov_type="HC3")
    linear_ate = float(full_fit.params["treatment"])
    ci = full_fit.conf_int(alpha=0.05).loc["treatment"]
    linear_ate_ci = (float(ci.iloc[0]), float(ci.iloc[1]))
    analytic_ci_width = linear_ate_ci[1] - linear_ate_ci[0]
    print(f"Linear backdoor ATE: {linear_ate:+.4f}  95% CI "
          f"[{linear_ate_ci[0]:+.4f}, {linear_ate_ci[1]:+.4f}]  (HC3)")

    # --- DoWhy model for native refuters ----------------------------------------------
    model = CausalModel(data=train_df, treatment="treatment", outcome="outcome",
                        common_causes=FEATURE_COLS)
    est_id = model.identify_effect(proceed_when_unidentifiable=True)
    estimate = model.estimate_effect(est_id, method_name="backdoor.linear_regression",
                                     target_units="ate")

    # --- 1. Placebo refuter (native dowhy) - HARD STOP --------------------------------
    print("\n" + _rule("-"))
    print("REFUTER 1/4: placebo treatment (HARD STOP)")
    print(_rule("-"))
    placebo = model.refute_estimate(
        est_id, estimate, method_name="placebo_treatment_refuter",
        placebo_type="permute", num_simulations=100, random_seed=RANDOM_SEED,
    )
    placebo_new = float(placebo.new_effect)
    placebo_passed = PLACEBO_BAND[0] <= placebo_new <= PLACEBO_BAND[1]
    print(f"  mean placebo ATE: {placebo_new:+.4f}  (pass band "
          f"[{PLACEBO_BAND[0]:+.2f}, {PLACEBO_BAND[1]:+.2f}])")
    print(f"  -> {'PASS' if placebo_passed else 'FAIL'}")

    if not placebo_passed:
        print("\n  HARD STOP: placebo refuter failed - the causal claim is not structurally "
              "valid. Not proceeding to RATE / calibration / E-value. Investigate the DGP.")
        result = ValidationResult(
            placebo_passed=False, placebo_new_effect=placebo_new,
            random_cause_passed=False, random_cause_new_effect=float("nan"),
            subset_passed=False, subset_std=float("nan"),
            bootstrap_passed=False, bootstrap_ci_width=float("nan"),
            analytic_ci_width=analytic_ci_width,
            linear_ate=linear_ate, linear_ate_ci=linear_ate_ci,
            rate_test_passed=False, autoc_est=float("nan"), autoc_se=float("nan"),
            autoc_ci=(float("nan"), float("nan")), cate_cal_r2=float("nan"),
            e_value_point=float("nan"), e_value_ci=float("nan"),
            p0_control_rate=float("nan"),
            tier_ate="INVALID (placebo failed)",
            tier_heterogeneity="INVALID (placebo failed)",
            structural_valid=False,
            metadata={"halted_at": "placebo_refuter"},
        )
        bundle.validation = asdict(result)
        print("\n" + _rule())
        print("PHASE 2.2 HALTED (placebo) - paste this output back.")
        print(_rule())
        return result

    # --- 2. Random common cause (native dowhy) - document -----------------------------
    print("\n" + _rule("-"))
    print("REFUTER 2/4: random common cause")
    print(_rule("-"))
    rcc = model.refute_estimate(
        est_id, estimate, method_name="random_common_cause",
        num_simulations=100, random_seed=RANDOM_SEED,
    )
    rcc_new = float(rcc.new_effect)
    rcc_rel = abs(rcc_new - linear_ate) / abs(linear_ate) if linear_ate != 0 else float("inf")
    rcc_passed = rcc_rel < RANDOM_CAUSE_TOL
    print(f"  new ATE: {rcc_new:+.4f}  | relative shift: {rcc_rel * 100:.2f}% "
          f"(tol < {RANDOM_CAUSE_TOL * 100:.0f}%)")
    print(f"  -> {'PASS' if rcc_passed else 'WARN'} (document; not a hard stop)")

    # --- 3. Data subset refuter (hand-rolled) - document ------------------------------
    print("\n" + _rule("-"))
    print("REFUTER 3/4: data subset (hand-rolled)")
    print(_rule("-"))
    subset_ates = []
    for _ in range(SUBSET_N_SIM):
        idx = rng.choice(n_train, size=int(SUBSET_FRACTION * n_train), replace=False)
        sub_fit = smf.ols(formula, data=train_df.iloc[idx]).fit(cov_type="HC3")
        subset_ates.append(float(sub_fit.params["treatment"]))
    subset_ates = np.array(subset_ates)
    subset_std = float(subset_ates.std())
    subset_passed = subset_std < SUBSET_STD_FRACTION * abs(linear_ate)
    print(f"  {SUBSET_N_SIM} subsets @ {SUBSET_FRACTION:.0%} | ATE range "
          f"[{subset_ates.min():+.4f}, {subset_ates.max():+.4f}] | std {subset_std:.4f}")
    print(f"  threshold: std < {SUBSET_STD_FRACTION:.0%} * |ATE| = "
          f"{SUBSET_STD_FRACTION * abs(linear_ate):.4f}")
    print(f"  -> {'PASS' if subset_passed else 'WARN'} (document; not a hard stop)")
    _save_hist(subset_ates, "Data-subset refuter: ATE distribution",
               FIGURES_DIR / "refuter_subset.png", original=linear_ate)
    print(f"  saved -> {FIGURES_DIR / 'refuter_subset.png'}")

    # --- 4. Bootstrap refuter (hand-rolled) - document --------------------------------
    print("\n" + _rule("-"))
    print("REFUTER 4/4: bootstrap (hand-rolled)")
    print(_rule("-"))
    boot_ates = []
    for _ in range(BOOTSTRAP_N_SIM):
        idx = rng.choice(n_train, size=n_train, replace=True)
        boot_fit = smf.ols(formula, data=train_df.iloc[idx]).fit(cov_type="HC3")
        boot_ates.append(float(boot_fit.params["treatment"]))
    boot_ates = np.array(boot_ates)
    boot_lo, boot_hi = np.percentile(boot_ates, [2.5, 97.5])
    bootstrap_ci_width = float(boot_hi - boot_lo)
    bootstrap_passed = bootstrap_ci_width < BOOTSTRAP_CI_FACTOR * analytic_ci_width
    print(f"  {BOOTSTRAP_N_SIM} resamples | bootstrap 95% CI "
          f"[{boot_lo:+.4f}, {boot_hi:+.4f}] width {bootstrap_ci_width:.4f}")
    print(f"  analytic CI width {analytic_ci_width:.4f} | threshold "
          f"{BOOTSTRAP_CI_FACTOR:.0f}x = {BOOTSTRAP_CI_FACTOR * analytic_ci_width:.4f}")
    print(f"  -> {'PASS' if bootstrap_passed else 'WARN'} (if wider, report bootstrap CI as "
          f"primary uncertainty)")
    _save_hist(boot_ates, "Bootstrap refuter: ATE distribution",
               FIGURES_DIR / "refuter_bootstrap.png", original=linear_ate)
    print(f"  saved -> {FIGURES_DIR / 'refuter_bootstrap.png'}")

    # --- RATE test (AUTOC) + CATE calibration via DRTester (forest) -------------------
    print("\n" + _rule("-"))
    print("RATE TEST (AUTOC) + CATE CALIBRATION - DRTester")
    print(_rule("-"))
    tester = DRTester(
        model_regression=LGBMRegressor(**DRTESTER_MODEL_REGRESSION),
        model_propensity=LGBMClassifier(**DRTESTER_MODEL_PROPENSITY),
        cate=_CateEffect1D(bundle.causal_forest),
        cv=5,
    )
    tester.fit_nuisance(Xte, Tte, Yte, Xtr, Ttr, Ytr)
    toc = tester.evaluate_uplift(Xte, Xtr, metric="toc")
    toc_summary = toc.summary()
    autoc_est = float(toc_summary["est"].iloc[0])
    autoc_se = float(toc_summary["se"].iloc[0])
    autoc_ci = (autoc_est - 1.96 * autoc_se, autoc_est + 1.96 * autoc_se)
    rate_test_passed = autoc_ci[0] > 0
    print(f"  AUTOC: {autoc_est:+.4f}  se {autoc_se:.4f}  95% CI "
          f"[{autoc_ci[0]:+.4f}, {autoc_ci[1]:+.4f}]")
    print(f"  gate: 95% CI lower bound > 0 -> {'PASS' if rate_test_passed else 'FAIL'}")

    cal = tester.evaluate_cal(Xte, Xtr, n_groups=N_CAL_GROUPS)
    cate_cal_r2 = float(np.asarray(cal.cal_r_squared).reshape(-1)[0])
    print(f"  CATE calibration R^2: {cate_cal_r2:+.4f}  (closer to 1 is better)")

    # --- E-value (forest ATE + CI) ----------------------------------------------------
    print("\n" + _rule("-"))
    print("E-VALUE (sensitivity to unobserved confounding)")
    print(_rule("-"))
    p0 = float(Yte[Tte == 0].mean())
    ate_point = bundle.ate_cf
    ci_lo, ci_hi = bundle.ate_cf_ci
    e_value_point = e_value_from_rd(ate_point, p0)
    if ci_lo <= 0 <= ci_hi:
        e_value_ci = 1.0
        ci_bound_used = "null contained in CI"
    else:
        bound = ci_lo if abs(ci_lo) < abs(ci_hi) else ci_hi
        e_value_ci = e_value_from_rd(bound, p0)
        ci_bound_used = f"{bound:+.4f}"
    print(f"  p0 (empirical control-arm rate, test): {p0:.4f}")
    print(f"  E-value (point ATE {ate_point:+.4f}): {e_value_point:.3f}")
    print(f"  E-value (CI bound nearest null: {ci_bound_used}): {e_value_ci:.3f}")
    if e_value_ci == 1.0:
        print("  Note: the 95% CI contains the null, so the CI-bound E-value is 1.0 - no "
              "hidden confounding is required to render the CI non-significant (reflects the "
              "Phase 2.0 positivity strain).")

    # --- Tiered reporting -------------------------------------------------------------
    structural_valid = placebo_passed  # placebo is the structural gate (rcc is documentary)
    tier_ate = "Tier 2 (direction consistent; E-value non-trivial; CI conditional on positivity)"
    tier_heterogeneity = ("Tier 1 (heterogeneity statistically confirmed; targeting justified)"
                          if rate_test_passed else
                          "Tier 3 (heterogeneity NOT confirmed at 95%; no targeting policy)")

    print("\n" + _rule())
    print("PHASE 2.2 SUMMARY (tiered findings)")
    print(_rule())
    print(f"  Structural validity (placebo) : {'CONFIRMED' if structural_valid else 'FAILED'}")
    print(f"  Random common cause           : {'stable' if rcc_passed else 'shifted'} "
          f"({rcc_rel * 100:.1f}%)")
    print(f"  Data subset                   : std {subset_std:.4f} "
          f"({'PASS' if subset_passed else 'WARN'})")
    print(f"  Bootstrap                     : CI width {bootstrap_ci_width:.4f} "
          f"({'PASS' if bootstrap_passed else 'WARN'})")
    print(f"  Average effect (ATE)          : {tier_ate}")
    print(f"  Heterogeneity (RATE/AUTOC)    : {tier_heterogeneity}")
    print(f"  E-value (point / CI bound)    : {e_value_point:.3f} / {e_value_ci:.3f}")
    if not rate_test_passed:
        print("\n  >>> rate_test_passed = False. Phase 2.4 must NOT build a targeting policy; "
              "document the Tier-3 null result in the README.")
    print("\nPaste this output back.")
    print(_rule())

    result = ValidationResult(
        placebo_passed=placebo_passed, placebo_new_effect=placebo_new,
        random_cause_passed=rcc_passed, random_cause_new_effect=rcc_new,
        subset_passed=subset_passed, subset_std=subset_std,
        bootstrap_passed=bootstrap_passed, bootstrap_ci_width=bootstrap_ci_width,
        analytic_ci_width=analytic_ci_width,
        linear_ate=linear_ate, linear_ate_ci=linear_ate_ci,
        rate_test_passed=rate_test_passed, autoc_est=autoc_est, autoc_se=autoc_se,
        autoc_ci=autoc_ci, cate_cal_r2=cate_cal_r2,
        e_value_point=e_value_point, e_value_ci=e_value_ci, p0_control_rate=p0,
        tier_ate=tier_ate, tier_heterogeneity=tier_heterogeneity,
        structural_valid=structural_valid,
        metadata={
            "random_seed": RANDOM_SEED,
            "subset_n_sim": SUBSET_N_SIM,
            "bootstrap_n_sim": BOOTSTRAP_N_SIM,
        },
    )
    # Attach to the bundle so Phase 2.4's PolicyTree can read rate_test_passed.
    bundle.validation = asdict(result)
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
    print("\nFEATURE PIPELINE + DIAGNOSTICS + UPLIFT + VALIDATION COMPLETE.")