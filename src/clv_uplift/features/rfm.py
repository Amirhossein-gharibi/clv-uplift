# src/clv_uplift/features/rfm.py
"""
Phase 1.2 / 1.3 - RFM features, CLV segmentation, synthetic treatment + outcome,
and the stratified train/test split.

Pipeline order (confirmed with the teaching chat - each step feeds the next):
    load_raw()
      -> clean_transactions()          # produces the cancel_count column
      -> build_rfm()                   # recency, frequency, monetary, r/f/m scores, cancel_rate
      -> assign_clv_segment()          # clv_segment (reporting only; never a model feature)
      -> create_synthetic_treatment()  # treatment AND outcome (same function)
      -> split_data()                  # stratified on treatment x outcome

Treatment assignment is STOCHASTIC (sigmoid propensity), not a hard threshold. A
deterministic rule (recency<=90 AND monetary>=500) makes treatment a deterministic
function of observed covariates, which destroys positivity/overlap and renders
DML-family estimators non-identified (Phase 2.0 measured a 99.71% extreme-propensity
fraction under the deterministic rule). The stochastic rule keeps treatment CORRELATED
with the covariates that also drive the outcome (preserving confounding) while spreading
propensities across (0,1) (restoring overlap) - modelling a realistic CRM contact pool
with holdouts, not retreating from confounding.

Run standalone (full chain, prints every verification step):
    python -m clv_uplift.features.rfm
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import logging
from sklearn.model_selection import train_test_split

from clv_uplift.config import (
    SNAPSHOT_DATE_STR,
    RFM_QUANTILES,
    CLV_SEGMENTS,
    TREATMENT_RECENCY_DAYS,
    TREATMENT_SPEND_THRESHOLD,
    RANDOM_SEED,
    TEST_SIZE,
    FEATURE_COLS,
)
from clv_uplift.data.audit import load_raw
from clv_uplift.data.loader import clean_transactions

# Outcome logit intercept. Centers the synthetic base rate near 0.50 so the treatment
# effect is visible on a meaningful probability scale (without it the base rate pins
# near 0.90 - a ceiling effect).
logger = logging.getLogger(__name__)
OUTCOME_LOGIT_INTERCEPT = -2.5

# Stochastic treatment-assignment coefficients (sigmoid propensity).
#   TREAT_BETA0      : intercept; THIS is the knob to tune (more negative -> lower rate)
#                      in 0.25 steps if the treatment rate falls outside the 25-40% band.
#                      Set to -1.5: at -0.5 the rate came out 51.55% because the monetary
#                      term is centered on the config threshold (500) while the sample
#                      median spend (~668) is higher, giving a positive mean monetary_z
#                      (~+0.37 -> ~+0.55 added to every logit). -1.5 offsets that to target
#                      a ~30% rate.
#   TREAT_BETA_*     : slopes on standardized recency / log-monetary; 1.5 gives meaningful
#                      but non-degenerate propensity variation. These (not BETA0) mainly
#                      drive the extreme-propensity fraction and the SMD magnitudes.
# recency and monetary are standardized AROUND the config threshold values, so those
# constants remain meaningful as centering points (not cutoffs). monetary uses log1p
# because raw spend is heavily right-skewed (a 50,000 customer is not 100x more
# treatable than a 500 one).
TREAT_BETA0         = -1.5
TREAT_BETA_RECENCY  = 1.5
TREAT_BETA_MONETARY = 1.5

# Acceptance targets agreed with the teaching chat.
TREATMENT_RATE_TARGET = (0.25, 0.40)   # tuning band for the stochastic rule (subset of
                                       # the handoff's 20-50% Phase 1.3 gate)
OUTCOME_RATE_TARGET   = (0.40, 0.60)   # overall positive rate should land in this band
TREATED_FAILURE_FLOOR = 150            # min members in the "treated, did-not-convert" cell


def _rule(char: str = "=", width: int = 70) -> str:
    return char * width


def build_rfm(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate cleaned transaction rows to one row per customer with RFM features.

    Input  : cleaned transaction-level DataFrame (InvoiceNo, CustomerID, InvoiceDate,
             Quantity, UnitPrice, cancel_count).
    Output : customer-level DataFrame INDEXED BY CustomerID with columns
             recency_days, frequency, monetary_value, r_score, f_score, m_score,
             rfm_score, cancel_count, cancel_rate.

    Scoring choices (not pinned by the handoff - flagged to the teaching chat):
      * Quartile scores via rank(method="first") -> qcut, which guarantees
        RFM_QUANTILES unique bins despite heavy ties in frequency (plain qcut raises
        "Bin edges must be unique").
      * Direction: most-recent (lowest recency_days) gets the HIGHEST r_score; highest
        frequency / monetary get the highest f_score / m_score, so a high score means
        "better customer" - which the synthetic-outcome logits assume.
      * Labels derived from RFM_QUANTILES, not hardcoded to 1..4.
    """
    print(_rule())
    print("BUILD RFM (build_rfm)")
    print(_rule())
    n_tx = len(df)
    print(f"Transactions in : {n_tx:,}")

    snapshot = pd.Timestamp(SNAPSHOT_DATE_STR)
    work = df.copy()
    work["amount"] = work["Quantity"] * work["UnitPrice"]

    rfm = work.groupby("CustomerID").agg(
        last_purchase=("InvoiceDate", "max"),
        frequency=("InvoiceNo", "nunique"),
        monetary_value=("amount", "sum"),
        cancel_count=("cancel_count", "first"),
    )
    rfm["recency_days"] = (snapshot - pd.to_datetime(rfm["last_purchase"])).dt.days
    rfm = rfm.drop(columns=["last_purchase"])
    print(f"Customers out   : {len(rfm):,}")

    # Quartile scores (rank-then-qcut for tie safety; labels derived from config).
    high_labels    = list(range(1, RFM_QUANTILES + 1))      # ascending: high value -> high score
    recency_labels = list(range(RFM_QUANTILES, 0, -1))      # descending: low recency -> high score
    rfm["r_score"] = pd.qcut(rfm["recency_days"].rank(method="first"),
                             q=RFM_QUANTILES, labels=recency_labels).astype(int)
    rfm["f_score"] = pd.qcut(rfm["frequency"].rank(method="first"),
                             q=RFM_QUANTILES, labels=high_labels).astype(int)
    rfm["m_score"] = pd.qcut(rfm["monetary_value"].rank(method="first"),
                             q=RFM_QUANTILES, labels=high_labels).astype(int)
    rfm["rfm_score"] = rfm["r_score"] + rfm["f_score"] + rfm["m_score"]

    # cancel_rate = cancelled invoices / all invoice events (purchases + cancellations).
    # Denominator >= 1 because every retained customer has >= 1 purchase invoice.
    rfm["cancel_count"] = rfm["cancel_count"].fillna(0).astype(int)
    rfm["cancel_rate"] = rfm["cancel_count"] / (rfm["frequency"] + rfm["cancel_count"])

    print("\nRFM summary (describe):")
    print(rfm[["recency_days", "frequency", "monetary_value", "cancel_rate"]]
          .describe().round(2).to_string())
    print("\nScore distributions:")
    for col in ["r_score", "f_score", "m_score"]:
        counts = rfm[col].value_counts().sort_index()
        print(f"  {col}: " + ", ".join(f"{k}:{v}" for k, v in counts.items()))
    return rfm


def _segment(row: pd.Series) -> str:
    """
    Canonical CLV segmentation rule (Document 7). First match wins - priority order is
    significant: Champions (all three high) is the strict subset, Loyal relaxes recency,
    At-Risk captures customers who were valuable but have gone quiet, Lost is residual.
    """
    r, f, m = row["r_score"], row["f_score"], row["m_score"]
    if r >= 3 and f >= 3 and m >= 3:
        return "Champions"
    elif f >= 3 and m >= 3:
        return "Loyal"
    elif r <= 2 and (f >= 3 or m >= 3):
        return "At-Risk"
    else:
        return "Lost"


def assign_clv_segment(rfm: pd.DataFrame) -> pd.DataFrame:
    """
    Map each customer to one of CLV_SEGMENTS = [Champions, Loyal, At-Risk, Lost] using
    the canonical three-dimensional priority rule in _segment().

    clv_segment is reporting-only: NOT a model feature (excluded from FEATURE_COLS by
    design), used only for GATE grouping (Phase 2.4) and the fairness audit, and OFF the
    critical path for the split (strat_key uses treatment and outcome, not clv_segment).
    Applied row-wise so the first-match-wins precedence in _segment() is preserved
    exactly; on ~4k customers this is instant.
    """
    print(_rule())
    print("ASSIGN CLV SEGMENT (assign_clv_segment)")
    print(_rule())
    n_before = len(rfm)
    rfm = rfm.copy()
    seg = rfm.apply(_segment, axis=1)
    rfm["clv_segment"] = pd.Categorical(seg, categories=CLV_SEGMENTS)

    print(f"Rows before: {n_before:,} | Rows after: {len(rfm):,} (unchanged)")
    print("Segment distribution:")
    dist = rfm["clv_segment"].value_counts().reindex(CLV_SEGMENTS).fillna(0).astype(int)
    for seg_name, cnt in dist.items():
        print(f"  {seg_name:10s}: {cnt:,} ({cnt / len(rfm) * 100:.1f}%)")
    return rfm


def create_synthetic_treatment(rfm: pd.DataFrame) -> pd.DataFrame:
    """
    Assign STOCHASTIC treatment via a sigmoid propensity, then simulate the binary
    outcome (same function).

    Treatment (stochastic): a logistic propensity in standardized recency and
    log-monetary, both centered on the config thresholds (TREATMENT_RECENCY_DAYS,
    TREATMENT_SPEND_THRESHOLD). Recent / higher-spend customers are MORE LIKELY to be
    treated, not certain to be - which preserves confounding while restoring overlap.

    Outcome (synthetic, binary): logistic in r/f/m scores plus a genuine positive
    treatment effect (coefficient 0.4) and Gaussian noise, with intercept
    OUTCOME_LOGIT_INTERCEPT = -2.5 centering the base rate near 0.50.

    A single RNG seeded with RANDOM_SEED draws, in order: treatment Bernoulli, outcome
    noise, outcome Bernoulli - so the whole DGP is reproducible.
    """
    print(_rule())
    print("CREATE SYNTHETIC TREATMENT + OUTCOME (create_synthetic_treatment)")
    print(_rule())
    n_before = len(rfm)
    rfm = rfm.copy()
    rng = np.random.default_rng(RANDOM_SEED)

    # --- Stochastic treatment (sigmoid propensity) -------------------------------
    recency_z = -(rfm["recency_days"] - TREATMENT_RECENCY_DAYS) / rfm["recency_days"].std()
    monetary_z = (
        (np.log1p(rfm["monetary_value"]) - np.log1p(TREATMENT_SPEND_THRESHOLD))
        / np.log1p(rfm["monetary_value"]).std()
    )
    treat_logit = (
        TREAT_BETA0
        + TREAT_BETA_RECENCY * recency_z
        + TREAT_BETA_MONETARY * monetary_z
    )
    p_treat = 1.0 / (1.0 + np.exp(-treat_logit))
    rfm["treatment"] = (rng.uniform(size=len(rfm)) < p_treat).astype(int)

    # --- Synthetic binary outcome ------------------------------------------------
    logits = (
        OUTCOME_LOGIT_INTERCEPT
        + 0.5 * rfm["r_score"].astype(float)
        + 0.3 * rfm["f_score"].astype(float)
        + 0.2 * rfm["m_score"].astype(float)
        + 0.4 * rfm["treatment"].astype(float)
        + rng.normal(0, 0.5, size=len(rfm))
    )
    prob_y = 1.0 / (1.0 + np.exp(-logits))
    rfm["outcome"] = (rng.uniform(size=len(rfm)) < prob_y).astype(int)

    treat_rate = rfm["treatment"].mean()
    outcome_rate = rfm["outcome"].mean()
    print(f"Rows before: {n_before:,} | Rows after: {len(rfm):,} (unchanged)")
    print(f"Propensity range: [{p_treat.min():.4f}, {p_treat.max():.4f}]  "
          f"(stochastic - should span the interior, not pin at 0/1)")

    print(f"Treatment rate : {treat_rate * 100:.2f}%  (n_treated={int(rfm['treatment'].sum()):,})")
    lo_t, hi_t = TREATMENT_RATE_TARGET
    if not (lo_t <= treat_rate <= hi_t):
        print(f"  WARNING: treatment rate {treat_rate * 100:.2f}% is OUTSIDE the "
              f"{lo_t * 100:.0f}-{hi_t * 100:.0f}% target. Adjust TREAT_BETA0 in 0.25 "
              f"steps (more negative -> lower rate) and re-run.")
    else:
        print(f"  OK: treatment rate is within the {lo_t * 100:.0f}-{hi_t * 100:.0f}% "
              f"target band.")

    print(f"Outcome rate   : {outcome_rate * 100:.2f}%  (n_positive={int(rfm['outcome'].sum()):,})")
    lo_o, hi_o = OUTCOME_RATE_TARGET
    if not (lo_o <= outcome_rate <= hi_o):
        print(f"  WARNING: outcome rate {outcome_rate * 100:.2f}% is OUTSIDE the "
              f"{lo_o * 100:.0f}-{hi_o * 100:.0f}% target. Revisit the logit intercept.")
    else:
        print(f"  OK: outcome rate is within the {lo_o * 100:.0f}-{hi_o * 100:.0f}% "
              f"target band.")

    by_arm = rfm.groupby("treatment")["outcome"].mean()
    c, t = by_arm.get(0, float("nan")), by_arm.get(1, float("nan"))
    print(f"Naive outcome by arm: control={c:.3f}, treated={t:.3f}, naive diff={t - c:+.3f}")
    print("  (Naive diff is confounded - NOT the causal effect; the uplift model corrects it.)")
    return rfm


def split_data(rfm: pd.DataFrame):
    """
    Stratified train/test split on the treatment x outcome key.

    Binary treatment x binary outcome yields four natural strata ("0_0", "0_1", "1_0",
    "1_1") - no qcut needed (the blueprint's qcut(outcome, q=4) was an error: qcut on a
    binary variable raises "Bin edges must be unique"). Returns the full train/test
    frames (indexed by CustomerID) so downstream phases slice FEATURE_COLS, treatment,
    and outcome as needed.
    """
    print(_rule())
    print("STRATIFIED TRAIN/TEST SPLIT (split_data)")
    print(_rule())
    rfm = rfm.copy()
    rfm["strat_key"] = rfm["treatment"].astype(str) + "_" + rfm["outcome"].astype(str)

    print("Stratum sizes (treatment_outcome):")
    strat_counts = rfm["strat_key"].value_counts().sort_index()
    for k, v in strat_counts.items():
        print(f"  {k}: {v:,}")
    if (strat_counts < 2).any():
        print("  WARNING: a stratum has < 2 members; stratified split will fail. Investigate.")

    treated_fail = int(strat_counts.get("1_0", 0))
    if treated_fail < TREATED_FAILURE_FLOOR:
        print(f"  WARNING: treated-failure cell (1_0) has only {treated_fail} members "
              f"(< {TREATED_FAILURE_FLOOR} target). Treated-arm outcome model will be "
              f"data-starved.")
    else:
        print(f"  OK: treated-failure cell (1_0) has {treated_fail} members "
              f"(>= {TREATED_FAILURE_FLOOR}).")

    rfm_train, rfm_test = train_test_split(
        rfm,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=rfm["strat_key"],
    )

    # Hard assertion - no customer appears in both splits.
    leak = set(rfm_test.index) & set(rfm_train.index)
    assert len(leak) == 0, f"Index leak detected: {len(leak)} customers in both splits"

    n_total = len(rfm)
    print(f"\nTotal customers: {n_total:,}")
    print(f"Train: {len(rfm_train):,} ({len(rfm_train) / n_total * 100:.1f}%)")
    print(f"Test : {len(rfm_test):,} ({len(rfm_test) / n_total * 100:.1f}%)")
    print("Index leak check: PASSED (0 overlap)")

    print("\nstrat_key proportions (should match across train/test):")
    train_prop = rfm_train["strat_key"].value_counts(normalize=True).sort_index()
    test_prop  = rfm_test["strat_key"].value_counts(normalize=True).sort_index()
    for k in strat_counts.index:
        print(f"  {k}: train={train_prop.get(k, 0) * 100:.2f}%  test={test_prop.get(k, 0) * 100:.2f}%")

    # Feature-set integrity.
    missing_feats = [c for c in FEATURE_COLS if c not in rfm_train.columns]
    forbidden = [c for c in ["rfm_score", "treatment", "outcome", "CustomerID",
                             "clv_segment", "strat_key"] if c in FEATURE_COLS]
    assert not missing_feats, f"FEATURE_COLS missing from split: {missing_feats}"
    assert not forbidden, f"Forbidden columns leaked into FEATURE_COLS: {forbidden}"
    print(f"\nFEATURE_COLS ({len(FEATURE_COLS)}): {FEATURE_COLS}")
    print("Feature-set integrity: PASSED (all present; no forbidden columns)")

    return rfm_train, rfm_test

class RFMBinner:
    """
    Modal-value lookup binner: maps a raw RFM value to the score build_rfm() actually
    assigned, looked up deterministically at serving time.

    Why modal lookup (not value-based quantile edges): build_rfm() scores via
    pd.qcut(X.rank(method="first"), q=4) - rank-based EQUAL-COUNT binning that splits ties
    across adjacent bins. frequency is a heavily-tied integer (q25=1, q50=2, q75=5), so its
    value-quantile edges collapse to [1, 2, 5, 206] -> only 3 bins, ~50% of customers pinned
    to f_score=1, a scale the forest never trained on. Modal lookup instead records, per
    distinct training value, the MAJORITY score build_rfm gave it - preserving the full 1..4
    scale and reproducing training for every value except the irreducible within-tie minority
    (where the rank split was arbitrary row-order anyway). That residual is measured and
    logged at fit time, not hidden.

    Unseen serving values (e.g. a monetary sum between two training values): TRUE-nearest
    training value by sorted position (ties -> lower value). For near-continuous monetary
    this is 1-NN on the training distribution; for integer frequency a never-seen count gets
    the nearer observed tier's score.

    Recency inversion is automatic: the lookup stores build_rfm's already-inverted r_score
    (lower days -> higher score). An out-of-range-LOW recency (more recent than any training
    customer) lands at searchsorted index 0 -> smallest known recency -> r_score 4 (correct);
    out-of-range-HIGH -> r_score 1. No fillna special-casing, no worst-score bug.

    Fit on TRAINING data only - never the full set or test.
    """

    def __init__(self, n_quantiles: int = 4):
        # n_quantiles is informational; the score levels come from the training scores.
        self.n_quantiles = n_quantiles
        self.modal_lookup_: dict = {}     # {feature: {raw_value: modal_score}}
        self.known_values_: dict = {}     # {feature: sorted np.ndarray of training values}
        self.known_scores_: dict = {}     # {feature: scores aligned to known_values_}
        self.mismatch_rates_: dict = {}   # {feature: fraction differing from training scores}

    def fit(self, X_raw: "pd.DataFrame", scores_train: "pd.DataFrame") -> "RFMBinner":
        for raw_col, score_col in [("recency_days", "r_score"),
                                   ("frequency", "f_score"),
                                   ("monetary_value", "m_score")]:
            ld = pd.DataFrame({"value": X_raw[raw_col].values,
                               "score": scores_train[score_col].values})
            modal = ld.groupby("value")["score"].agg(
                lambda s: int(s.mode().iloc[0])  # majority score; ties -> lower (mode sorts)
            ).to_dict()
            self.modal_lookup_[raw_col] = modal
            sv = np.array(sorted(modal.keys()))
            self.known_values_[raw_col] = sv
            self.known_scores_[raw_col] = np.array([modal[v] for v in sv], dtype=int)

            binner_scores = X_raw[raw_col].map(modal).values
            mismatch = float((scores_train[score_col].values != binner_scores).mean())
            self.mismatch_rates_[raw_col] = mismatch
            logger.info(
                "RFMBinner fit - %s: %d unique values -> modal lookup. "
                "Mismatch vs rank-based training scores: %.1f%% "
                "(irreducible tie minority - acceptable).",
                raw_col, len(modal), mismatch * 100,
            )
        return self

    def _lookup(self, values: "np.ndarray", feature: str) -> "np.ndarray":
        modal = self.modal_lookup_[feature]
        kv = self.known_values_[feature]
        ks = self.known_scores_[feature]
        res = np.empty(len(values), dtype=int)
        for i, v in enumerate(values):
            if v in modal:
                res[i] = modal[v]
                continue
            idx = int(np.searchsorted(kv, v))
            if idx <= 0:
                res[i] = ks[0]
            elif idx >= len(kv):
                res[i] = ks[-1]
            else:
                lo, hi = kv[idx - 1], kv[idx]
                res[i] = ks[idx - 1] if (v - lo) <= (hi - v) else ks[idx]  # true nearest
        return res

    def transform(self, X_raw: "pd.DataFrame") -> "pd.DataFrame":
        if not self.modal_lookup_:
            raise RuntimeError("RFMBinner.fit() must be called before transform().")
        out = pd.DataFrame(index=X_raw.index)
        out["recency_days"] = X_raw["recency_days"].values
        out["frequency"] = X_raw["frequency"].values
        out["monetary_value"] = X_raw["monetary_value"].values
        out["r_score"] = self._lookup(X_raw["recency_days"].values, "recency_days")
        out["f_score"] = self._lookup(X_raw["frequency"].values, "frequency")
        out["m_score"] = self._lookup(X_raw["monetary_value"].values, "monetary_value")
        out["cancel_rate"] = (
            X_raw["cancel_rate"].values if "cancel_rate" in X_raw.columns else 0.0
        )
        return out[FEATURE_COLS]

if __name__ == "__main__":
    raw = load_raw()
    clean = clean_transactions(raw)
    rfm = build_rfm(clean)
    rfm = assign_clv_segment(rfm)
    rfm = create_synthetic_treatment(rfm)
    rfm_train, rfm_test = split_data(rfm)
    print("\n" + _rule())
    print("FEATURE PIPELINE COMPLETE - paste this output back.")
    print(_rule())