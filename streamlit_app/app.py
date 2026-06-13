# streamlit_app/app.py
# Phase 4.2 - CLV Uplift dashboard.
# Single-customer CATE estimation with honest, tier-aware presentation:
#   - sidebar raw-RFM form (CATEFeatures contract) + example-customer loader
#   - model-context panel from GET /model-info (population ATE, E-value, tier badges)
#   - /predict: CATE + 95% CI error-bar chart (zero line, breakeven line, population dot)
#   - /explain: degraded panel shown honestly (surrogate R^2 < gate -> attributions withheld)
import os

import matplotlib
matplotlib.use("Agg")                       # headless backend - required in Docker
import matplotlib.pyplot as plt

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")
BREAKEVEN = 0.025

import json
from pathlib import Path

# Phase 4.3: examples now live in a committed JSON file (data-free).
_examples_path = Path(__file__).resolve().parent.parent / "examples" / "sample_customers.json"
try:
    EXAMPLE_CUSTOMERS = json.load(open(_examples_path, encoding="utf-8"))
except (FileNotFoundError, ValueError):
    EXAMPLE_CUSTOMERS = []
    
SEGMENTS = ["Champions", "Loyal", "At-Risk", "Lost"]

st.set_page_config(page_title="CLV Uplift", layout="wide")


@st.cache_data(show_spinner=False)
def fetch_model_info():
    """Population context, fetched once and cached. Returns dict or None (503/unreachable)."""
    try:
        r = requests.get(f"{API_URL}/api/v1/model-info", timeout=5)
        return r.json() if r.status_code == 200 else None
    except Exception:
        return None


def post_json(path: str, payload: dict):
    """POST helper -> (status_code, json|None, error_str|None)."""
    try:
        r = requests.post(f"{API_URL}{path}", json=payload, timeout=15)
        try:
            return r.status_code, r.json(), None
        except ValueError:
            return r.status_code, None, r.text
    except Exception as exc:
        return None, None, str(exc)


def cate_band_color(lo: float, hi: float) -> str:
    """Green: CI entirely above breakeven. Amber: CI spans breakeven. Red: CI spans 0 or below."""
    if lo > BREAKEVEN:
        return "#2e7d32"   # green
    if lo > 0:
        return "#f9a825"   # amber
    return "#c62828"       # red


def render_cate_chart(cate: float, lo: float, hi: float, pop_ate):
    fig, ax = plt.subplots(figsize=(8, 1.9))
    color = cate_band_color(lo, hi)

    ax.errorbar(cate, 0, xerr=[[cate - lo], [hi - cate]], fmt="o", color=color,
                ecolor=color, elinewidth=3, capsize=6, markersize=10, zorder=3,
                label="Customer CATE (95% CI)")
    ax.axvline(0.0, color="#555", linestyle="-", linewidth=1, zorder=1)
    ax.text(0.0, 0.42, "no effect", ha="center", va="bottom", fontsize=8, color="#555")
    ax.axvline(BREAKEVEN, color="#888", linestyle="--", linewidth=1, zorder=1)
    ax.text(BREAKEVEN, -0.46, f"break-even {BREAKEVEN}", ha="center", va="top",
            fontsize=8, color="#888")
    if pop_ate is not None:
        ax.scatter([pop_ate], [0], marker="D", s=70, color="#1565c0", zorder=4,
                   label=f"Population ATE ({pop_ate:+.3f})")

    ax.set_yticks([])
    ax.set_ylim(-0.8, 0.8)
    lo_x = min(lo, 0.0, (pop_ate if pop_ate is not None else 0.0)) - 0.05
    hi_x = max(hi, BREAKEVEN, (pop_ate if pop_ate is not None else 0.0)) + 0.05
    ax.set_xlim(lo_x, hi_x)
    ax.set_xlabel("CATE (incremental conversion probability)")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)        # prevent the figure-accumulation memory leak


# ── Header ────────────────────────────────────────────────────────────────────────────
st.title("CLV Uplift - Causal CATE Explorer")
st.caption("Estimate a customer's incremental treatment effect (CATE) with honest, "
           "tier-aware certification. Raw RFM in; quartile scoring handled server-side.")

info = fetch_model_info()

# ── Model context panel ─────────────────────────────────────────────────────────────────
with st.container():
    if info is None:
        st.warning("Model context unavailable - the API returned no trained bundle "
                   "(/model-info 503) or could not be reached. /score still works; "
                   "train the model to enable CATE endpoints.")
    else:
        st.subheader("Model context (population-level)")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Population ATE", f"{info['ate_point']:+.3f}",
                  help=f"95% CI [{info['ate_ci_lower']:+.3f}, {info['ate_ci_upper']:+.3f}]")
        c2.metric("Confounding correction", f"+{info['confounding_correction']:.3f}",
                  help=f"Naive association {info['naive_association']:+.3f} - causal ATE "
                       f"{info['ate_point']:+.3f}")
        c3.metric("E-value (point)", f"{info['e_value_point']:.2f}",
                  help="A hidden confounder would need this RR on both treatment and outcome "
                       "to explain the effect away.")
        c4.metric("E-value (CI bound)", f"{info['e_value_ci']:.2f}",
                  help="1.00 means the 95% CI already contains the null.")

        tier_ate = info["tier_ate"]
        tier_het = info["tier_heterogeneity"]
        targeting = "CERTIFIED" if info["rate_test_passed"] else "NOT certified"
        badge = (f"ATE: Tier {tier_ate} (real, direction-consistent)  |  "
                 f"Heterogeneity: Tier {tier_het} (RATE {'passed' if info['rate_test_passed'] else 'null'})  |  "
                 f"Targeting: {targeting}")
        if info["rate_test_passed"]:
            st.success(badge)
        else:
            st.info(badge + f"  |  Positivity strain: {info['positivity_strain']}")

st.divider()

# ── Sidebar: example loader + form ──────────────────────────────────────────────────────
st.sidebar.header("Customer features")

labels = ["(manual entry)"] + [c["label"] for c in EXAMPLE_CUSTOMERS]
choice = st.sidebar.selectbox("Load example customer", labels, index=0)
preset = next((c for c in EXAMPLE_CUSTOMERS if c["label"] == choice), None)

def _d(key, fallback):
    return preset[key] if preset else fallback

customer_id = st.sidebar.text_input("Customer ID", value=(preset["label"].split(" - ")[0] if preset else "C_999"))
recency_days = st.sidebar.number_input("Recency (days since last purchase)", min_value=0,
                                       max_value=3650, value=int(_d("recency_days", 30)), step=1)
frequency = st.sidebar.number_input("Frequency (distinct purchases)", min_value=1,
                                    value=int(_d("frequency", 5)), step=1)
monetary_value = st.sidebar.number_input("Monetary value (GBP total spend)", min_value=0.0,
                                         value=float(_d("monetary_value", 500.0)), step=10.0)
cancel_rate = st.sidebar.slider("Cancel rate", min_value=0.0, max_value=1.0,
                                value=float(_d("cancel_rate", 0.0)), step=0.01)
clv_segment = st.sidebar.selectbox("CLV segment (reporting only)", SEGMENTS,
                                   index=SEGMENTS.index(_d("clv_segment", "Loyal")))

predict_clicked = st.sidebar.button("Estimate CATE", type="primary", use_container_width=True)

payload = {
    "customer_id": customer_id,
    "recency_days": int(recency_days),
    "frequency": int(frequency),
    "monetary_value": float(monetary_value),
    "cancel_rate": float(cancel_rate),
    "clv_segment": clv_segment,
}

# ── Results ─────────────────────────────────────────────────────────────────────────────
if predict_clicked:
    status, pred, err = post_json("/api/v1/predict", payload)

    if status == 503:
        st.error("CATE endpoints are unavailable (503): no trained model bundle is loaded. "
                 "Run training, then restart the API.")
    elif status != 200 or pred is None:
        st.error(f"Prediction failed (HTTP {status}). {err or ''}")
    else:
        left, right = st.columns([3, 2])

        with left:
            st.subheader(f"CATE for {pred['customer_id']}")
            render_cate_chart(pred["cate_estimate"], pred["cate_ci_lower"],
                              pred["cate_ci_upper"],
                              info["ate_point"] if info else None)

        with right:
            st.metric("CATE estimate", f"{pred['cate_estimate']:+.4f}",
                      help=f"95% CI [{pred['cate_ci_lower']:+.4f}, {pred['cate_ci_upper']:+.4f}]")
            if pred["targeting_certified"]:
                st.success("Targeting certified")
            else:
                st.error("Not certified for targeting")
            st.caption(f"Confidence: **{pred['confidence']}**")

        st.warning(pred["recommended_action"])

        # ── Explanation (honest degraded panel) ────────────────────────────────────────
        st.divider()
        st.subheader("Why this CATE? (feature attributions)")
        e_status, expl, e_err = post_json("/api/v1/explain", payload)

        if e_status == 503:
            st.info("Explanation unavailable (503): no trained bundle loaded.")
        elif e_status != 200 or expl is None:
            st.error(f"Explain failed (HTTP {e_status}). {e_err or ''}")
        elif expl["explanation_available"]:
            st.write("Top feature contributions to this customer's CATE (interventional SHAP "
                     "on the fidelity-checked surrogate):")
            feats = expl.get("top_features", [])
            if feats:
                fig, ax = plt.subplots(figsize=(7, 0.5 * len(feats) + 1))
                names = [f["feature"] for f in feats][::-1]
                vals = [f["shap_value"] for f in feats][::-1]
                colors = ["#2e7d32" if v >= 0 else "#c62828" for v in vals]
                ax.barh(range(len(names)), vals, color=colors)
                ax.set_yticks(range(len(names)))
                ax.set_yticklabels(names)
                ax.axvline(0, color="#555", linewidth=1)
                ax.set_xlabel("SHAP value (impact on predicted CATE)")
                if expl.get("baseline_cate") is not None:
                    ax.set_title(f"baseline CATE {expl['baseline_cate']:+.4f}", fontsize=9)
                fig.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
        else:
            # The live path this run: surrogate below the fidelity gate -> attributions withheld.
            st.markdown("#### Feature attributions withheld")
            r2 = expl.get("surrogate_r2")
            thr = expl.get("surrogate_r2_threshold")
            st.markdown(
                f"- Surrogate fidelity **R² = {r2:.3f}** - below the required threshold of "
                f"**{thr:.2f}**.\n"
                f"- SHAP values were **withheld** rather than computed on an unfaithful surrogate.\n"
                f"- This reflects the **positivity constraints** in the training data "
                f"({info['positivity_strain'] if info else 'high extreme-propensity fraction'})."
            )
            st.info("What this means: the model has a view of *who* benefits, but it cannot "
                    "reliably explain *why* at the individual level. Withholding the explanation "
                    "is the honest result - not a rendered-but-meaningless SHAP plot.")
            if expl.get("explanation_unavailable_reason"):
                with st.expander("Full reason from the API"):
                    st.write(expl["explanation_unavailable_reason"])
else:
    st.info("Set the customer's features in the sidebar (or load an example), then click "
            "**Estimate CATE**.")