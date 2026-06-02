"""
R&D Portfolio Correlation Calculator
Beta-binomial model for correlated binary outcomes.

Usage:
    pip install streamlit plotly
    streamlit run beta_binomial_calculator.py
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from math import lgamma, comb, log, exp

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="R&D Portfolio Correlation Calculator",
    page_icon="🧬",
    layout="wide",
)

# ── Styling ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: #f4f7f8;
        border: 1px solid #d9e1e5;
        border-radius: 12px;
        padding: 16px 20px;
        text-align: center;
    }
    .metric-label { font-size: 12px; color: #999; font-weight: 600; margin-bottom: 4px; }
    .metric-value { font-size: 32px; font-weight: 900; color: #003366; line-height: 1; }
    .metric-sub   { font-size: 12px; color: #999; margin-top: 6px; }
    .note-box {
        background: #f4f7f8;
        border-left: 4px solid #006699;
        border-radius: 6px;
        padding: 12px 16px;
        font-size: 14px;
        margin-top: 8px;
    }
    .footer {
        font-size: 12px;
        color: #999;
        margin-top: 24px;
        padding-top: 12px;
        border-top: 1px solid #d9e1e5;
    }
</style>
""", unsafe_allow_html=True)


# ── Core math ──────────────────────────────────────────────────────────────────
def log_choose(n: int, k: int) -> float:
    return lgamma(n + 1) - lgamma(k + 1) - lgamma(n - k + 1)


def beta_binomial_pmf(n: int, k: int, mean_p: float, rho: float) -> float:
    """Exact beta-binomial PMF, with boundary handling for rho=0 and rho=1."""
    if rho <= 1e-10:
        # Ordinary binomial
        return exp(log_choose(n, k) + k * log(mean_p) + (n - k) * log(1 - mean_p))
    if rho >= 0.999999:
        # All-or-nothing limit
        if k == 0:   return 1 - mean_p
        if k == n:   return mean_p
        return 0.0

    concentration = (1 / rho) - 1
    alpha = mean_p * concentration
    beta  = (1 - mean_p) * concentration

    log_p = (
        log_choose(n, k)
        + lgamma(k + alpha)
        + lgamma(n - k + beta)
        + lgamma(alpha + beta)
        - lgamma(n + alpha + beta)
        - lgamma(alpha)
        - lgamma(beta)
    )
    return exp(log_p)


def compute_distribution(n: int, mean_p: float, rho: float):
    """Return normalised PMF, CDF, and survival function arrays."""
    raw = np.array([beta_binomial_pmf(n, k, mean_p, rho) for k in range(n + 1)])
    pmf = raw / raw.sum()                              # normalise for float safety
    cdf = np.cumsum(pmf)
    sf  = 1 - np.concatenate([[0], cdf[:-1]])          # P(X >= k)
    return pmf, cdf, sf


# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("## 🧬 R&D Portfolio Correlation Calculator")
st.markdown(
    "Beta-binomial model for correlated binary (success/failure) outcomes. "
    "Correlation is induced by a shared latent success probability drawn from a Beta distribution."
)
st.divider()

# ── Inputs ─────────────────────────────────────────────────────────────────────
col_in, col_out = st.columns([1, 2], gap="large")

with col_in:
    st.markdown("**Model inputs**")

    n = st.slider("Number of assets (n)", min_value=1, max_value=50, value=10, step=1)
    p = st.slider("Per-asset PoS (%)", min_value=1, max_value=99, value=20, step=1) / 100
    rho = st.slider("Pairwise correlation ρ (%)", min_value=0, max_value=100, value=0, step=1) / 100
    target = st.slider("Success threshold (≥ k)", min_value=1, max_value=n, value=2, step=1)

    # Derived parameters
    if rho > 1e-10:
        concentration = (1 / rho) - 1
        alpha = p * concentration
        beta  = (1 - p) * concentration
        st.markdown(
            f"**Implied Beta params:** α = {alpha:.2f}, β = {beta:.2f}, "
            f"concentration = {concentration:.2f}"
        )
    else:
        st.markdown("**ρ = 0:** reduces to standard Binomial(*n*, *p*)")

# ── Compute ────────────────────────────────────────────────────────────────────
pmf, cdf, sf = compute_distribution(n, p, rho)
ks    = np.arange(n + 1)
mean  = float(np.dot(ks, pmf))
var   = float(np.dot((ks - mean) ** 2, pmf))
sd    = var ** 0.5
p0    = pmf[0]
p_at_least_one    = 1 - p0
p_at_least_target = float(sf[target])

# ── Results ────────────────────────────────────────────────────────────────────
with col_out:

    # Metric cards
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(f"""
        <div class="metric-card">
          <div class="metric-label">P(zero successes)</div>
          <div class="metric-value">{p0:.1%}</div>
          <div class="metric-sub">Total failure</div>
        </div>""", unsafe_allow_html=True)
    with m2:
        st.markdown(f"""
        <div class="metric-card">
          <div class="metric-label">P(≥1 success)</div>
          <div class="metric-value">{p_at_least_one:.1%}</div>
          <div class="metric-sub">Any success</div>
        </div>""", unsafe_allow_html=True)
    with m3:
        st.markdown(f"""
        <div class="metric-card">
          <div class="metric-label">P(≥{target} successes)</div>
          <div class="metric-value">{p_at_least_target:.1%}</div>
          <div class="metric-sub">Threshold outcome</div>
        </div>""", unsafe_allow_html=True)
    with m4:
        st.markdown(f"""
        <div class="metric-card">
          <div class="metric-label">Expected successes</div>
          <div class="metric-value">{mean:.1f}</div>
          <div class="metric-sub">SD: {sd:.1f}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Chart ──────────────────────────────────────────────────────────────────
    bar_colors = [
        "#cc1f36" if k == 0 else "#006699" if k >= target else "#89d1d6"
        for k in ks
    ]
    hover = [f"k = {k}<br>P = {pmf[k]:.2%}<br>P(≥{k}) = {sf[k]:.2%}" for k in ks]

    fig = go.Figure(go.Bar(
        x=ks, y=pmf,
        marker_color=bar_colors,
        hovertext=hover,
        hoverinfo="text",
        showlegend=False,
    ))
    fig.update_layout(
        title="Distribution of portfolio successes",
        xaxis_title="Number of successes (k)",
        yaxis_title="Probability",
        yaxis_tickformat=".0%",
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=40, r=20, t=40, b=40),
        height=320,
        font=dict(family="Arial", size=13),
        xaxis=dict(tickmode="linear", dtick=1 if n <= 20 else 2),
    )
    fig.add_annotation(
        text=f"<span style='color:#cc1f36'>■</span> Zero  "
             f"<span style='color:#89d1d6'>■</span> Below threshold  "
             f"<span style='color:#006699'>■</span> At/above threshold (≥{target})",
        xref="paper", yref="paper", x=0, y=1.08,
        showarrow=False, font=dict(size=11), align="left",
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Interpretation ─────────────────────────────────────────────────────────
    corr_text = "With independent outcomes" if rho == 0 else f"With {rho:.0%} shared-risk correlation"
    st.markdown(f"""
    <div class="note-box">
    <strong>Interpretation:</strong> {corr_text}, this portfolio has a
    <strong>{p0:.1%}</strong> probability of zero successes and a
    <strong>{p_at_least_target:.1%}</strong> probability of achieving at least {target} success(es).
    Correlation does not change the expected number of successes materially,
    but it widens the outcome distribution — increasing both tail risks.
    </div>
    """, unsafe_allow_html=True)

    # ── PMF table (expandable) ─────────────────────────────────────────────────
    with st.expander("Show probability table"):
        import pandas as pd
        df = pd.DataFrame({
            "Successes (k)":  ks,
            "P(X = k)":       [f"{v:.4%}" for v in pmf],
            "P(X ≤ k)":       [f"{v:.4%}" for v in cdf],
            "P(X ≥ k)":       [f"{v:.4%}" for v in sf],
        })
        st.dataframe(df, use_container_width=True, hide_index=True)

        csv = df.to_csv(index=False)
        st.download_button("Download CSV", csv, "beta_binomial_distribution.csv", "text/csv")

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
Model note: This tool uses a beta-binomial model for positively correlated Bernoulli outcomes.
At ρ = 0 it reduces to Binomial(n, p); at ρ → 1 it collapses to an all-or-nothing Bernoulli.
This is a decision-support approximation — not a substitute for asset-specific PTRS analysis,
scenario modelling, or full portfolio valuation.
</div>
""", unsafe_allow_html=True)
