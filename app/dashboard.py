"""
Interactive credit-risk dashboard (Streamlit).

    streamlit run app/dashboard.py

Five tabs cover the full workflow: portfolio overview, WoE/IV feature analysis,
model performance (scorecard vs XGBoost), the points scorecard with a live
borrower-scoring form, and an IFRS 9 ECL cockpit with scenario sliders that
re-stage the book and recompute expected loss in real time.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import roc_curve

from credit_risk import config, ecl
from credit_risk.config import ECLParams
from credit_risk.datasets import resolve_dataset
from credit_risk.pipeline import load_artifacts, load_reports, train_and_save

st.set_page_config(page_title="Credit Risk — PD & IFRS 9 ECL", page_icon="🏦", layout="wide")
ACCENT = "#2E4057"

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(180deg, #f5f7fb 0%, #edf3ff 100%);
    }
    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 2rem;
        max-width: 1600px;
    }
    [data-testid="stSidebar"] {
        width: 320px;
        background: linear-gradient(180deg, #16263d 0%, #203750 100%);
        border-right: 1px solid rgba(255,255,255,0.08);
    }
    [data-testid="stSidebar"] > div {
        padding-top: 1rem;
    }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] label, [data-testid="stSidebar"] .stRadio {
        color: #f4f8ff !important;
    }
    [data-testid="stSidebar"] .stRadio > div {
        display: flex;
        flex-direction: column;
        gap: 0.55rem;
        width: 100%;
    }
    [data-testid="stSidebar"] .stRadio > div > label {
        width: 100% !important;
        display: flex !important;
        align-items: center;
        justify-content: flex-start;
    }
    [data-testid="stSidebar"] .stRadio label {
        width: 100% !important;
        background: rgba(255,255,255,0.08);
        border: 1px solid rgba(255,255,255,0.12);
        border-radius: 14px;
        padding: 0.75rem 0.8rem !important;
        font-size: 1.06rem !important;
        font-weight: 600;
        box-shadow: 0 4px 10px rgba(0,0,0,0.08);
        transition: transform 0.18s ease, background 0.18s ease, box-shadow 0.18s ease;
    }
    [data-testid="stSidebar"] .stRadio label:hover {
        background: rgba(255,255,255,0.14);
        transform: translateX(2px);
        box-shadow: 0 6px 14px rgba(0,0,0,0.12);
    }
    [data-testid="stSidebar"] .stRadio input:checked + div {
        color: #ffffff !important;
    }
    [data-testid="stSidebar"] .stRadio label:has(input:checked) {
        background: linear-gradient(90deg, rgba(113,159,255,0.35), rgba(68,122,244,0.25));
        border-color: rgba(172, 205, 255, 0.55);
        box-shadow: 0 8px 20px rgba(55, 108, 214, 0.2);
    }
    [data-testid="stSidebar"] .stMarkdownContainer {
        padding: 0 0.2rem 0.8rem 0.2rem;
    }
    .stForm {
        background: rgba(255,255,255,0.74);
        border: 1px solid rgba(25,50,97,0.08);
        border-radius: 18px;
        padding: 1rem 1rem 0.4rem 1rem;
        box-shadow: 0 10px 28px rgba(47, 76, 116, 0.08);
    }
    div[data-testid="stMetric"] {
        background: rgba(255,255,255,0.74);
        border: 1px solid rgba(25,50,97,0.08);
        border-radius: 16px;
        box-shadow: 0 8px 22px rgba(26,46,79,0.06);
        padding: 0.85rem 0.9rem;
    }
    div[data-testid="stDataFrame"] {
        border-radius: 16px;
        overflow: hidden;
        box-shadow: 0 8px 22px rgba(26,46,79,0.07);
    }
    .css-1v0mbdj, .css-1p05t8w, .css-2trqyj {
        background: rgba(255,255,255,0.7);
        border-radius: 14px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading data, models and precomputed reports…")
def load_all():
    bundle = resolve_dataset()
    reports = load_reports()
    try:
        encoder, scorecard, gbm, rf = load_artifacts()
        if reports["scored_portfolio"].empty:
            # Models exist but the dashboard report artifacts don't (e.g. an
            # older run) — force a retrain so everything is saved together.
            raise FileNotFoundError("Precomputed report artifacts are missing")
    except (FileNotFoundError, OSError):
        # `bundle` above is the full, unsampled portfolio (for display). Training
        # itself should use the same manageable sample size as train.py — not
        # the full multi-million-row file — so this stays consistent whether
        # it's triggered here or from the command line.
        train_bundle = resolve_dataset(sample_size=config.TRAIN_SAMPLE_SIZE)
        result = train_and_save(train_bundle.df, spec=train_bundle.spec)
        encoder, scorecard, gbm, rf = result.encoder, result.scorecard, result.gbm, result.rf
        reports = load_reports()

    df = bundle.df
    scored = reports["scored_portfolio"]
    pd_sc = scored["pd_scorecard"].to_numpy()
    score = scored["score"].to_numpy()
    rating = scored["rating"].to_numpy()
    pd_gbm = scored["pd_gbm"].to_numpy()
    pd_rf = scored["pd_rf"].to_numpy()

    metrics = {}
    if config.METRICS_JSON.exists():
        metrics = json.loads(config.METRICS_JSON.read_text(encoding="utf-8"))
    return bundle, encoder, scorecard, gbm, rf, df, pd_sc, score, rating, pd_gbm, pd_rf, metrics, reports


try:
    bundle, encoder, scorecard, gbm, rf, df, pd_sc, score, rating, pd_gbm, pd_rf, metrics, reports = load_all()
except FileNotFoundError as exc:
    st.error(str(exc))
    st.stop()

st.title("🏦 Credit Risk — PD Scorecard & IFRS 9 ECL")
st.caption(
    f"Data source: **{bundle.name}** · {len(df):,} loans · "
    f"default rate {df[bundle.spec.target].mean():.1%}"
)

# Sidebar navigation (improved UI)
st.sidebar.title("🏦 Credit Risk")
# st.sidebar.markdown("Choose a page to view — 'Score a borrower' is the default for quick scoring.")
page = st.sidebar.radio(
    "Menu",
    ["🧾 Score a borrower", "📊 Overview", "🔍 Feature Analysis", "🎯 Model Performance", "🧪 Validation", "💳 Scorecard", "🏦 IFRS 9 ECL"],
    index=0,
)
# normalize page label by removing leading emoji (makes comparisons below stable)
page = page.split(" ", 1)[1] if " " in page else page

# Everything else the dashboard shows (score distribution, gains/calibration/
# masterscale, PSI-by-vintage, ROC curves, base-case ECL) is precomputed at
# train time and loaded via `reports` / `load_all()` above — no recompute here.
#
# The one page with genuinely live, user-driven computation is the IFRS 9 ECL
# cockpit: the sliders are scenario parameters chosen at runtime, so staging
# and loss have to be recalculated when they change.
@st.cache_data(show_spinner="Recomputing ECL scenario…")
def compute_ecl_scenario(df, pd_sc, lgd_base, macro, sicr_abs):
    params = ECLParams(lgd_base=lgd_base, macro_pd_multiplier=macro, sicr_pd_absolute=sicr_abs)
    return ecl.compute_ecl(df, pd_sc, params)

# ---------------------------------------------------------------------------
# 1. Overview
# ---------------------------------------------------------------------------
if page == "Overview":
    base_ecl = reports["ecl_portfolio"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Loans", f"{len(df):,}")
    c2.metric("Default rate", f"{df[bundle.spec.target].mean():.1%}")
    c3.metric("Total EAD", f"${base_ecl['ead'].sum()/1e6:,.1f}M")
    c4.metric("Total ECL", f"${base_ecl['ecl'].sum()/1e6:,.1f}M")

    left, right = st.columns(2)
    with left:
        fig = px.histogram(x=score, nbins=40, title="Credit score distribution",
                           color_discrete_sequence=[ACCENT])
        fig.update_layout(xaxis_title="Scorecard points", yaxis_title="Loans")
        st.plotly_chart(fig, width="stretch")
    with right:
        by_purpose = df.assign(pd=pd_sc).groupby("purpose")["pd"].mean().sort_values()
        fig = px.bar(by_purpose, orientation="h", title="Average PD by loan purpose",
                     color_discrete_sequence=[ACCENT])
        fig.update_layout(xaxis_title="Mean PD", yaxis_title="", showlegend=False)
        st.plotly_chart(fig, width="stretch")

# ---------------------------------------------------------------------------
# 2. Feature analysis (WoE / IV)
# ---------------------------------------------------------------------------
if page == "Feature Analysis":
    iv = reports["iv_summary"]
    st.subheader("Information Value")
    fig = px.bar(iv.sort_values("iv"), x="iv", y="feature", orientation="h", color="strength",
                 title="Feature Information Value (predictive power)")
    st.plotly_chart(fig, width="stretch")

    feat = st.selectbox("Inspect a feature's WoE bins", list(encoder.bins_))
    binning = encoder.bins_[feat]
    table = binning.table.copy()
    fig = go.Figure()
    fig.add_bar(x=table["bin"].astype(str), y=table["event_rate"], name="Event (default) rate",
                marker_color="#C44E52", yaxis="y2", opacity=0.4)
    fig.add_scatter(x=table["bin"].astype(str), y=table["woe"], name="WoE",
                    mode="lines+markers", marker_color=ACCENT)
    fig.update_layout(
        title=f"{feat} — WoE and default rate by bin (IV={binning.iv:.3f})",
        yaxis=dict(title="WoE"), yaxis2=dict(title="Default rate", overlaying="y", side="right"),
        xaxis_title="Bin",
    )
    st.plotly_chart(fig, width="stretch")
    st.dataframe(table, width="stretch")

# ---------------------------------------------------------------------------
# 3. Model performance
# ---------------------------------------------------------------------------
if page == "Model Performance":
    st.subheader("Discrimination (held-out test set)")
    if metrics:
        rf_metrics = metrics.get("random_forest", {"auroc": np.nan, "gini": np.nan, "ks": np.nan})
        perf = pd.DataFrame({
            "Scorecard (LR)": metrics["scorecard"],
            "XGBoost": metrics["xgboost"],
            "Random Forest": rf_metrics,
        }).T
        perf.columns = [c.upper() for c in perf.columns]
        st.dataframe(perf.style.format("{:.3f}"), width="stretch")
        st.caption(f"PSI (train vs test): {metrics.get('psi_train_test', float('nan')):.4f} "
                   "— below 0.10 indicates a stable model.")
    else:
        st.info("Run `python scripts/train.py` to compute held-out metrics.")

    y = df[bundle.spec.target].to_numpy()

    col1, col2 = st.columns(2)
    with col1:
        fig = go.Figure()
        series = [("Scorecard", pd_sc, ACCENT), ("XGBoost", pd_gbm, "#C44E52"), ("Random Forest", pd_rf, "#7DCE82")]
        for label, scores, color in series:
            fpr, tpr, _ = roc_curve(y, scores)
            fig.add_scatter(x=fpr, y=tpr, mode="lines", name=label, line_color=color)
        fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash", color="grey"),
                        showlegend=False)
        fig.update_layout(title="ROC curve", xaxis_title="False positive rate",
                          yaxis_title="True positive rate")
        st.plotly_chart(fig, width="stretch")
    with col2:
        order = np.argsort(pd_sc)
        ys = y[order]
        cum_bad = np.cumsum(ys) / max(ys.sum(), 1)
        cum_good = np.cumsum(1 - ys) / max((1 - ys).sum(), 1)
        x = np.linspace(0, 1, len(ys))
        fig = go.Figure()
        fig.add_scatter(x=x, y=cum_good, mode="lines", name="Cumulative good", line_color=ACCENT)
        fig.add_scatter(x=x, y=cum_bad, mode="lines", name="Cumulative bad", line_color="#C44E52")
        fig.update_layout(title="KS separation (scorecard)", xaxis_title="Population sorted by score",
                          yaxis_title="Cumulative share")
        st.plotly_chart(fig, width="stretch")

# ---------------------------------------------------------------------------
# 4. Validation (calibration, gains/KS, out-of-time)
# ---------------------------------------------------------------------------
if page == "Validation":
    if metrics:
        ci = metrics["scorecard_ci"]
        hl = metrics["hosmer_lemeshow"]
        oot = metrics.get("out_of_time")
        c1, c2, c3 = st.columns(3)
        c1.metric("Gini (95% CI)", f"{ci['gini']['point']:.3f}",
                  f"[{ci['gini']['ci_low']:.3f}, {ci['gini']['ci_high']:.3f}]")
        c2.metric("Hosmer–Lemeshow p", f"{hl['p_value']:.3f}",
                  "well calibrated" if hl["p_value"] > 0.05 else "recalibrate")
        if oot:
            c3.metric(f"Out-of-time AUROC ({oot['test_year']})", f"{oot['auroc']:.3f}",
                      f"PSI {oot['score_psi']:.3f}")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Calibration")
        calib = reports["calibration"]
        fig = go.Figure()
        fig.add_scatter(x=calib["predicted"], y=calib["observed"], mode="markers+lines",
                        name="Model", line_color=ACCENT)
        lim = float(max(calib["predicted"].max(), calib["observed"].max()))
        fig.add_scatter(x=[0, lim], y=[0, lim], mode="lines", name="Perfect",
                        line=dict(dash="dash", color="grey"))
        fig.update_layout(xaxis_title="Predicted PD", yaxis_title="Observed default rate")
        st.plotly_chart(fig, width="stretch")
    with col2:
        st.subheader("Rating masterscale")
        ms = reports["masterscale"]
        fig = go.Figure()
        fig.add_bar(x=ms["grade"], y=ms["predicted_pd"], name="Predicted PD", marker_color=ACCENT)
        fig.add_bar(x=ms["grade"], y=ms["observed_dr"], name="Observed default", marker_color="#C44E52")
        fig.update_layout(barmode="group", xaxis_title="Rating grade", yaxis_title="Default rate")
        st.plotly_chart(fig, width="stretch")

    st.subheader("Gains / KS by decile")
    gains = reports["gains_table"]
    st.dataframe(
        gains.style.format({"bad_rate": "{:.1%}", "cum_bad_rate": "{:.1%}",
                            "cum_good_rate": "{:.1%}", "ks": "{:.3f}", "lift": "{:.2f}"}),
        width="stretch",
    )

    if not reports["psi_vintage"].empty:
        st.subheader("Score stability by vintage (PSI vs earliest)")
        st.dataframe(reports["psi_vintage"].style.format({"psi_vs_base": "{:.4f}"}), width="stretch")


# ---------------------------------------------------------------------------
# 5. Scorecard + borrower scoring
# ---------------------------------------------------------------------------
def humanize_field_name(field_name: str) -> str:
    labels = {
        "annual_income": "Annual income",
        "loan_amount": "Loan amount",
        "loan_term": "Loan term",
        "interest_rate": "Interest rate",
        "dti": "Debt-to-income ratio",
        "revolving_utilization": "Credit card utilization",
        "num_delinquencies_2yr": "Delinquencies in last 2 years",
        "num_open_accounts": "Number of open accounts",
        "employment_length": "Employment length",
        "credit_history_length": "Credit history length",
        "fico": "FICO score",
        "installment": "Monthly installment",
        "mort_acc": "Mortgage accounts",
        "pub_rec": "Public records",
        "pub_rec_bankruptcies": "Public bankruptcies",
        "total_acc": "Total accounts",
        "home_ownership": "Home ownership",
        "purpose": "Loan purpose",
        "sub_grade": "Credit sub-grade",
        "verification_status": "Verification status",
        "application_type": "Application type",
        "num_credit_lines": "Number of credit lines",
        "loan_purpose": "Loan purpose",
        "credit_score": "Credit score",
    }
    return labels.get(field_name, field_name.replace("_", " ").title())


def field_help_text(field_name: str) -> str:
    help_map = {
        "annual_income": "Borrower’s annual gross income before taxes.",
        "loan_amount": "Requested principal amount of the loan.",
        "loan_term": "Length of the loan in months.",
        "interest_rate": "Annual interest rate applied to the credit product.",
        "dti": "Debt-to-income ratio, reflecting monthly debt obligations versus income.",
        "revolving_utilization": "Percentage of revolving credit limit currently used.",
        "num_delinquencies_2yr": "Number of delinquencies in the last 24 months.",
        "num_open_accounts": "Total number of open credit accounts.",
        "employment_length": "Years employed in current job or role.",
        "credit_history_length": "Length of the borrower’s credit history.",
        "fico": "Borrower’s FICO credit score.",
        "installment": "Monthly installment payment on the loan.",
        "mort_acc": "Number of mortgage accounts on the borrower’s credit profile.",
        "pub_rec": "Number of public record derogatories.",
        "pub_rec_bankruptcies": "Number of bankruptcy records on file.",
        "total_acc": "Total number of credit accounts opened.",
        "home_ownership": "Borrower’s housing status, including renter or owner.",
        "purpose": "Primary purpose for which the loan is being used.",
        "sub_grade": "Internal risk grade assigned to the borrower's credit quality.",
        "verification_status": "Whether income or other details were verified.",
        "application_type": "Application channel or borrower type.",
        "num_credit_lines": "Total number of active credit lines across products.",
    }
    return help_map.get(field_name, "Enter the borrower’s value for this field. This helps estimate risk and repayment ability.")


# "Score a borrower" page: first and default
if page == "Score a borrower":
    st.subheader("Score a borrower")
    st.caption("Enter borrower details below. The form estimates the borrower's credit risk using the model's pricing and default assumptions.")
    with st.form("score_form"):
        cols = st.columns(3)
        inputs = {}
        for i, feat in enumerate(bundle.spec.numeric):
            col = cols[i % 3]
            median = float(df[feat].median())
            label = humanize_field_name(feat)
            inputs[feat] = col.number_input(label, value=round(median, 2), help=field_help_text(feat), format="%.2f")
            col.caption(field_help_text(feat))
        for i, feat in enumerate(bundle.spec.categorical):
            col = cols[i % 3]
            options = sorted(df[feat].astype(str).unique())
            label = humanize_field_name(feat)
            inputs[feat] = col.selectbox(label, options, help=field_help_text(feat))
            col.caption(field_help_text(feat))
        submitted = st.form_submit_button("Calculate PD, credit score & rating")

    if submitted:
        row = pd.DataFrame([inputs])
        row_woe = encoder.transform(row)
        pd_hat = float(scorecard.predict_pd(row_woe)[0])
        pts = float(scorecard.score(row_woe)[0])
        grade = scorecard.rating(np.array([pts]))[0]
        m1, m2, m3 = st.columns(3)
        m1.metric("Probability of default", f"{pd_hat:.2%}")
        m2.metric("Credit score", f"{pts:.0f}")
        m3.metric("Rating grade", grade)

# "Scorecard" page shows the points table (separated from the scoring form)
if page == "Scorecard":
    st.subheader("Points scorecard")
    st.caption("Base score 600 · 50:1 base odds · 20 points to double the odds.")
    scorecard_table = (
        pd.read_csv(config.SCORECARD_CSV) if config.SCORECARD_CSV.exists() else scorecard.scorecard_table()
    )
    st.dataframe(scorecard_table, width="stretch", height=320)

# ---------------------------------------------------------------------------
# 6. IFRS 9 ECL cockpit
# ---------------------------------------------------------------------------
if page == "IFRS 9 ECL":
    st.subheader("Scenario assumptions")
    c1, c2, c3 = st.columns(3)
    lgd_base = c1.slider("Baseline LGD", 0.20, 0.90, ECLParams.lgd_base, 0.05)
    macro = c2.slider("Macro PD multiplier (stress)", 0.5, 3.0, 1.0, 0.1)
    sicr_abs = c3.slider("SICR absolute PD threshold", 0.05, 0.50, ECLParams.sicr_pd_absolute, 0.05)

    # Scenario ECL is genuinely live — the sliders are runtime parameters.
    result = compute_ecl_scenario(df, pd_sc, lgd_base, macro, sicr_abs)
    summary = ecl.portfolio_summary(result)

    # Base (default-parameter) ECL was already computed at train time.
    base = ecl.portfolio_summary(reports["ecl_portfolio"])
    total_ecl = summary.loc["Total", "total_ecl"]
    delta = total_ecl - base.loc["Total", "total_ecl"]
    k1, k2, k3 = st.columns(3)
    k1.metric("Total ECL", f"${total_ecl/1e6:,.2f}M", f"{delta/1e6:+,.2f}M vs base")
    k2.metric("Portfolio coverage", f"{summary.loc['Total', 'coverage_ratio']:.2%}")
    k3.metric("Stage 2 + 3 loans", f"{int((result['stage'] >= 2).sum()):,}")

    disp = summary.copy()
    for col in ("total_ead", "total_ecl"):
        disp[col] = disp[col].map(lambda v: f"${v/1e6:,.2f}M")
    for col in ("avg_pd", "coverage_ratio"):
        disp[col] = disp[col].map(lambda v: f"{v:.2%}")
    disp["n_loans"] = disp["n_loans"].map(lambda v: f"{int(v):,}")
    st.dataframe(disp, width="stretch")

    c1, c2 = st.columns(2)
    stages = result["stage"].value_counts().reindex([1, 2, 3]).fillna(0)
    with c1:
        fig = px.bar(x=[f"Stage {i}" for i in stages.index], y=stages.values,
                     title="Loans by IFRS 9 stage", color_discrete_sequence=[ACCENT])
        fig.update_layout(xaxis_title="", yaxis_title="Loans")
        st.plotly_chart(fig, width="stretch")
    with c2:
        ecl_by_stage = result.groupby("stage")["ecl"].sum().reindex([1, 2, 3]).fillna(0)
        fig = px.bar(x=[f"Stage {i}" for i in ecl_by_stage.index], y=ecl_by_stage.values / 1e6,
                     title="ECL by stage ($M)", color_discrete_sequence=["#C44E52"])
        fig.update_layout(xaxis_title="", yaxis_title="ECL ($M)")
        st.plotly_chart(fig, width="stretch")
