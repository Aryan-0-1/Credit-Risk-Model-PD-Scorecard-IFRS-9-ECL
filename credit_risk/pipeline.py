"""
End-to-end orchestration: data → WoE/IV → scorecard + XGBoost → metrics → ECL.

``run_pipeline`` returns a :class:`PipelineResult` bundling every artifact so
both the training script and the Streamlit dashboard can reuse one code path.
Artifacts are persisted with :func:`save_artifacts` / :func:`load_artifacts`.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from . import config, ecl, explain, metrics, validation
from .gbm import GBMModel
from .random_forest import RandomForestModel
from .scorecard import Scorecard
from .woe import WoEEncoder


@dataclass
class PipelineResult:
    encoder: WoEEncoder
    scorecard: Scorecard
    gbm: GBMModel
    rf: RandomForestModel
    metrics: dict
    iv_summary: pd.DataFrame
    scorecard_table: pd.DataFrame
    ecl_portfolio: pd.DataFrame            # per-loan ECL detail
    ecl_summary: pd.DataFrame              # aggregated by stage
    spec: config.FeatureSpec
    # --- validation artifacts ---
    gains_table: pd.DataFrame
    calibration: pd.DataFrame
    masterscale: pd.DataFrame
    psi_vintage: pd.DataFrame
    shap_importance: pd.DataFrame
    data: pd.DataFrame = field(repr=False)


def run_pipeline(
    df: pd.DataFrame,
    spec: config.FeatureSpec = config.LENDING_CLUB_SPEC,
    test_size: float = 0.3,
    random_state: int = config.RANDOM_STATE,
    train_on_full_data: bool = False,
) -> PipelineResult:
    X = df[spec.model_features]
    y = df[spec.target]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    # --- Feature engineering: WoE / IV (fit on train only) -------------
    encoder = WoEEncoder(spec.numeric, spec.categorical)
    Xtr_woe = encoder.fit_transform(X_train, y_train)
    Xte_woe = encoder.transform(X_test)

    # --- Champion: logistic scorecard ----------------------------------
    scorecard = Scorecard(encoder).fit(Xtr_woe, y_train)
    sc_pd_test = scorecard.predict_pd(Xte_woe)

    # --- Challenger: XGBoost -------------------------------------------
    gbm = GBMModel(spec).fit(X_train, y_train)
    gbm_pd_test = gbm.predict_pd(X_test)

    # --- Challenger: Random Forest --------------------------------------
    rf = RandomForestModel(spec).fit(X_train, y_train)
    rf_pd_test = rf.predict_pd(X_test)

    # --- Discrimination metrics on the held-out test set ---------------
    sc_score_test = scorecard.score(Xte_woe)
    sc_rating_test = scorecard.rating(sc_score_test)
    model_metrics = {
        "scorecard": metrics.evaluate(y_test, sc_pd_test).as_dict(),
        "xgboost": metrics.evaluate(y_test, gbm_pd_test).as_dict(),
        "random_forest": metrics.evaluate(y_test, rf_pd_test).as_dict(),
        "psi_train_test": round(metrics.psi(scorecard.predict_pd(Xtr_woe), sc_pd_test), 4),
        "default_rate": round(float(y.mean()), 4),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        # --- validation diagnostics ---
        "scorecard_ci": {m: validation.bootstrap_metric_ci(y_test, sc_pd_test, m)
                         for m in ("auroc", "gini", "ks")},
        "hosmer_lemeshow": validation.hosmer_lemeshow(y_test, sc_pd_test),
        "out_of_time": validation.out_of_time_validation(df, spec),
    }

    # --- Validation artifacts ------------------------------------------
    gains = validation.gains_table(y_test, sc_pd_test)
    calibration = validation.calibration_table(y_test, sc_pd_test)
    masterscale = validation.rating_masterscale(sc_rating_test, sc_pd_test, y_test)
    shap_importance = explain.shap_importance(gbm, X_test)

    # --- IFRS 9 ECL on the full book (scored by the champion) ----------
    full_woe = encoder.transform(X)
    pd_full = scorecard.predict_pd(full_woe)
    full_score = scorecard.score(full_woe)
    ecl_portfolio = ecl.compute_ecl(df, pd_full)
    ecl_summary = ecl.portfolio_summary(ecl_portfolio)

    psi_vintage = (
        validation.psi_by_vintage(full_score, df["vintage"])
        if "vintage" in df.columns else pd.DataFrame()
    )

    return PipelineResult(
        encoder=encoder,
        scorecard=scorecard,
        gbm=gbm,
        rf=rf,
        metrics=model_metrics,
        iv_summary=encoder.iv_summary(),
        scorecard_table=scorecard.scorecard_table(),
        ecl_portfolio=ecl_portfolio,
        ecl_summary=ecl_summary,
        spec=spec,
        gains_table=gains,
        calibration=calibration,
        masterscale=masterscale,
        psi_vintage=psi_vintage,
        shap_importance=shap_importance,
        data=df,
    )


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------
def save_artifacts(result: PipelineResult, models_dir: Path = config.MODELS_DIR,
                   reports_dir: Path = config.REPORTS_DIR) -> None:
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(result.encoder, models_dir / "woe_encoder.joblib")
    joblib.dump(result.scorecard, models_dir / "scorecard.joblib")
    joblib.dump(result.rf, models_dir / "random_forest.joblib")
    result.gbm.model.save_model(models_dir / "xgboost.json")
    (models_dir / "features.json").write_text(
        json.dumps(result.spec.to_dict(), indent=2), encoding="utf-8"
    )

    config.METRICS_JSON.write_text(json.dumps(result.metrics, indent=2), encoding="utf-8")
    # Save core artifacts: encoder, scorecard, model, features and metrics
    result.scorecard_table.to_csv(config.SCORECARD_CSV, index=False)
    result.ecl_portfolio.to_csv(config.ECL_CSV, index=False)
    # Keep metrics for the dashboard and a minimal validation report
    config.METRICS_JSON.write_text(json.dumps(result.metrics, indent=2), encoding="utf-8")

    # --- Dashboard artifacts: everything the app needs that does NOT ------
    # --- depend on live user input, precomputed once here. ----------------
    result.gains_table.to_csv(config.GAINS_TABLE_CSV, index=False)
    result.calibration.to_csv(config.CALIBRATION_CSV, index=False)
    result.masterscale.to_csv(config.MASTERSCALE_CSV, index=False)
    result.iv_summary.to_csv(config.IV_SUMMARY_CSV, index=False)
    if not result.psi_vintage.empty:
        result.psi_vintage.to_csv(config.PSI_VINTAGE_CSV, index=False)

    try:
        from .report import write_validation_report
        write_validation_report(result, reports_dir / "validation_report.md")
    except Exception:
        # Writing the report is optional; continue silently on error
        pass


def save_full_portfolio_scores(
    encoder: WoEEncoder,
    scorecard: Scorecard,
    gbm: GBMModel,
    rf: RandomForestModel,
    spec: config.FeatureSpec = config.LENDING_CLUB_SPEC,
    reports_dir: Path = config.REPORTS_DIR,
) -> None:
    """Score every resolved loan in the source file — not just the training
    sample — with already-fitted models, and save the result.

    Training only ever needs a manageable sample (``config.TRAIN_SAMPLE_SIZE``),
    but the dashboard is meant to show the full historical book. This function
    bridges that gap once, offline, so the dashboard never has to score
    millions of rows itself. No refitting happens here — just inference.
    """
    # Local import avoids a circular import (datasets.py doesn't need pipeline.py).
    from .datasets import resolve_dataset

    full = resolve_dataset(sample_size=None).df
    X_full = full[spec.model_features]

    woe_full = encoder.transform(X_full)
    pd_full = scorecard.predict_pd(woe_full)
    score_full = scorecard.score(woe_full)

    scored = pd.DataFrame(
        {
            "pd_scorecard": pd_full,
            "score": score_full,
            "rating": scorecard.rating(score_full),
            "pd_gbm": gbm.predict_pd(X_full),
            "pd_rf": rf.predict_pd(X_full),
        },
        index=full.index,
    )
    reports_dir.mkdir(parents=True, exist_ok=True)
    scored.to_parquet(config.SCORED_PORTFOLIO_PARQUET, index=False)

    # ecl.compute_ecl returns df.copy() plus the ECL fields, so this single
    # file also carries every original engineered feature — it's what the
    # dashboard loads as `df`, with no raw CSV read needed at app start.
    ecl_full = ecl.compute_ecl(full, pd_full)
    ecl_full.to_parquet(config.FULL_PORTFOLIO_PARQUET, index=False)

    # Downsampled ROC/KS curves for the dashboard's Model Performance page.
    # Computing roc_curve() and sorting 1.3M rows live, on every page view,
    # was the actual bottleneck there — this removes it entirely.
    y_full = full[spec.target].to_numpy()
    roc_rows = []
    for label, s in (("Scorecard", pd_full), ("XGBoost", scored["pd_gbm"]), ("Random Forest", scored["pd_rf"])):
        fpr, tpr = metrics.roc_points(y_full, s)
        roc_rows.append(pd.DataFrame({"model": label, "fpr": fpr, "tpr": tpr}))
    pd.concat(roc_rows, ignore_index=True).to_csv(config.ROC_CURVES_CSV, index=False)

    ks_x, ks_good, ks_bad = metrics.ks_points(y_full, pd_full)
    pd.DataFrame({"population_share": ks_x, "cum_good": ks_good, "cum_bad": ks_bad}).to_csv(
        config.KS_CURVE_CSV, index=False
    )


def save_dashboard_figures(
    encoder: WoEEncoder,
    reports_dir: Path = config.REPORTS_DIR,
    figures_dir: Path = config.DASHBOARD_FIGURES_DIR,
) -> None:
    """Render every dashboard chart that doesn't depend on live user input to
    a static PNG, once. The Streamlit app then just displays the image file —
    no chart object is built and shipped to the browser on each page view.

    Only the IFRS 9 ECL cockpit is exempt: its charts respond to slider
    input, so they're genuinely live and stay as interactive Plotly charts
    in the dashboard itself.

    Uses matplotlib (already a project dependency, headless-safe via the Agg
    backend) rather than Plotly's static export — as of Kaleido v1, that
    route requires a separate Chrome install, an extra moving part this
    doesn't need.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ACCENT, RED, GREEN = "#2E4057", "#C44E52", "#7DCE82"
    figures_dir.mkdir(parents=True, exist_ok=True)
    reports = load_reports(reports_dir)
    full, scored = reports["full_portfolio"], reports["scored_portfolio"]

    def _save(fig, name: str):
        fig.tight_layout()
        fig.savefig(figures_dir / name, dpi=150)
        plt.close(fig)

    # --- Overview -----------------------------------------------------
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(scored["score"], bins=40, color=ACCENT)
    ax.set_xlabel("Scorecard points")
    ax.set_ylabel("Loans")
    ax.set_title("Credit score distribution")
    _save(fig, "score_distribution.png")

    by_purpose = full.assign(pd=scored["pd_scorecard"].to_numpy()).groupby("purpose")["pd"].mean().sort_values()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh(by_purpose.index, by_purpose.to_numpy(), color=ACCENT)
    ax.set_xlabel("Mean PD")
    ax.set_title("Average PD by loan purpose")
    _save(fig, "pd_by_purpose.png")

    # --- Feature Analysis -----------------------------------------------
    iv = reports["iv_summary"].sort_values("iv")
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(iv["feature"], iv["iv"], color=ACCENT)
    ax.set_xlabel("Information Value")
    ax.set_title("Feature Information Value (predictive power)")
    _save(fig, "information_value.png")

    for feat, binning in encoder.bins_.items():
        table = binning.table.copy()
        bins = table["bin"].astype(str)
        fig, ax1 = plt.subplots(figsize=(8, 4.5))
        ax2 = ax1.twinx()
        ax2.bar(bins, table["event_rate"], color=RED, alpha=0.4, label="Event (default) rate")
        ax1.plot(bins, table["woe"], color=ACCENT, marker="o", label="WoE", zorder=3)
        ax1.set_ylabel("WoE")
        ax2.set_ylabel("Default rate")
        ax1.set_xlabel("Bin")
        ax1.set_title(f"{feat} — WoE and default rate by bin (IV={binning.iv:.3f})")
        ax1.tick_params(axis="x", rotation=45)
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
        _save(fig, f"woe_{feat}.png")

    # --- Model Performance ----------------------------------------------
    roc = reports["roc_curves"]
    colors = {"Scorecard": ACCENT, "XGBoost": RED, "Random Forest": GREEN}
    fig, ax = plt.subplots(figsize=(5.5, 5))
    for label, group in roc.groupby("model", sort=False):
        ax.plot(group["fpr"], group["tpr"], color=colors.get(label, ACCENT), label=label)
    ax.plot([0, 1], [0, 1], "--", color="grey", linewidth=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curve")
    ax.legend(loc="lower right")
    _save(fig, "roc_curve.png")

    ks = reports["ks_curve"]
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot(ks["population_share"], ks["cum_good"], color=ACCENT, label="Cumulative good")
    ax.plot(ks["population_share"], ks["cum_bad"], color=RED, label="Cumulative bad")
    ax.set_xlabel("Population sorted by score")
    ax.set_ylabel("Cumulative share")
    ax.set_title("KS separation (scorecard)")
    ax.legend(loc="upper left")
    _save(fig, "ks_curve.png")

    # --- Validation -----------------------------------------------------
    calib = reports["calibration"]
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot(calib["predicted"], calib["observed"], color=ACCENT, marker="o", label="Model")
    lim = float(max(calib["predicted"].max(), calib["observed"].max()))
    ax.plot([0, lim], [0, lim], "--", color="grey", label="Perfect")
    ax.set_xlabel("Predicted PD")
    ax.set_ylabel("Observed default rate")
    ax.set_title("Calibration")
    ax.legend(loc="upper left")
    _save(fig, "calibration.png")

    ms = reports["masterscale"]
    x = np.arange(len(ms))
    width = 0.35
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.bar(x - width / 2, ms["predicted_pd"], width, color=ACCENT, label="Predicted PD")
    ax.bar(x + width / 2, ms["observed_dr"], width, color=RED, label="Observed default")
    ax.set_xticks(x)
    ax.set_xticklabels(ms["grade"])
    ax.set_xlabel("Rating grade")
    ax.set_ylabel("Default rate")
    ax.set_title("Rating masterscale")
    ax.legend()
    _save(fig, "masterscale.png")


def train_and_save(
    df: pd.DataFrame,
    spec: config.FeatureSpec = config.LENDING_CLUB_SPEC,
    **kwargs,
) -> PipelineResult:
    """Run the pipeline on *df* (typically a sample), save every model and
    validation artifact, then score the full historical book for the
    dashboard. Single entry point for anything that needs a from-scratch
    train (the CLI script, and the dashboard's first-run fallback).
    """
    result = run_pipeline(df, spec=spec, **kwargs)
    save_artifacts(result)
    save_full_portfolio_scores(result.encoder, result.scorecard, result.gbm, result.rf, spec=spec)
    save_dashboard_figures(result.encoder)
    return result


def load_artifacts(models_dir: Path = config.MODELS_DIR):
    """Load the trained encoder, scorecard, XGBoost and Random Forest models from disk."""
    from xgboost import XGBClassifier

    spec = config.LENDING_CLUB_SPEC
    features_path = models_dir / "features.json"
    if features_path.exists():
        spec = config.FeatureSpec.from_dict(json.loads(features_path.read_text(encoding="utf-8")))

    encoder = joblib.load(models_dir / "woe_encoder.joblib")
    scorecard = joblib.load(models_dir / "scorecard.joblib")

    gbm = GBMModel(spec)
    gbm.model = XGBClassifier()
    gbm.model.load_model(models_dir / "xgboost.json")

    rf = RandomForestModel(spec)
    if (models_dir / "random_forest.joblib").exists():
        rf = joblib.load(models_dir / "random_forest.joblib")

    return encoder, scorecard, gbm, rf


def load_reports(reports_dir: Path = config.REPORTS_DIR) -> dict:
    """Load the precomputed report/validation tables saved by :func:`save_artifacts`.

    Returns a dict of DataFrames (empty DataFrame for anything not yet saved,
    e.g. on a fresh checkout before ``train.py`` has been run). Callers should
    treat an empty ``scored_portfolio`` as "artifacts not built yet".
    """
    def _read_csv(path: Path, **kwargs) -> pd.DataFrame:
        return pd.read_csv(path, **kwargs) if path.exists() else pd.DataFrame()

    def _read_parquet(path: Path) -> pd.DataFrame:
        return pd.read_parquet(path) if path.exists() else pd.DataFrame()

    full_portfolio = _read_parquet(config.FULL_PORTFOLIO_PARQUET)
    return {
        "scored_portfolio": _read_parquet(config.SCORED_PORTFOLIO_PARQUET),
        "full_portfolio": full_portfolio,
        "ecl_portfolio": full_portfolio,  # same table: features + stage/pd_12m/ecl columns
        "gains_table": _read_csv(config.GAINS_TABLE_CSV),
        "calibration": _read_csv(config.CALIBRATION_CSV),
        "masterscale": _read_csv(config.MASTERSCALE_CSV),
        "psi_vintage": _read_csv(config.PSI_VINTAGE_CSV),
        "iv_summary": _read_csv(config.IV_SUMMARY_CSV),
        "roc_curves": _read_csv(config.ROC_CURVES_CSV),
        "ks_curve": _read_csv(config.KS_CURVE_CSV),
    }
