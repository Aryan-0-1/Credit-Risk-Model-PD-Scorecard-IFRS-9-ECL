"""
Train the full pipeline, print a summary, and save all artifacts + figures.

    python scripts/train.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")  # headless: save figures without a display
import matplotlib.pyplot as plt

from credit_risk import config, metrics
from credit_risk.datasets import resolve_dataset
from credit_risk.pipeline import run_pipeline, save_artifacts, save_dashboard_figures, save_full_portfolio_scores


def _print_metrics(m: dict) -> None:
    print("\n=== Discrimination (held-out test set) ===")
    header = f"{'model':<12}{'AUROC':>8}{'Gini':>8}{'KS':>8}"
    print(header)
    print("-" * len(header))
    for name in ("scorecard", "xgboost", "random_forest"):
        s = m.get(name, {"auroc": float("nan"), "gini": float("nan"), "ks": float("nan")})
        print(f"{name:<16}{s['auroc']:>8.3f}{s['gini']:>8.3f}{s['ks']:>8.3f}")
    print(f"\nDefault rate: {m['default_rate']:.1%}   PSI(train,test): {m['psi_train_test']:.4f}")

    ci = m["scorecard_ci"]["gini"]
    hl = m["hosmer_lemeshow"]
    print(f"Scorecard Gini 95% CI: [{ci['ci_low']:.3f}, {ci['ci_high']:.3f}]")
    print(f"Hosmer-Lemeshow: chi2={hl['statistic']}, p={hl['p_value']} "
          f"({'well calibrated' if hl['p_value'] > 0.05 else 'recalibrate'})")
    if (oot := m.get("out_of_time")):
        print(f"Out-of-time (train {oot['train_years']} -> test {oot['test_year']}): "
              f"AUROC {oot['auroc']:.3f}, score PSI {oot['score_psi']:.4f}")


def _save_figures(result) -> None:
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Recreate test predictions for the plots.
    df = result.data
    X = df[result.spec.model_features]
    y = df[result.spec.target]
    woe = result.encoder.transform(X)
    sc_pd = result.scorecard.predict_pd(woe)
    gbm_pd = result.gbm.predict_pd(X)
    rf_pd = result.rf.predict_pd(X)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    metrics.plot_roc(axes[0], {"Scorecard": (y, sc_pd), "XGBoost": (y, gbm_pd), "Random Forest": (y, rf_pd)})
    metrics.plot_ks(axes[1], y, sc_pd, title="Scorecard KS")
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "model_performance.png", dpi=130)
    plt.close(fig)

    # Information Value bar chart.
    iv = result.iv_summary
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(iv["feature"][::-1], iv["iv"][::-1], color="#2E4057")
    ax.set_xlabel("Information Value")
    ax.set_title("Feature Information Value")
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "information_value.png", dpi=130)
    plt.close(fig)

    print(f"Saved figures -> {config.FIGURES_DIR}")


def main() -> None:
    bundle = resolve_dataset(sample_size=config.TRAIN_SAMPLE_SIZE)
    print(f"Data source: {bundle.name}  ({len(bundle.df):,} loans)")
    if bundle.source_file:
        print(f"            {bundle.source_file}")

    print("Training pipeline...")
    result = run_pipeline(bundle.df, spec=bundle.spec)
    _print_metrics(result.metrics)

    print("\n=== IFRS 9 ECL portfolio summary ===")
    summary = result.ecl_summary.copy()
    summary["total_ead"] = summary["total_ead"].map(lambda v: f"{v:,.0f}")
    summary["total_ecl"] = summary["total_ecl"].map(lambda v: f"{v:,.0f}")
    summary["avg_pd"] = summary["avg_pd"].map(lambda v: f"{v:.2%}")
    summary["coverage_ratio"] = summary["coverage_ratio"].map(lambda v: f"{v:.2%}")
    print(summary.to_string())

    save_artifacts(result)
    print("\nScoring the full historical portfolio for the dashboard "
          "(reads the whole source file — can take a few minutes)...")
    save_full_portfolio_scores(result.encoder, result.scorecard, result.gbm, result.rf, spec=bundle.spec)
    print("Rendering dashboard chart images...")
    save_dashboard_figures(result.encoder)
    _save_figures(result)
    print("\nArtifacts saved to models/ and reports/. Launch the dashboard with:")
    print("   streamlit run app/dashboard.py")


if __name__ == "__main__":
    main()
