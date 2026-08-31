"""Generate a Markdown model-validation report from a :class:`PipelineResult`."""
from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd


def _df_md(df: pd.DataFrame, floatfmt: str = "{:.4f}") -> str:
    """Render a DataFrame as a GitHub-flavoured Markdown table (no deps)."""
    cols = [str(c) for c in df.columns]
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    rows = []
    for _, r in df.iterrows():
        cells = []
        for c in df.columns:
            v = r[c]
            if isinstance(v, float):
                if pd.isna(v):
                    cells.append("")
                elif v.is_integer():
                    cells.append(str(int(v)))       # counts / band ids
                else:
                    cells.append(floatfmt.format(v))
            else:
                cells.append(str(v))
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join([header, sep] + rows)


def write_validation_report(result, path: Path) -> Path:
    m = result.metrics
    sc, xgb, rf = m["scorecard"], m["xgboost"], m.get("random_forest", {"auroc": float("nan"), "gini": float("nan"), "ks": float("nan")})
    ci = m["scorecard_ci"]
    hl = m["hosmer_lemeshow"]
    oot = m.get("out_of_time")

    lines: list[str] = []
    w = lines.append

    w("# Model Validation Report — PD Scorecard\n")
    w(f"_Generated {date.today().isoformat()} · {m['n_train']:,} train / {m['n_test']:,} test loans · "
      f"observed default rate {m['default_rate']:.2%}_\n")

    w("## 1. Discrimination\n")
    w("| Model | AUROC | Gini | KS |")
    w("|---|---|---|---|")
    w(f"| Scorecard (champion) | {sc['auroc']:.3f} | {sc['gini']:.3f} | {sc['ks']:.3f} |")
    w(f"| XGBoost (challenger) | {xgb['auroc']:.3f} | {xgb['gini']:.3f} | {xgb['ks']:.3f} |")
    w(f"| Random Forest (challenger) | {rf['auroc']:.3f} | {rf['gini']:.3f} | {rf['ks']:.3f} |\n")
    w("**Scorecard bootstrap 95% CIs:** "
      + " · ".join(f"{k.upper()} {v['point']:.3f} [{v['ci_low']:.3f}, {v['ci_high']:.3f}]"
                   for k, v in ci.items()) + "\n")

    w("## 2. Rank ordering — gains / KS by decile\n")
    w(_df_md(result.gains_table) + "\n")

    w("## 3. Calibration\n")
    w(f"**Hosmer–Lemeshow:** chi-square = {hl['statistic']} (dof {hl['dof']}), p = {hl['p_value']} "
      + ("— PDs are well calibrated (fail to reject H0)."
         if hl["p_value"] > 0.05 else "— calibration imperfect (reject H0); consider recalibration.") + "\n")
    w("Predicted vs observed default rate by PD decile:\n")
    w(_df_md(result.calibration) + "\n")

    w("## 4. Rating masterscale\n")
    w(_df_md(result.masterscale) + "\n")

    w("## 5. Out-of-time validation\n")
    if oot:
        w(f"Trained on vintages {oot['train_years']} ({oot['n_train']:,} loans), "
          f"tested on {oot['test_year']} ({oot['n_test']:,} loans):\n")
        w(f"- AUROC **{oot['auroc']:.3f}** · Gini {oot['gini']:.3f} · KS {oot['ks']:.3f}")
        w(f"- Score-distribution PSI (in-time vs out-of-time): **{oot['score_psi']:.4f}** "
          + ("(stable, < 0.10)" if oot["score_psi"] < 0.10 else "(shift detected, >= 0.10)") + "\n")
    else:
        w("_Not available: the dataset lacks usable origination vintages._\n")

    if not result.psi_vintage.empty:
        w("Score-distribution PSI by vintage (vs earliest):\n")
        w(_df_md(result.psi_vintage) + "\n")

    w("## 6. IFRS 9 ECL summary\n")
    ecl = result.ecl_summary.copy()
    for col in ("total_ead", "total_ecl"):
        ecl[col] = ecl[col].map(lambda v: f"{v:,.0f}")
    for col in ("avg_pd", "coverage_ratio"):
        ecl[col] = ecl[col].map(lambda v: f"{v:.2%}")
    ecl["n_loans"] = ecl["n_loans"].map(lambda v: f"{int(v):,}")
    w(_df_md(ecl.reset_index(names="stage")) + "\n")

    path = Path(path)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
