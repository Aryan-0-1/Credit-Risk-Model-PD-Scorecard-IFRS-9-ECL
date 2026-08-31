"""
IFRS 9 Expected Credit Loss (ECL) — three-stage impairment model.

IFRS 9 classifies every exposure into one of three stages and measures loss
accordingly:

    Stage 1  performing            -> 12-month ECL
    Stage 2  significant increase  -> lifetime ECL
             in credit risk (SICR)
    Stage 3  credit-impaired       -> lifetime ECL (PD = 1)

Core identity:  ECL = PD × LGD × EAD  (discounted at the effective interest rate).

Staging logic implemented here
------------------------------
* **Stage 3** if 90+ days past due (credit-impaired backstop).
* **Stage 2** if a significant increase in credit risk is detected — either a
  relative jump in PD since initial recognition (``PD_now ≥ multiple × PD_orig``),
  a high absolute PD, or the 30-day-past-due backstop.
* **Stage 1** otherwise.

Lifetime PD is built from the 12-month PD with a constant-hazard term structure:
``marginal_PDₜ = (1 − p)^{t−1} · p`` over the loan's remaining life, so lifetime
``PD = 1 − (1 − p)^T``. EAD is held flat across the horizon (a common
simplifying assumption for amortising retail books).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import DEFAULT_ECL, ECLParams

# LGD adjustments (added to the base LGD), reflecting collateral / product mix.
_HOME_LGD_ADJ = {"MORTGAGE": -0.15, "OWN": -0.10, "RENT": 0.05}
_PURPOSE_LGD_ADJ = {
    "small_business": 0.15,
    "other": 0.05,
    "home_improvement": -0.05,
    "debt_consolidation": 0.0,
    "credit_card": 0.0,
    "major_purchase": 0.0,
}


def estimate_lgd(df: pd.DataFrame, params: ECLParams = DEFAULT_ECL) -> np.ndarray:
    """Per-loan Loss Given Default from a base rate plus product adjustments."""
    lgd = np.full(len(df), params.lgd_base, dtype=float)
    lgd += df["home_ownership"].astype(str).map(_HOME_LGD_ADJ).fillna(0.0).to_numpy()
    lgd += df["purpose"].astype(str).map(_PURPOSE_LGD_ADJ).fillna(0.0).to_numpy()
    return np.clip(lgd, 0.05, 0.95)


def assign_stage(
    pd_12m: np.ndarray,
    origination_pd: np.ndarray,
    days_past_due: np.ndarray,
    params: ECLParams = DEFAULT_ECL,
) -> np.ndarray:
    """Return the IFRS 9 stage (1, 2 or 3) for each exposure."""
    pd_12m = np.asarray(pd_12m, dtype=float)
    origination_pd = np.asarray(origination_pd, dtype=float)
    days_past_due = np.asarray(days_past_due)

    # Relative SICR only bites once PD clears a floor, so estimation noise on
    # very-low-PD loans doesn't spuriously double a tiny number into Stage 2.
    relative = (pd_12m >= params.sicr_pd_multiple * origination_pd) & (
        pd_12m >= params.sicr_pd_floor
    )
    sicr = relative | (pd_12m >= params.sicr_pd_absolute) | (days_past_due >= params.dpd_stage2)

    # Low-credit-risk exemption keeps very safe, non-delinquent loans in Stage 1.
    low_risk = (pd_12m < params.low_credit_risk_pd) & (days_past_due < params.dpd_stage2)

    stage = np.ones(len(pd_12m), dtype=int)
    stage[sicr & ~low_risk] = 2
    stage[days_past_due >= params.dpd_stage3] = 3
    return stage


def _lifetime_ecl(p: np.ndarray, term_months: np.ndarray, lgd: np.ndarray,
                  ead: np.ndarray, rate: np.ndarray) -> np.ndarray:
    """Discounted lifetime ECL via a constant-hazard PD term structure."""
    years = np.ceil(term_months / 12).astype(int)
    max_years = int(years.max()) if len(years) else 1
    ecl = np.zeros(len(p), dtype=float)
    survival = np.ones(len(p), dtype=float)  # (1-p)^(t-1)
    for t in range(1, max_years + 1):
        marginal_pd = survival * p              # (1-p)^(t-1) * p
        active = years >= t
        ecl += active * marginal_pd * lgd * ead / (1 + rate) ** t
        survival *= (1 - p)
    return ecl


def compute_ecl(
    df: pd.DataFrame,
    pd_12m: np.ndarray,
    params: ECLParams = DEFAULT_ECL,
) -> pd.DataFrame:
    """Attach staging, PD/LGD/EAD and ECL to a copy of *df*.

    *pd_12m* is the model's current 12-month PD for each row. A macro stress
    multiplier from *params* is applied before staging and loss measurement.
    """
    out = df.copy()
    pd_12m = np.clip(np.asarray(pd_12m, dtype=float) * params.macro_pd_multiplier, 0, 1)

    lgd = estimate_lgd(out, params)
    ead = out["ead"].to_numpy(dtype=float)
    rate = out["interest_rate"].to_numpy(dtype=float) / 100.0
    term = out["loan_term"].to_numpy()

    stage = assign_stage(pd_12m, out["origination_pd"].to_numpy(), out["days_past_due"].to_numpy(), params)
    lifetime_pd = 1 - (1 - pd_12m) ** np.ceil(term / 12)

    # ECL by stage.
    ecl_12m = pd_12m * lgd * ead / (1 + rate)
    ecl_lifetime = _lifetime_ecl(pd_12m, term, lgd, ead, rate)
    ecl = np.where(stage == 1, ecl_12m, ecl_lifetime)
    ecl = np.where(stage == 3, lgd * ead, ecl)  # impaired: PD = 1

    out["stage"] = stage
    out["pd_12m"] = pd_12m
    out["lifetime_pd"] = lifetime_pd
    out["lgd"] = lgd
    out["ecl"] = ecl
    out["coverage_ratio"] = ecl / np.where(ead > 0, ead, np.nan)
    return out


def portfolio_summary(ecl_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate exposure and ECL by stage, plus a portfolio total."""
    grouped = ecl_df.groupby("stage").agg(
        n_loans=("ecl", "size"),
        total_ead=("ead", "sum"),
        total_ecl=("ecl", "sum"),
        avg_pd=("pd_12m", "mean"),
    )
    grouped["coverage_ratio"] = grouped["total_ecl"] / grouped["total_ead"]
    grouped = grouped.reindex([1, 2, 3]).fillna(0.0)
    grouped.index = [f"Stage {i}" for i in grouped.index]

    total = pd.DataFrame(
        {
            "n_loans": [grouped["n_loans"].sum()],
            "total_ead": [grouped["total_ead"].sum()],
            "total_ecl": [grouped["total_ecl"].sum()],
            "avg_pd": [ecl_df["pd_12m"].mean()],
            "coverage_ratio": [grouped["total_ecl"].sum() / grouped["total_ead"].sum()],
        },
        index=["Total"],
    )
    return pd.concat([grouped, total])
