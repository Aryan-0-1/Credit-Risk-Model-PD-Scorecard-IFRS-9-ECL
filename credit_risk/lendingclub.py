from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from .config import LENDING_CLUB_SPEC

# Raw columns we read (subset of the ~150 Lending Club ships). Only those that
# exist in the file are actually used.
_RAW_COLUMNS = [
    "loan_amnt", "funded_amnt", "term", "int_rate", "grade", "sub_grade", "emp_length",
    "home_ownership", "annual_inc", "purpose", "dti", "delinq_2yrs",
    "revol_util", "open_acc", "earliest_cr_line", "issue_d", "loan_status",
    # richer origination-time predictors (used only if present in the file)
    "fico_range_low", "fico_range_high", "installment", "mort_acc", "pub_rec",
    "pub_rec_bankruptcies", "total_acc", "verification_status", "application_type",
]

_GOOD_STATUS = {"Fully Paid", "Does not meet the credit policy. Status:Fully Paid"}
_BAD_STATUS = {
    "Charged Off", "Default", "Does not meet the credit policy. Status:Charged Off",
}

# Historical Lending Club default rate by grade -> origination PD proxy.
_GRADE_PD = {"A": 0.03, "B": 0.07, "C": 0.13, "D": 0.20, "E": 0.28, "F": 0.35, "G": 0.42}

# loan_status -> days-past-due proxy for IFRS 9 staging backstops.
_STATUS_DPD = {
    "Fully Paid": 0, "Current": 0, "Issued": 0,
    "In Grace Period": 15, "Late (16-30 days)": 25, "Late (31-120 days)": 60,
    "Default": 90, "Charged Off": 120,
    "Does not meet the credit policy. Status:Fully Paid": 0,
    "Does not meet the credit policy. Status:Charged Off": 120,
}


def _col(resolved: pd.DataFrame, name: str, default) -> pd.Series:
    """Return column *name* as a Series, or a constant-*default* Series if absent."""
    if name in resolved.columns:
        return resolved[name]
    return pd.Series(default, index=resolved.index)


def _parse_term(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.astype(str).str.extract(r"(\d+)", expand=False), errors="coerce")


def _parse_percent(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.astype(str).str.replace("%", "", regex=False).str.strip(), errors="coerce")


def _parse_emp_length(s: pd.Series) -> pd.Series:
    txt = s.astype(str)
    years = txt.str.extract(r"(\d+)", expand=False)
    years = pd.to_numeric(years, errors="coerce")
    years = years.mask(txt.str.contains("< 1", na=False), 0.0)
    return years


def _parse_credit_date(s: pd.Series) -> pd.Series:
    # Lending Club uses "Dec-2015" or occasionally "Dec-15".
    parsed = pd.to_datetime(s, format="%b-%Y", errors="coerce")
    missing = parsed.isna()
    if missing.any():
        parsed.loc[missing] = pd.to_datetime(s[missing], format="%b-%y", errors="coerce")
    return parsed


def _read_sample(path: Path, columns: list[str], sample_size: int, random_state: int) -> pd.DataFrame:
    """Uniformly sample resolved loans while streaming a large CSV.

    Keeping only the rows with the largest random keys is equivalent to a
    uniform sample, but avoids loading the multi-gigabyte source file at once.
    """
    rng = np.random.default_rng(random_state)
    kept = pd.DataFrame()
    statuses = _GOOD_STATUS | _BAD_STATUS
    for chunk in pd.read_csv(path, usecols=columns, low_memory=False, chunksize=100_000):
        chunk = chunk[chunk["loan_status"].astype(str).isin(statuses)].copy()
        if chunk.empty:
            continue
        chunk["_sample_key"] = rng.random(len(chunk))
        kept = pd.concat([kept, chunk], ignore_index=True).nlargest(sample_size, "_sample_key")
    return kept.drop(columns="_sample_key")


def load_lending_club(path, sample_size: int | None = None, random_state: int = 42) -> pd.DataFrame:
    """Load and clean a Lending Club file into the pipeline schema.

    Only resolved loans (Fully Paid / Charged Off / Default) are kept, since
    those are the loans with a known 12-month-style outcome for PD training.
    """
    if isinstance(path, (str, Path)):
        # Real file: peek the header so we only read the columns we need — the
        # Kaggle export is ~2 GB with 150 columns.
        path = Path(path)
        name = path.name
        usecols = [c for c in _RAW_COLUMNS if c in pd.read_csv(path, nrows=0).columns]
        raw = (
            _read_sample(path, usecols, sample_size, random_state)
            if sample_size else pd.read_csv(path, usecols=usecols, low_memory=False)
        )
    else:
        # File-like object (e.g. StringIO in tests): read once, then select.
        name = getattr(path, "name", "input")
        raw = pd.read_csv(path, low_memory=False)
        usecols = [c for c in _RAW_COLUMNS if c in raw.columns]
        raw = raw[usecols]

    missing_required = {"loan_amnt", "term", "int_rate", "annual_inc", "loan_status"} - set(usecols)
    if missing_required:
        raise ValueError(f"{name} is missing required columns: {sorted(missing_required)}")

    # Keep only resolved loans and build the binary target.
    status = raw["loan_status"].astype(str)
    resolved = raw[status.isin(_GOOD_STATUS | _BAD_STATUS)].copy()
    resolved["default"] = resolved["loan_status"].isin(_BAD_STATUS).astype(int)

    if sample_size and not isinstance(path, (str, Path)) and len(resolved) > sample_size:
        resolved = resolved.sample(sample_size, random_state=random_state)

    df = pd.DataFrame(index=resolved.index)
    df["annual_income"] = pd.to_numeric(resolved["annual_inc"], errors="coerce")
    df["loan_amount"] = pd.to_numeric(resolved["loan_amnt"], errors="coerce")
    df["loan_term"] = _parse_term(resolved["term"])
    df["interest_rate"] = _parse_percent(resolved["int_rate"])
    df["dti"] = pd.to_numeric(_col(resolved, "dti", np.nan), errors="coerce")
    df["revolving_utilization"] = _parse_percent(_col(resolved, "revol_util", np.nan)) / 100.0
    df["num_delinquencies_2yr"] = pd.to_numeric(_col(resolved, "delinq_2yrs", 0), errors="coerce").fillna(0)
    df["num_open_accounts"] = pd.to_numeric(_col(resolved, "open_acc", np.nan), errors="coerce")
    df["employment_length"] = _parse_emp_length(_col(resolved, "emp_length", np.nan))

    # Credit-history length in years = issue date - earliest credit line;
    # issue year doubles as the origination vintage for out-of-time validation.
    if "issue_d" in usecols:
        issued = _parse_credit_date(resolved["issue_d"])
        df["vintage"] = issued.dt.year
        if "earliest_cr_line" in usecols:
            hist = issued - _parse_credit_date(resolved["earliest_cr_line"])
            df["credit_history_length"] = (hist.dt.days / 365.25).clip(lower=0)
        else:
            df["credit_history_length"] = np.nan
    else:
        df["credit_history_length"] = np.nan
        df["vintage"] = np.nan

    # Richer origination-time predictors (only used if present in the file).
    if {"fico_range_low", "fico_range_high"} <= set(usecols):
        df["fico"] = (pd.to_numeric(resolved["fico_range_low"], errors="coerce")
                      + pd.to_numeric(resolved["fico_range_high"], errors="coerce")) / 2
    else:
        df["fico"] = np.nan
    df["installment"] = pd.to_numeric(_col(resolved, "installment", np.nan), errors="coerce")
    df["mort_acc"] = pd.to_numeric(_col(resolved, "mort_acc", np.nan), errors="coerce")
    df["pub_rec"] = pd.to_numeric(_col(resolved, "pub_rec", np.nan), errors="coerce")
    df["pub_rec_bankruptcies"] = pd.to_numeric(_col(resolved, "pub_rec_bankruptcies", np.nan), errors="coerce")
    df["total_acc"] = pd.to_numeric(_col(resolved, "total_acc", np.nan), errors="coerce")

    # Categoricals.
    home = _col(resolved, "home_ownership", "OTHER").astype(str).str.upper()
    df["home_ownership"] = home.where(home.isin(["RENT", "MORTGAGE", "OWN"]), "OTHER")
    df["purpose"] = _col(resolved, "purpose", "other").astype(str)
    df["sub_grade"] = _col(resolved, "sub_grade", "unknown").astype(str)
    df["verification_status"] = _col(resolved, "verification_status", "unknown").astype(str)
    df["application_type"] = _col(resolved, "application_type", "unknown").astype(str)

    # IFRS 9 fields.
    df["ead"] = pd.to_numeric(_col(resolved, "funded_amnt", resolved["loan_amnt"]), errors="coerce")
    grade = _col(resolved, "grade", "C").astype(str).str.upper()
    df["origination_pd"] = grade.map(_GRADE_PD).fillna(0.13)
    df["days_past_due"] = resolved["loan_status"].map(_STATUS_DPD).fillna(0).astype(int)

    df["default"] = resolved["default"].to_numpy()

    # Impute remaining numeric gaps with the median; drop rows still lacking a
    # critical field, then reset to a clean RangeIndex.
    critical = ["annual_income", "loan_amount", "loan_term", "interest_rate", "ead"]
    df = df.dropna(subset=critical)
    numeric_cols = LENDING_CLUB_SPEC.numeric
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    # Safety net: a feature entirely absent from this file version (all-NaN,
    # so no median) must not break quantile binning downstream.
    df[numeric_cols] = df[numeric_cols].fillna(0)
    return df.reset_index(drop=True)
