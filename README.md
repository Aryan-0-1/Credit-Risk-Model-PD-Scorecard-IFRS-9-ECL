# Credit Risk Model — PD Scorecard & IFRS 9 ECL

A bank-grade probability-of-default (PD) modeling pipeline built on the Lending Club loan dataset, pairing an interpretable logistic scorecard (champion) against XGBoost and Random Forest (challengers), a full model-validation suite, and an IFRS 9 three-stage Expected Credit Loss engine — all wired into an interactive Streamlit dashboard.

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Get the Dataset](#get-the-dataset)
  - [Train](#train)
  - [Launch the Dashboard](#launch-the-dashboard)
- [Results](#results)
- [Dashboard Tour](#dashboard-tour)
- [How It Works](#how-it-works)
- [Performance Architecture](#performance-architecture)
- [Tech Stack](#tech-stack)
- [Data Source](#data-source)
- [Author](#Author)

---

## Overview

This project builds an end-to-end credit risk system on top of Lending Club's historical loan data (2007–2018):

1. **Predict probability of default (PD)** using a champion/challenger setup.
2. **Turn PD into a business-usable credit scorecard** — points-based, per-attribute, readable by a credit committee.
3. **Calculate IFRS 9 regulatory loss provisions** (Expected Credit Loss) across the full loan book.
4. **Validate everything** the way a model-risk function would — discrimination, calibration, stability, out-of-time performance.
5. **Serve it all** through an interactive dashboard for exploration, borrower-level scoring, and ECL scenario stress-testing.

## Key Features

- **Champion/challenger modeling** — a WoE-binned logistic regression scorecard (interpretable, monotonic, points-scaled) benchmarked against XGBoost and Random Forest.
- **Weight-of-Evidence (WoE) & Information Value (IV)** feature engineering with automatic monotonic binning.
- **Points-based scorecard** using the standard points-to-double-odds (PDO) convention — base score 600, 50:1 base odds, 20 PDO.
- **IFRS 9 three-stage ECL engine** — Stage 1 (12-month ECL), Stage 2 (lifetime ECL on significant credit deterioration), Stage 3 (credit-impaired, PD = 1), with per-loan LGD estimated from collateral/product mix.
- **Full model-validation toolkit** — gains/KS decile tables, a rating masterscale, Hosmer–Lemeshow calibration testing, bootstrapped confidence intervals, population stability index (PSI), and genuine out-of-time validation (train on early vintages, test on the latest).
- **Interactive Streamlit dashboard** — 7 tabs covering portfolio overview, feature analysis, model performance, validation diagnostics, the points scorecard, live borrower scoring, and an ECL cockpit with scenario sliders (LGD, macro stress, SICR threshold).
- **Fast by design** — every chart and metric that doesn't depend on live input is precomputed once at training time (see [Performance Architecture](#performance-architecture)); the dashboard only ever *loads*, it doesn't recompute.

## Project Structure

```
credit-risk-model/
├── app/
│   └── dashboard.py          # Streamlit app
├── credit_risk/               # Core package
│   ├── config.py              # Paths, feature spec, model assumptions
│   ├── datasets.py            # Dataset loading entry point
│   ├── lendingclub.py         # Raw CSV → cleaned feature frame
│   ├── woe.py                 # Weight-of-Evidence binning & IV
│   ├── scorecard.py           # Logistic scorecard + points scaling
│   ├── gbm.py                 # XGBoost challenger
│   ├── random_forest.py       # Random Forest challenger
│   ├── explain.py             # SHAP-based feature importance
│   ├── ecl.py                 # IFRS 9 staging & ECL calculation
│   ├── metrics.py             # AUROC / Gini / KS / PSI + chart helpers
│   ├── validation.py          # Gains, calibration, masterscale, OOT validation
│   ├── report.py              # Markdown validation report generator
│   └── pipeline.py            # Orchestration: train, save, load artifacts
├── scripts/
│   └── train.py               # CLI entry point — run this first
├── data/
│   └── accepted_2007_to_2018Q4.csv   # ⚠️ not committed — see below
├── models/                    # Generated: trained model artifacts
├── reports/                   # Generated: metrics, tables, chart images
├── requirements.txt
└── README.md
```

`data/`, `models/`, and `reports/` are generated locally and excluded via `.gitignore` — see [Get the Dataset](#get-the-dataset).

## Getting Started

### Prerequisites

- Python 3.10+
- ~4 GB free disk space (raw dataset + generated artifacts)
- The [Lending Club Loan Data](https://www.kaggle.com/datasets/wordsforthewise/lending-club) dataset from Kaggle (free account required)

### Installation

```bash
git clone https://github.com/<your-username>/credit-risk-model.git
cd credit-risk-model
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Get the Dataset

Download **`accepted_2007_to_2018Q4.csv`** from Kaggle and place it at:

```
data/accepted_2007_to_2018Q4.csv
```

The loader only reads the columns it needs and streams the file in chunks, so it works fine on modest hardware despite the file being ~2 GB.

### Train

```bash
python scripts/train.py
```

This fits the WoE encoder, the logistic scorecard, XGBoost, and Random Forest on a 150,000-loan sample; runs the full validation suite; computes IFRS 9 ECL; scores the *entire* historical book (1.3M+ loans) with the fitted models; and renders every dashboard chart to a static image. Expect this to take several minutes — most of it is XGBoost training and scoring the full portfolio.

Artifacts land in `models/` (trained models) and `reports/` (metrics, validation tables, chart images, ECL results).

### Launch the Dashboard

```bash
streamlit run app/dashboard.py
```

Opens at `http://localhost:8501`. Since everything is precomputed, this should load in a couple of seconds.

## Results

Held-out test set (45,000 loans), from a training run on the full dataset:

| Model | AUROC | Gini | KS |
|---|---|---|---|
| Scorecard (champion) | 0.707 | 0.413 | 0.300 |
| XGBoost (challenger) | 0.718 | 0.435 | 0.318 |
| Random Forest (challenger) | 0.714 | 0.427 | 0.314 |

Both challengers outperform the linear scorecard on raw discrimination, as expected — the scorecard is kept as the production model for its interpretability and monotonic, auditable structure, while the challengers quantify the performance left on the table.

**IFRS 9 ECL summary** (full portfolio, ~150K loans, base-case assumptions):

| Stage | Loans | EAD | ECL | Coverage |
|---|---|---|---|---|
| Stage 1 (performing) | 51,815 | $626.7M | $20.3M | 3.25% |
| Stage 2 (SICR) | 68,226 | $1,062.2M | $190.2M | 17.90% |
| Stage 3 (impaired) | 29,958 | $467.3M | $182.7M | 39.09% |
| **Total** | **149,999** | **$2,156.2M** | **$393.2M** | **18.23%** |

Exact numbers will vary run to run since training samples 150,000 loans with a fixed seed but the source file may differ by download date.

## Dashboard Tour

| Tab | What it shows |
|---|---|
| 🧾 Score a borrower | Live PD, credit score, and rating grade for a manually entered borrower |
| 📊 Overview | Portfolio-level KPIs, score distribution, average PD by loan purpose |
| 🔍 Feature Analysis | Information Value ranking and per-feature WoE/default-rate charts |
| 🎯 Model Performance | Champion vs. challenger discrimination — ROC curves, KS chart |
| 🧪 Validation | Calibration, rating masterscale, gains/KS by decile, PSI by vintage |
| 💳 Scorecard | The full points scorecard — every feature, bin, and point value |
| 🏦 IFRS 9 ECL | Live ECL cockpit — stress-test LGD, macro PD multiplier, and SICR threshold |

## How It Works

**Scorecard mechanics:** each numeric and categorical feature is binned via Weight-of-Evidence (`WoE = ln(P(good|bin) / P(bad|bin))`), monotonic in risk by construction. A logistic regression is fit on the WoE-transformed features, then rescaled into points using the standard PDO formula so a business user can read a borrower's score as a sum of per-attribute points.

**IFRS 9 staging:** every loan is assigned Stage 1, 2, or 3 based on relative/absolute PD deterioration since origination and days-past-due backstops (30-day for Stage 2, 90-day for Stage 3). Stage 1 gets 12-month ECL; Stages 2 and 3 get lifetime ECL via a constant-hazard PD term structure, discounted at the loan's effective rate.

**Validation:** the champion scorecard is evaluated on a held-out test set (gains/KS, calibration, Hosmer–Lemeshow, bootstrapped CIs) and, separately, via true out-of-time validation — refit on early loan vintages, tested on the most recent one.

## Performance Architecture

A meaningful part of this project is *how* the dashboard stays fast despite scoring 1.3M+ loans:

- **Nothing that doesn't depend on live input is computed in the dashboard.** Model scoring, validation tables, and every chart are computed once by `train.py` and saved to `reports/`. The dashboard only loads files.
- **Large tables are stored as Parquet**, not CSV — faster to read, smaller on disk, and dtype-safe.
- **Charts are pre-rendered to PNG** at training time with matplotlib, so the dashboard just calls `st.image()` — no chart object is built or shipped to the browser on every page view.
- **Curves are downsampled** before saving (e.g., ROC/KS curves to ~500 points) — visually identical to plotting the full 1.3M-point curve, at a fraction of the size.
- **Only the IFRS 9 ECL cockpit stays live**, since its charts genuinely respond to the scenario sliders.

The upshot: opening the dashboard costs roughly what it takes to load a handful of small files, not to reprocess a multi-gigabyte dataset.

## Tech Stack

- **Modeling:** scikit-learn (logistic regression, Random Forest), XGBoost, SHAP-style feature importance
- **Data:** pandas, NumPy, PyArrow (Parquet)
- **Validation & stats:** SciPy
- **Visualization:** Plotly (interactive dashboard charts), matplotlib (precomputed static charts)
- **App:** Streamlit
- **Persistence:** joblib

## Data Source

[Lending Club Loan Data](https://www.kaggle.com/datasets/wordsforthewise/lending-club) (Kaggle), covering accepted loans originated 2007–2018. The raw file is not included in this repository — download it from Kaggle and place it under `data/` as described above. Usage of the dataset is subject to Kaggle's and the original publisher's terms.

# 👤 Author

**Aryan Choudhary**

GitHub: [@Aryan-0-1](https://github.com/Aryan-0-1)

Repository: [Credit Risk Model — PD Scorecard & IFRS 9 ECL](https://github.com/Aryan-0-1/Credit-Risk-Model-PD-Scorecard-IFRS-9-ECL?utm_source=chatgpt.com)

---

## ⭐ If You Found This Project Useful

Consider giving the repository a ⭐ on GitHub.

Contributions, suggestions, and improvements are welcome.
