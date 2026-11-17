# Regional Credit Default Risk Mapping

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-189FDD?style=for-the-badge)
![GeoPandas](https://img.shields.io/badge/GeoPandas-139C5A?style=for-the-badge)
![SHAP](https://img.shields.io/badge/SHAP-FF0051?style=for-the-badge)
![Data-LendingClub](https://img.shields.io/badge/Data-LendingClub-2c5364?style=for-the-badge)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)

Does adding geographic risk information improve loan default prediction — and where are the highest-risk regions?

This project joins borrower-level lending data with state-level default rates and macroeconomic indicators, then tests whether geography carries predictive signal beyond what borrower features already capture.

## Research question

Two parts:

1. **Descriptive:** How does loan default rate vary by US state, and does it track regional economic conditions (e.g. unemployment)?
2. **Predictive:** Does adding a geographic risk feature improve a default prediction model over a borrower-features-only baseline?

**Note on the second question:** a null result is a legitimate finding. If state-level information is already captured indirectly by borrower features (income, DTI, credit grade), geography may add nothing — and reporting that honestly is more defensible than chasing an improvement that isn't there.

## Data

| Source | What | Where to get it |
|--------|------|-----------------|
| LendingClub Accepted Loans | Borrower features + loan outcomes + state | [Kaggle](https://www.kaggle.com/datasets/wordsforthewise/lending-club) — download `accepted_2007_to_2018Q4.csv.gz` into `data/raw/` |
| US States GeoJSON | State boundaries for mapping | Fetched automatically by the notebook (public domain, US Census cartographic boundaries) |
| State unemployment rates | Regional economic context (optional) | [FRED](https://fred.stlouisfed.org/) — series `<STATE>UR`, or skip this step |

`data/` is gitignored — the LendingClub file is large and is not redistributed here.

## Setup

```bash
pip install -r requirements.txt
```

```bash
jupyter notebook notebooks/01_regional_credit_risk.ipynb
```

## Method

1. Filter to resolved loans only (`Fully Paid` vs `Charged Off`/`Default`) — loans still current have no known outcome and must be excluded, not treated as non-default
2. Aggregate default rate by state; join to state geometries; render a choropleth
3. Train XGBoost on borrower features only → baseline PR-AUC / ROC-AUC
4. Add geographic risk feature → retrain, compare on the same held-out test set
5. SHAP (TreeExplainer) to check whether geography actually surfaces as an important feature or is redundant

### Leakage guardrails

The main trap in this project is computing state default rates on the full dataset and then using them as a feature — that leaks test-set outcomes into training. The notebook computes state-level rates **from the training split only** and maps them onto the test split. Unseen states in the test set fall back to the training-set global mean.

## Results

<!-- Fill in after running. Report the actual numbers, including if geography did not help. -->

| Model | ROC-AUC | PR-AUC |
|-------|---------|--------|
| Borrower features only | TBD | TBD |
| + geographic risk feature | TBD | TBD |

## Repo layout

```
data/raw/          # source data (gitignored)
data/processed/    # intermediate outputs (gitignored)
notebooks/         # analysis notebook
outputs/           # figures, maps, model artifacts
```
