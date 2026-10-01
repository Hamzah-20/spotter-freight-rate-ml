# Freight Rate Prediction — Machine Learning Engineer Assessment

## Overview

This repository contains my solution for the Spotter Machine Learning Engineer freight-rate prediction assessment.

The objective is to predict `posted_rate` for 12,000 future freight loads covering November and December 2025, using 48,000 labeled loads from January through October 2025.

The final solution uses a small CatBoost ensemble designed around temporal validation and robust freight-rate modeling.

## Final Model

The final prediction is:

- 75% CatBoost predicting `log1p(posted_rate / distance)`
- 25% CatBoost predicting `posted_rate` directly

The transformed model uses:

- depth: 7
- learning rate: 0.05
- L2 regularization: 10
- random strength: 0.5
- MAE loss

The raw-rate model provides a complementary prediction that slightly improved forward validation performance.

`quote_signal` was excluded from the final production models because its relationship with the target was temporally unstable across development months.

## Validation Strategy

Because the final prediction set occurs strictly after the labeled development period, random train/test splitting would not represent the production setting.

I used forward time-based validation:

| Training period | Validation period |
| --- | --- |
| Jan–Jul 2025 | Aug 2025 |
| Jan–Aug 2025 | Sep 2025 |
| Jan–Sep 2025 | Oct 2025 |

This tests how the model performs when predicting genuinely unseen future periods.

## Validation Results

### Business baseline

Equipment-specific median rate-per-mile baseline:

- Average MAE: $233.25
- Average WAPE: 9.83%

### Primary CatBoost model

CatBoost using `log1p(rate_per_mile)`:

- Average MAE: approximately $98.82
- Average WAPE: 4.16%

### Final ensemble

75% transformed-rate model + 25% raw-rate model:

- Average MAE: $98.49
- Average WAPE: 4.14%
- Average Median Absolute Error: $29.85
- Average P95 Absolute Error: $158.55

The ensemble reduced average MAE by approximately 58% relative to the business baseline.

## Data Quality Findings

Several data-quality issues were identified during EDA:

- 300 missing training weights
- 374 missing training `market_index` values
- 292 negative training weights
- approximately 1.4% extreme target observations
- eight pickup/delivery cities appear only in the final validation period
- substantial temporal instability in `quote_signal`

Negative weights were converted to their absolute magnitude while preserving a `weight_was_negative` indicator.

Missing values are retained and handled by CatBoost where appropriate.

Extreme target observations were investigated using cross-fitted residual analysis. Removing these observations produced only a small and inconsistent validation improvement, so the final model keeps all labeled rows and uses MAE as a robust training loss.

## Feature Engineering

The model includes:

- pickup and delivery cities
- equipment type
- lane category
- geographic coordinates
- latitude/longitude deltas
- geographic-distance proxy
- distance and log-distance
- corrected weight and log-weight
- weight-per-mile
- missing-value indicators
- calendar features
- cyclical month/day-of-week/day-of-year features

Geographic features provide useful fallback information for cities not observed in the labeled training set.

## December Scenario

The provided December scenario fixes:

- Pickup: Lexington
- Delivery: Fort Wayne
- Distance: 360 miles
- Equipment: Dry Van
- Weight: 32,000 lb

Only the date changes.

The final December predictions range from approximately $809 to $835, with a mean near $827.

As a sanity check:

- historical Lexington → Fort Wayne Dry Van median rate: approximately $808
- comparable 300–420 mile Dry Van median rate: approximately $844

## Repository Structure

```text
.
├── artifacts/
│   ├── final_raw_model.cbm
│   ├── final_transformed_model.cbm
│   ├── raw_feature_importance.csv
│   ├── transformed_feature_importance.csv
│   └── training_metadata.json
├── data/
│   ├── train_test.csv
│   ├── validation.csv
│   ├── validation_predictions_template.csv
│   ├── december_chart_inputs.csv
│   └── december_chart_inputs_original.csv
├── reports/
├── scorer_results/
│   └── candidate_december.png
├── src/
│   ├── features.py
│   ├── eda.py
│   ├── evaluate_baselines.py
│   ├── analyze_residuals.py
│   ├── compare_outlier_handling.py
│   ├── compare_targets.py
│   ├── tune_catboost.py
│   ├── compare_ensemble.py
│   ├── train_final.py
│   └── sanity_check.py
├── validation_predictions.csv
├── score.py
├── requirements.txt
└── README.md
```

## Setup

Create and activate a virtual environment:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

## Run EDA

```bash
python src/eda.py
```

## Run Validation Experiments

```bash
python src/evaluate_baselines.py
python src/analyze_residuals.py
python src/compare_outlier_handling.py
python src/compare_targets.py
python src/tune_catboost.py
python src/compare_ensemble.py
```

## Train Final Models

```bash
python src/train_final.py
```

This produces:

- `validation_predictions.csv`
- final CatBoost model artifacts
- completed December predictions

## Run Official Scorer

```bash
python score.py --predictions validation_predictions.csv --december-predictions data/december_chart_inputs.csv
```

Expected validation:

```text
Validated 12,000 final predictions.
Validated 31 fixed December predictions.
Created chart: scorer_results/candidate_december.png
```

Final hidden validation metrics are calculated by Spotter after submission.

## Reproducibility

A fixed random seed of `42` is used throughout the modeling pipeline.

The final training iteration counts are calibrated using October 2025 as the most recent labeled validation period, after which the final models are retrained on all labeled January–October data.

## Author

Hamzah Al-Basyouni

## Data Availability

The original assessment datasets are intentionally not committed to this repository. Place the assessment-provided CSV files under `data/` using the filenames expected by the scripts before running the pipeline.

Generated submission outputs, summarized experiment results, and the official scorer chart are included.
