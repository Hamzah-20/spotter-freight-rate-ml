from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from features import CATEGORICAL_FEATURES, prepare_features


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "train_test.csv"
REPORTS_DIR = PROJECT_ROOT / "reports"

TARGET = "posted_rate"
RANDOM_SEED = 42


FOLDS = [
    {
        "name": "August",
        "train_end": "2025-07-31",
        "valid_start": "2025-08-01",
        "valid_end": "2025-08-31",
    },
    {
        "name": "September",
        "train_end": "2025-08-31",
        "valid_start": "2025-09-01",
        "valid_end": "2025-09-30",
    },
    {
        "name": "October",
        "train_end": "2025-09-30",
        "valid_start": "2025-10-01",
        "valid_end": "2025-10-31",
    },
]


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    wape = (
        np.abs(y_true - y_pred).sum()
        / np.abs(y_true).sum()
        * 100
    )

    median_ae = np.median(np.abs(y_true - y_pred))

    return {
        "mae": mae,
        "rmse": rmse,
        "wape_pct": wape,
        "median_ae": median_ae,
    }


def rate_per_mile_baseline(
    train: pd.DataFrame,
    valid: pd.DataFrame,
) -> np.ndarray:
    """Business baseline: equipment-specific median rate per mile."""

    safe_train = train[train["distance"] > 0].copy()

    safe_train["rate_per_mile"] = (
        safe_train[TARGET] / safe_train["distance"]
    )

    equipment_medians = (
        safe_train
        .groupby("equipment")["rate_per_mile"]
        .median()
    )

    global_median = safe_train["rate_per_mile"].median()

    validation_rpm = (
        valid["equipment"]
        .map(equipment_medians)
        .fillna(global_median)
    )

    return (
        validation_rpm.to_numpy()
        * valid["distance"].to_numpy()
    )


def fit_catboost(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    include_quote_signal: bool,
) -> np.ndarray:

    X_train = prepare_features(
        train,
        include_quote_signal=include_quote_signal,
    )

    X_valid = prepare_features(
        valid,
        include_quote_signal=include_quote_signal,
    )

    y_train = train[TARGET].to_numpy()
    y_valid = valid[TARGET].to_numpy()

    categorical_features = [
        column
        for column in CATEGORICAL_FEATURES
        if column in X_train.columns
    ]

    model = CatBoostRegressor(
        iterations=1200,
        learning_rate=0.04,
        depth=8,
        loss_function="MAE",
        eval_metric="MAE",
        l2_leaf_reg=5.0,
        random_seed=RANDOM_SEED,
        verbose=False,
        allow_writing_files=False,
    )

    model.fit(
        X_train,
        y_train,
        cat_features=categorical_features,
        eval_set=(X_valid, y_valid),
        early_stopping_rounds=120,
        use_best_model=True,
        verbose=False,
    )

    predictions = model.predict(X_valid)

    return np.maximum(predictions, 1.0)


def print_metrics(
    fold_name: str,
    model_name: str,
    scores: dict,
) -> None:
    print(
        f"{fold_name:10s} | "
        f"{model_name:28s} | "
        f"MAE=${scores['mae']:8.2f} | "
        f"RMSE=${scores['rmse']:8.2f} | "
        f"WAPE={scores['wape_pct']:6.2f}% | "
        f"MedianAE=${scores['median_ae']:7.2f}"
    )


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(DATA_PATH)
    data["date"] = pd.to_datetime(data["date"], errors="raise")

    results = []

    print("=" * 115)
    print("FORWARD TIME-BASED BASELINE BENCHMARK")
    print("=" * 115)

    for fold in FOLDS:
        train_end = pd.Timestamp(fold["train_end"])
        valid_start = pd.Timestamp(fold["valid_start"])
        valid_end = pd.Timestamp(fold["valid_end"])

        train = data[data["date"] <= train_end].copy()

        valid = data[
            (data["date"] >= valid_start)
            & (data["date"] <= valid_end)
        ].copy()

        print()
        print(
            f"{fold['name']}: "
            f"train={len(train):,}, validation={len(valid):,}"
        )

        y_valid = valid[TARGET].to_numpy()

        # --------------------------------------------------------------
        # Business baseline
        # --------------------------------------------------------------
        baseline_predictions = rate_per_mile_baseline(
            train,
            valid,
        )

        baseline_scores = metrics(
            y_valid,
            baseline_predictions,
        )

        print_metrics(
            fold["name"],
            "Median rate/mile baseline",
            baseline_scores,
        )

        results.append(
            {
                "fold": fold["name"],
                "model": "median_rate_per_mile",
                **baseline_scores,
            }
        )

        # --------------------------------------------------------------
        # CatBoost WITHOUT quote_signal
        # --------------------------------------------------------------
        no_quote_predictions = fit_catboost(
            train,
            valid,
            include_quote_signal=False,
        )

        no_quote_scores = metrics(
            y_valid,
            no_quote_predictions,
        )

        print_metrics(
            fold["name"],
            "CatBoost - no quote_signal",
            no_quote_scores,
        )

        results.append(
            {
                "fold": fold["name"],
                "model": "catboost_without_quote_signal",
                **no_quote_scores,
            }
        )

        # --------------------------------------------------------------
        # CatBoost WITH quote_signal
        # --------------------------------------------------------------
        quote_predictions = fit_catboost(
            train,
            valid,
            include_quote_signal=True,
        )

        quote_scores = metrics(
            y_valid,
            quote_predictions,
        )

        print_metrics(
            fold["name"],
            "CatBoost - with quote_signal",
            quote_scores,
        )

        results.append(
            {
                "fold": fold["name"],
                "model": "catboost_with_quote_signal",
                **quote_scores,
            }
        )

    results_frame = pd.DataFrame(results)

    output_path = (
        REPORTS_DIR
        / "time_validation_baseline_results.csv"
    )

    results_frame.to_csv(
        output_path,
        index=False,
    )

    print()
    print("=" * 115)
    print("AVERAGE PERFORMANCE")
    print("=" * 115)

    summary = (
        results_frame
        .groupby("model")[
            ["mae", "rmse", "wape_pct", "median_ae"]
        ]
        .mean()
        .sort_values("mae")
    )

    print(summary.round(2).to_string())

    print()
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
