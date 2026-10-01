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

TARGET_MODES = [
    "raw_rate",
    "log_rate",
    "rate_per_mile",
    "log_rate_per_mile",
]


def metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict:
    error = np.abs(y_true - y_pred)

    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": np.sqrt(
            mean_squared_error(y_true, y_pred)
        ),
        "wape_pct": (
            error.sum()
            / np.abs(y_true).sum()
            * 100
        ),
        "median_ae": np.median(error),
        "p95_ae": np.quantile(error, 0.95),
    }


def make_training_target(
    frame: pd.DataFrame,
    mode: str,
) -> np.ndarray:

    rate = frame[TARGET].to_numpy(dtype=float)
    distance = frame["distance"].to_numpy(dtype=float)

    if mode == "raw_rate":
        return rate

    if mode == "log_rate":
        return np.log1p(rate)

    if mode == "rate_per_mile":
        return rate / distance

    if mode == "log_rate_per_mile":
        return np.log1p(rate / distance)

    raise ValueError(
        f"Unknown target mode: {mode}"
    )


def inverse_prediction(
    predictions: np.ndarray,
    valid: pd.DataFrame,
    mode: str,
) -> np.ndarray:

    distance = valid["distance"].to_numpy(dtype=float)

    if mode == "raw_rate":
        rate_predictions = predictions

    elif mode == "log_rate":
        rate_predictions = np.expm1(predictions)

    elif mode == "rate_per_mile":
        rate_predictions = predictions * distance

    elif mode == "log_rate_per_mile":
        rate_predictions = (
            np.expm1(predictions)
            * distance
        )

    else:
        raise ValueError(
            f"Unknown target mode: {mode}"
        )

    return np.maximum(
        rate_predictions,
        1.0,
    )


def fit_predict(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    mode: str,
) -> np.ndarray:

    X_train = prepare_features(
        train,
        include_quote_signal=False,
    )

    X_valid = prepare_features(
        valid,
        include_quote_signal=False,
    )

    y_train = make_training_target(
        train,
        mode,
    )

    y_valid_transformed = make_training_target(
        valid,
        mode,
    )

    categorical_features = [
        column
        for column in CATEGORICAL_FEATURES
        if column in X_train.columns
    ]

    model = CatBoostRegressor(
        iterations=1400,
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
        eval_set=(
            X_valid,
            y_valid_transformed,
        ),
        early_stopping_rounds=150,
        use_best_model=True,
        verbose=False,
    )

    transformed_predictions = (
        model.predict(X_valid)
    )

    return inverse_prediction(
        transformed_predictions,
        valid,
        mode,
    )


def print_result(
    fold: str,
    mode: str,
    scores: dict,
) -> None:
    print(
        f"{fold:10s} | "
        f"{mode:20s} | "
        f"MAE=${scores['mae']:8.2f} | "
        f"RMSE=${scores['rmse']:8.2f} | "
        f"WAPE={scores['wape_pct']:6.2f}% | "
        f"MedianAE=${scores['median_ae']:7.2f} | "
        f"P95AE=${scores['p95_ae']:7.2f}"
    )


def main() -> None:
    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    data = pd.read_csv(
        DATA_PATH
    )

    data["date"] = pd.to_datetime(
        data["date"],
        errors="raise",
    )

    results = []

    print("=" * 125)
    print("TARGET REPRESENTATION BENCHMARK")
    print("=" * 125)

    for fold in FOLDS:
        train_end = pd.Timestamp(
            fold["train_end"]
        )

        valid_start = pd.Timestamp(
            fold["valid_start"]
        )

        valid_end = pd.Timestamp(
            fold["valid_end"]
        )

        train = data[
            data["date"] <= train_end
        ].copy()

        valid = data[
            (data["date"] >= valid_start)
            & (data["date"] <= valid_end)
        ].copy()

        print()
        print(
            f"{fold['name']}: "
            f"train={len(train):,}, "
            f"validation={len(valid):,}"
        )

        for mode in TARGET_MODES:
            predictions = fit_predict(
                train,
                valid,
                mode,
            )

            scores = metrics(
                valid[TARGET].to_numpy(),
                predictions,
            )

            print_result(
                fold["name"],
                mode,
                scores,
            )

            results.append(
                {
                    "fold": fold["name"],
                    "target_mode": mode,
                    **scores,
                }
            )

    results_frame = pd.DataFrame(
        results
    )

    output = (
        REPORTS_DIR
        / "target_representation_results.csv"
    )

    results_frame.to_csv(
        output,
        index=False,
    )

    print()
    print("=" * 125)
    print("AVERAGE PERFORMANCE")
    print("=" * 125)

    summary = (
        results_frame
        .groupby("target_mode")[
            [
                "mae",
                "rmse",
                "wape_pct",
                "median_ae",
                "p95_ae",
            ]
        ]
        .mean()
        .sort_values("mae")
    )

    print(
        summary
        .round(2)
        .to_string()
    )

    print()
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
