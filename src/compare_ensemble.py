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

BLEND_WEIGHTS = [
    1.00,
    0.90,
    0.75,
    0.50,
]


def calculate_metrics(
    actual: np.ndarray,
    predicted: np.ndarray,
) -> dict:

    absolute_error = np.abs(
        actual - predicted
    )

    return {
        "mae": mean_absolute_error(
            actual,
            predicted,
        ),
        "rmse": np.sqrt(
            mean_squared_error(
                actual,
                predicted,
            )
        ),
        "wape_pct": (
            absolute_error.sum()
            / np.abs(actual).sum()
            * 100
        ),
        "median_ae": np.median(
            absolute_error
        ),
        "p95_ae": np.quantile(
            absolute_error,
            0.95,
        ),
    }


def categorical_columns(
    frame: pd.DataFrame,
) -> list[str]:

    return [
        column
        for column in CATEGORICAL_FEATURES
        if column in frame.columns
    ]


def train_transformed_model(
    train: pd.DataFrame,
    valid: pd.DataFrame,
) -> np.ndarray:

    X_train = prepare_features(
        train,
        include_quote_signal=False,
    )

    X_valid = prepare_features(
        valid,
        include_quote_signal=False,
    )

    y_train = np.log1p(
        train[TARGET].to_numpy(dtype=float)
        / train["distance"].to_numpy(dtype=float)
    )

    y_valid = np.log1p(
        valid[TARGET].to_numpy(dtype=float)
        / valid["distance"].to_numpy(dtype=float)
    )

    model = CatBoostRegressor(
        iterations=1800,
        depth=7,
        learning_rate=0.05,
        l2_leaf_reg=10.0,
        random_strength=0.5,
        loss_function="MAE",
        eval_metric="MAE",
        random_seed=RANDOM_SEED,
        verbose=False,
        allow_writing_files=False,
        thread_count=-1,
    )

    model.fit(
        X_train,
        y_train,
        cat_features=categorical_columns(
            X_train
        ),
        eval_set=(
            X_valid,
            y_valid,
        ),
        early_stopping_rounds=180,
        use_best_model=True,
        verbose=False,
    )

    predicted_log_rpm = model.predict(
        X_valid
    )

    predicted_rpm = np.expm1(
        predicted_log_rpm
    )

    predicted_rate = (
        predicted_rpm
        * valid["distance"].to_numpy(
            dtype=float
        )
    )

    return np.maximum(
        predicted_rate,
        1.0,
    )


def train_raw_model(
    train: pd.DataFrame,
    valid: pd.DataFrame,
) -> np.ndarray:

    X_train = prepare_features(
        train,
        include_quote_signal=False,
    )

    X_valid = prepare_features(
        valid,
        include_quote_signal=False,
    )

    y_train = train[TARGET].to_numpy(
        dtype=float
    )

    y_valid = valid[TARGET].to_numpy(
        dtype=float
    )

    model = CatBoostRegressor(
        iterations=1400,
        depth=8,
        learning_rate=0.04,
        l2_leaf_reg=5.0,
        random_strength=1.0,
        loss_function="MAE",
        eval_metric="MAE",
        random_seed=RANDOM_SEED,
        verbose=False,
        allow_writing_files=False,
        thread_count=-1,
    )

    model.fit(
        X_train,
        y_train,
        cat_features=categorical_columns(
            X_train
        ),
        eval_set=(
            X_valid,
            y_valid,
        ),
        early_stopping_rounds=150,
        use_best_model=True,
        verbose=False,
    )

    predictions = model.predict(
        X_valid
    )

    return np.maximum(
        predictions,
        1.0,
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
    print("PRIMARY MODEL VS SIMPLE ENSEMBLE")
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

        print(
            "  Training transformed model..."
        )

        transformed_predictions = (
            train_transformed_model(
                train,
                valid,
            )
        )

        print(
            "  Training raw-rate model..."
        )

        raw_predictions = (
            train_raw_model(
                train,
                valid,
            )
        )

        actual = valid[TARGET].to_numpy(
            dtype=float
        )

        for transformed_weight in BLEND_WEIGHTS:

            raw_weight = (
                1.0
                - transformed_weight
            )

            predictions = (
                transformed_weight
                * transformed_predictions
                + raw_weight
                * raw_predictions
            )

            scores = calculate_metrics(
                actual,
                predictions,
            )

            model_name = (
                f"{int(transformed_weight * 100)}% transformed "
                f"+ {int(raw_weight * 100)}% raw"
            )

            print(
                f"{fold['name']:10s} | "
                f"{model_name:29s} | "
                f"MAE=${scores['mae']:8.2f} | "
                f"RMSE=${scores['rmse']:8.2f} | "
                f"WAPE={scores['wape_pct']:6.2f}% | "
                f"MedianAE=${scores['median_ae']:7.2f} | "
                f"P95AE=${scores['p95_ae']:7.2f}"
            )

            results.append(
                {
                    "fold": fold["name"],
                    "transformed_weight": (
                        transformed_weight
                    ),
                    "raw_weight": (
                        raw_weight
                    ),
                    **scores,
                }
            )

    results_frame = pd.DataFrame(
        results
    )

    output_path = (
        REPORTS_DIR
        / "ensemble_results.csv"
    )

    results_frame.to_csv(
        output_path,
        index=False,
    )

    print()
    print("=" * 125)
    print("AVERAGE PERFORMANCE")
    print("=" * 125)

    summary = (
        results_frame
        .groupby(
            [
                "transformed_weight",
                "raw_weight",
            ]
        )
        .agg(
            mae=("mae", "mean"),
            mae_std=("mae", "std"),
            rmse=("rmse", "mean"),
            wape_pct=("wape_pct", "mean"),
            median_ae=("median_ae", "mean"),
            p95_ae=("p95_ae", "mean"),
        )
        .sort_values("mae")
    )

    print(
        summary
        .round(2)
        .to_string()
    )

    print()
    print(
        f"Saved: {output_path}"
    )


if __name__ == "__main__":
    main()
