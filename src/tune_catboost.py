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


CONFIGS = {
    "baseline": {
        "depth": 8,
        "learning_rate": 0.04,
        "l2_leaf_reg": 5.0,
        "random_strength": 1.0,
    },
    "shallow_regularized": {
        "depth": 7,
        "learning_rate": 0.05,
        "l2_leaf_reg": 10.0,
        "random_strength": 0.5,
    },
    "deep_slow": {
        "depth": 9,
        "learning_rate": 0.03,
        "l2_leaf_reg": 8.0,
        "random_strength": 0.5,
    },
    "strong_regularization": {
        "depth": 8,
        "learning_rate": 0.03,
        "l2_leaf_reg": 15.0,
        "random_strength": 0.2,
    },
}


def transformed_target(frame: pd.DataFrame) -> np.ndarray:
    rate_per_mile = (
        frame[TARGET].to_numpy(dtype=float)
        / frame["distance"].to_numpy(dtype=float)
    )

    return np.log1p(rate_per_mile)


def inverse_target(
    transformed_predictions: np.ndarray,
    frame: pd.DataFrame,
) -> np.ndarray:

    rate_per_mile = np.expm1(
        transformed_predictions
    )

    rate = (
        rate_per_mile
        * frame["distance"].to_numpy(dtype=float)
    )

    return np.maximum(rate, 1.0)


def calculate_metrics(
    actual: np.ndarray,
    predicted: np.ndarray,
) -> dict:

    errors = np.abs(actual - predicted)

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
            errors.sum()
            / np.abs(actual).sum()
            * 100
        ),
        "median_ae": np.median(errors),
        "p95_ae": np.quantile(
            errors,
            0.95,
        ),
    }


def fit_configuration(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    params: dict,
) -> tuple[np.ndarray, int]:

    X_train = prepare_features(
        train,
        include_quote_signal=False,
    )

    X_valid = prepare_features(
        valid,
        include_quote_signal=False,
    )

    y_train = transformed_target(train)
    y_valid = transformed_target(valid)

    categorical_features = [
        column
        for column in CATEGORICAL_FEATURES
        if column in X_train.columns
    ]

    model = CatBoostRegressor(
        iterations=1800,
        loss_function="MAE",
        eval_metric="MAE",
        random_seed=RANDOM_SEED,
        verbose=False,
        allow_writing_files=False,
        thread_count=-1,
        **params,
    )

    model.fit(
        X_train,
        y_train,
        cat_features=categorical_features,
        eval_set=(X_valid, y_valid),
        early_stopping_rounds=180,
        use_best_model=True,
        verbose=False,
    )

    predictions = inverse_target(
        model.predict(X_valid),
        valid,
    )

    best_iteration = (
        model.get_best_iteration()
    )

    return predictions, best_iteration


def main() -> None:
    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    data = pd.read_csv(DATA_PATH)

    data["date"] = pd.to_datetime(
        data["date"],
        errors="raise",
    )

    results = []

    print("=" * 125)
    print("CATBOOST LIMITED HYPERPARAMETER BENCHMARK")
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

        actual = valid[TARGET].to_numpy()

        for config_name, params in CONFIGS.items():

            predictions, best_iteration = (
                fit_configuration(
                    train,
                    valid,
                    params,
                )
            )

            scores = calculate_metrics(
                actual,
                predictions,
            )

            print(
                f"{fold['name']:10s} | "
                f"{config_name:22s} | "
                f"MAE=${scores['mae']:8.2f} | "
                f"RMSE=${scores['rmse']:8.2f} | "
                f"WAPE={scores['wape_pct']:6.2f}% | "
                f"MedianAE=${scores['median_ae']:7.2f} | "
                f"P95AE=${scores['p95_ae']:7.2f} | "
                f"best_iter={best_iteration}"
            )

            results.append(
                {
                    "fold": fold["name"],
                    "config": config_name,
                    "best_iteration": best_iteration,
                    **scores,
                    **params,
                }
            )

    results_frame = pd.DataFrame(results)

    output_path = (
        REPORTS_DIR
        / "catboost_tuning_results.csv"
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
        .groupby("config")
        .agg(
            mae=("mae", "mean"),
            mae_std=("mae", "std"),
            rmse=("rmse", "mean"),
            wape_pct=("wape_pct", "mean"),
            median_ae=("median_ae", "mean"),
            p95_ae=("p95_ae", "mean"),
            avg_best_iteration=(
                "best_iteration",
                "mean",
            ),
        )
        .sort_values("mae")
    )

    print(
        summary
        .round(2)
        .to_string()
    )

    print()
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
