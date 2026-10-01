from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import KFold

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


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict:
    absolute_errors = np.abs(y_true - y_pred)

    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": np.sqrt(
            mean_squared_error(y_true, y_pred)
        ),
        "wape_pct": (
            absolute_errors.sum()
            / np.abs(y_true).sum()
            * 100
        ),
        "median_ae": np.median(absolute_errors),
    }


def build_model(
    *,
    screening: bool = False,
    seed: int = RANDOM_SEED,
) -> CatBoostRegressor:

    if screening:
        return CatBoostRegressor(
            iterations=450,
            learning_rate=0.06,
            depth=7,
            loss_function="MAE",
            l2_leaf_reg=5.0,
            random_seed=seed,
            verbose=False,
            allow_writing_files=False,
        )

    return CatBoostRegressor(
        iterations=1200,
        learning_rate=0.04,
        depth=8,
        loss_function="MAE",
        eval_metric="MAE",
        l2_leaf_reg=5.0,
        random_seed=seed,
        verbose=False,
        allow_writing_files=False,
    )


def cross_fitted_training_predictions(
    train: pd.DataFrame,
) -> np.ndarray:
    """
    Generate predictions for every training row from a model that did not
    train on that row.

    These random inner folds are used ONLY for label-quality screening.
    They are not used to estimate temporal model generalization.
    """

    X = prepare_features(
        train,
        include_quote_signal=False,
    )

    y = train[TARGET].to_numpy()

    categorical_features = [
        column
        for column in CATEGORICAL_FEATURES
        if column in X.columns
    ]

    kfold = KFold(
        n_splits=4,
        shuffle=True,
        random_state=RANDOM_SEED,
    )

    predictions = np.zeros(
        len(train),
        dtype=float,
    )

    for inner_fold, (fit_idx, predict_idx) in enumerate(
        kfold.split(X),
        start=1,
    ):
        print(
            f"      screening fold "
            f"{inner_fold}/4..."
        )

        model = build_model(
            screening=True,
            seed=RANDOM_SEED + inner_fold,
        )

        model.fit(
            X.iloc[fit_idx],
            y[fit_idx],
            cat_features=categorical_features,
            verbose=False,
        )

        predictions[predict_idx] = model.predict(
            X.iloc[predict_idx]
        )

    return np.maximum(
        predictions,
        1.0,
    )


def identify_extreme_labels(
    train: pd.DataFrame,
    cross_fitted_predictions: np.ndarray,
) -> pd.DataFrame:

    result = train.copy()

    result["screen_prediction"] = (
        cross_fitted_predictions
    )

    result["prediction_ratio"] = (
        result[TARGET]
        / result["screen_prediction"]
    )

    result["log_ratio"] = np.log(
        result[TARGET].clip(lower=1e-6)
        / result["screen_prediction"].clip(lower=1e-6)
    )

    median_log_ratio = (
        result["log_ratio"].median()
    )

    mad = np.median(
        np.abs(
            result["log_ratio"]
            - median_log_ratio
        )
    )

    if mad <= 0:
        raise RuntimeError(
            "MAD is zero; cannot calculate robust anomaly scores."
        )

    result["robust_z"] = (
        0.6745
        * (
            result["log_ratio"]
            - median_log_ratio
        )
        / mad
    )

    result["extreme_label"] = (
        (result["robust_z"].abs() >= 6.0)
        & (
            (result["prediction_ratio"] >= 2.0)
            | (result["prediction_ratio"] <= 0.5)
        )
    )

    return result


def fit_final_model(
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

    y_train = train[TARGET].to_numpy()
    y_valid = valid[TARGET].to_numpy()

    categorical_features = [
        column
        for column in CATEGORICAL_FEATURES
        if column in X_train.columns
    ]

    model = build_model()

    model.fit(
        X_train,
        y_train,
        cat_features=categorical_features,
        eval_set=(X_valid, y_valid),
        early_stopping_rounds=120,
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


def display_result(
    fold: str,
    model: str,
    scores: dict,
) -> None:
    print(
        f"{fold:10s} | "
        f"{model:18s} | "
        f"MAE=${scores['mae']:8.2f} | "
        f"RMSE=${scores['rmse']:8.2f} | "
        f"WAPE={scores['wape_pct']:6.2f}% | "
        f"MedianAE=${scores['median_ae']:7.2f}"
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
    anomaly_records = []

    print("=" * 115)
    print("RAW VS ROBUST-CLEANED TRAINING")
    print("=" * 115)

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
        print("-" * 115)
        print(
            f"{fold['name']} | "
            f"historical train={len(train):,} | "
            f"untouched future validation={len(valid):,}"
        )
        print("-" * 115)

        # --------------------------------------------------------------
        # Cross-fitted anomaly screening using TRAINING DATA ONLY
        # --------------------------------------------------------------
        print(
            "  [1/3] Generating cross-fitted "
            "training predictions..."
        )

        screening_predictions = (
            cross_fitted_training_predictions(
                train
            )
        )

        screened = identify_extreme_labels(
            train,
            screening_predictions,
        )

        anomaly_mask = (
            screened["extreme_label"]
        )

        anomaly_count = int(
            anomaly_mask.sum()
        )

        anomaly_pct = (
            anomaly_count
            / len(screened)
            * 100
        )

        print(
            f"  Detected extreme training labels: "
            f"{anomaly_count:,} "
            f"({anomaly_pct:.2f}%)"
        )

        detected = screened[
            anomaly_mask
        ].copy()

        detected["outer_fold"] = (
            fold["name"]
        )

        anomaly_records.append(
            detected[
                [
                    "outer_fold",
                    "load_id",
                    "date",
                    "pickup",
                    "delivery",
                    "distance",
                    "equipment",
                    TARGET,
                    "screen_prediction",
                    "prediction_ratio",
                    "robust_z",
                ]
            ]
        )

        cleaned_train = screened[
            ~anomaly_mask
        ].drop(
            columns=[
                "screen_prediction",
                "prediction_ratio",
                "log_ratio",
                "robust_z",
                "extreme_label",
            ]
        )

        # --------------------------------------------------------------
        # RAW MODEL
        # --------------------------------------------------------------
        print(
            "  [2/3] Training raw model..."
        )

        raw_predictions = fit_final_model(
            train,
            valid,
        )

        raw_scores = calculate_metrics(
            valid[TARGET].to_numpy(),
            raw_predictions,
        )

        display_result(
            fold["name"],
            "Raw training",
            raw_scores,
        )

        results.append(
            {
                "fold": fold["name"],
                "model": "raw_training",
                "training_rows": len(train),
                "removed_rows": 0,
                **raw_scores,
            }
        )

        # --------------------------------------------------------------
        # CLEANED MODEL
        # --------------------------------------------------------------
        print(
            "  [3/3] Training cleaned model..."
        )

        clean_predictions = fit_final_model(
            cleaned_train,
            valid,
        )

        clean_scores = calculate_metrics(
            valid[TARGET].to_numpy(),
            clean_predictions,
        )

        display_result(
            fold["name"],
            "Robust cleaned",
            clean_scores,
        )

        results.append(
            {
                "fold": fold["name"],
                "model": "robust_cleaned",
                "training_rows": len(cleaned_train),
                "removed_rows": anomaly_count,
                **clean_scores,
            }
        )

        improvement = (
            (
                raw_scores["mae"]
                - clean_scores["mae"]
            )
            / raw_scores["mae"]
            * 100
        )

        print(
            f"  MAE improvement from cleaning: "
            f"{improvement:+.2f}%"
        )

    # ------------------------------------------------------------------
    # Save outputs
    # ------------------------------------------------------------------
    results_frame = pd.DataFrame(
        results
    )

    results_path = (
        REPORTS_DIR
        / "outlier_handling_comparison.csv"
    )

    results_frame.to_csv(
        results_path,
        index=False,
    )

    all_anomalies = pd.concat(
        anomaly_records,
        ignore_index=True,
    )

    anomalies_path = (
        REPORTS_DIR
        / "cross_fitted_training_anomalies.csv"
    )

    all_anomalies.to_csv(
        anomalies_path,
        index=False,
    )

    print()
    print("=" * 115)
    print("AVERAGE PERFORMANCE")
    print("=" * 115)

    summary = (
        results_frame
        .groupby("model")[
            [
                "mae",
                "rmse",
                "wape_pct",
                "median_ae",
            ]
        ]
        .mean()
        .sort_values("mae")
    )

    print(
        summary.round(2).to_string()
    )

    print()
    print("=" * 115)
    print("AVERAGE TRAINING CLEANING")
    print("=" * 115)

    cleaning_summary = (
        results_frame[
            results_frame["model"]
            == "robust_cleaned"
        ][
            [
                "fold",
                "training_rows",
                "removed_rows",
            ]
        ]
    )

    cleaning_summary = (
        cleaning_summary.copy()
    )

    cleaning_summary["removed_pct"] = (
        cleaning_summary["removed_rows"]
        / (
            cleaning_summary["training_rows"]
            + cleaning_summary["removed_rows"]
        )
        * 100
    )

    print(
        cleaning_summary
        .round(2)
        .to_string(index=False)
    )

    print()
    print("Saved:")
    print(
        f"  {results_path}"
    )
    print(
        f"  {anomalies_path}"
    )


if __name__ == "__main__":
    main()
