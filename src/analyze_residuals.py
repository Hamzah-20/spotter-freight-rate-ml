from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error

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


def fit_predict(
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

    return np.maximum(
        model.predict(X_valid),
        1.0,
    )


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

    out_of_time_rows = []

    print("=" * 100)
    print("OUT-OF-TIME RESIDUAL ANALYSIS")
    print("=" * 100)

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

        predictions = fit_predict(
            train,
            valid,
        )

        valid["prediction"] = predictions

        valid["residual"] = (
            valid[TARGET]
            - valid["prediction"]
        )

        valid["absolute_error"] = (
            valid["residual"].abs()
        )

        valid["prediction_ratio"] = (
            valid[TARGET]
            / valid["prediction"]
        )

        valid["log_ratio"] = np.log(
            valid[TARGET].clip(lower=1e-6)
            / valid["prediction"].clip(lower=1e-6)
        )

        valid["fold"] = fold["name"]

        out_of_time_rows.append(valid)

        print(
            f"{fold['name']:10s} | "
            f"rows={len(valid):,} | "
            f"MAE=${mean_absolute_error(valid[TARGET], predictions):,.2f}"
        )

    oof = pd.concat(
        out_of_time_rows,
        ignore_index=True,
    )

    # ------------------------------------------------------------------
    # Robust anomaly score using log-ratio residuals
    # ------------------------------------------------------------------
    median_log_ratio = oof["log_ratio"].median()

    mad = np.median(
        np.abs(
            oof["log_ratio"]
            - median_log_ratio
        )
    )

    if mad == 0:
        raise RuntimeError(
            "MAD is zero; robust anomaly score cannot be calculated."
        )

    oof["robust_z"] = (
        0.6745
        * (
            oof["log_ratio"]
            - median_log_ratio
        )
        / mad
    )

    oof["abs_robust_z"] = (
        oof["robust_z"].abs()
    )

    # Conservative anomaly rule.
    oof["extreme_anomaly"] = (
        (oof["abs_robust_z"] >= 6.0)
        & (
            (oof["prediction_ratio"] >= 2.0)
            | (oof["prediction_ratio"] <= 0.5)
        )
    )

    print()
    print("=" * 100)
    print("RESIDUAL DISTRIBUTION")
    print("=" * 100)

    absolute_error = oof["absolute_error"]

    print(
        f"Rows analysed   : {len(oof):,}"
    )
    print(
        f"Median abs err  : ${absolute_error.median():,.2f}"
    )
    print(
        f"P90 abs err     : ${absolute_error.quantile(0.90):,.2f}"
    )
    print(
        f"P95 abs err     : ${absolute_error.quantile(0.95):,.2f}"
    )
    print(
        f"P99 abs err     : ${absolute_error.quantile(0.99):,.2f}"
    )
    print(
        f"Max abs err     : ${absolute_error.max():,.2f}"
    )

    print()
    print("=" * 100)
    print("PREDICTION RATIO DISTRIBUTION")
    print("=" * 100)

    ratio = oof["prediction_ratio"]

    for quantile in [
        0.001,
        0.005,
        0.01,
        0.05,
        0.50,
        0.95,
        0.99,
        0.995,
        0.999,
    ]:
        print(
            f"Q{quantile:>6.3f}: "
            f"{ratio.quantile(quantile):.4f}"
        )

    print()
    print("=" * 100)
    print("ROBUST LOG-RATIO ANALYSIS")
    print("=" * 100)

    print(
        f"Median log ratio : "
        f"{median_log_ratio:.6f}"
    )

    print(
        f"MAD              : "
        f"{mad:.6f}"
    )

    anomaly_count = int(
        oof["extreme_anomaly"].sum()
    )

    anomaly_pct = (
        anomaly_count / len(oof) * 100
    )

    print(
        f"Extreme anomalies: "
        f"{anomaly_count:,} "
        f"({anomaly_pct:.2f}%)"
    )

    # ------------------------------------------------------------------
    # Diagnostic metrics with and without extreme rows.
    # This is analysis only. We are NOT deleting them yet.
    # ------------------------------------------------------------------
    normal = oof[
        ~oof["extreme_anomaly"]
    ]

    anomalies = oof[
        oof["extreme_anomaly"]
    ]

    print()
    print("=" * 100)
    print("DIAGNOSTIC PERFORMANCE")
    print("=" * 100)

    print(
        f"All rows MAE      : "
        f"${oof['absolute_error'].mean():,.2f}"
    )

    print(
        f"Non-extreme MAE   : "
        f"${normal['absolute_error'].mean():,.2f}"
    )

    if len(anomalies) > 0:
        print(
            f"Extreme rows MAE  : "
            f"${anomalies['absolute_error'].mean():,.2f}"
        )

    print()
    print("=" * 100)
    print("ANOMALIES BY MONTH")
    print("=" * 100)

    monthly = (
        oof.groupby("fold")
        .agg(
            rows=("load_id", "size"),
            anomalies=("extreme_anomaly", "sum"),
            mae=("absolute_error", "mean"),
        )
    )

    monthly["anomaly_pct"] = (
        monthly["anomalies"]
        / monthly["rows"]
        * 100
    )

    print(
        monthly.round(2).to_string()
    )

    # ------------------------------------------------------------------
    # Largest residuals for manual inspection
    # ------------------------------------------------------------------
    columns = [
        "load_id",
        "fold",
        "date",
        "pickup",
        "delivery",
        "distance",
        "equipment",
        "weight",
        "market_index",
        "quote_signal",
        TARGET,
        "prediction",
        "absolute_error",
        "prediction_ratio",
        "robust_z",
        "extreme_anomaly",
    ]

    top_residuals = (
        oof.sort_values(
            "absolute_error",
            ascending=False,
        )[columns]
        .head(50)
    )

    anomalies_output = (
        oof[
            oof["extreme_anomaly"]
        ][columns]
        .sort_values(
            "absolute_error",
            ascending=False,
        )
    )

    oof[columns].to_csv(
        REPORTS_DIR
        / "oof_residual_analysis.csv",
        index=False,
    )

    top_residuals.to_csv(
        REPORTS_DIR
        / "top_50_residuals.csv",
        index=False,
    )

    anomalies_output.to_csv(
        REPORTS_DIR
        / "extreme_target_anomalies.csv",
        index=False,
    )

    print()
    print("=" * 100)
    print("TOP 15 LARGEST RESIDUALS")
    print("=" * 100)

    display_columns = [
        "load_id",
        "fold",
        "pickup",
        "delivery",
        "distance",
        TARGET,
        "prediction",
        "absolute_error",
        "prediction_ratio",
        "robust_z",
    ]

    print(
        top_residuals[
            display_columns
        ]
        .head(15)
        .round(2)
        .to_string(index=False)
    )

    print()
    print("Saved:")
    print(
        "  reports/oof_residual_analysis.csv"
    )
    print(
        "  reports/top_50_residuals.csv"
    )
    print(
        "  reports/extreme_target_anomalies.csv"
    )


if __name__ == "__main__":
    main()
