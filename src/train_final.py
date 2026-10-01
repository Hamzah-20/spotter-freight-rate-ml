from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error

from features import CATEGORICAL_FEATURES, prepare_features


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

TRAIN_PATH = DATA_DIR / "train_test.csv"
VALIDATION_PATH = DATA_DIR / "validation.csv"
TEMPLATE_PATH = DATA_DIR / "validation_predictions_template.csv"
DECEMBER_PATH = DATA_DIR / "december_chart_inputs.csv"

OUTPUT_PATH = PROJECT_ROOT / "validation_predictions.csv"

TARGET = "posted_rate"

TRANSFORMED_WEIGHT = 0.75
RAW_WEIGHT = 0.25

RANDOM_SEED = 42


TRANSFORMED_PARAMS = {
    "depth": 7,
    "learning_rate": 0.05,
    "l2_leaf_reg": 10.0,
    "random_strength": 0.5,
}

RAW_PARAMS = {
    "depth": 8,
    "learning_rate": 0.04,
    "l2_leaf_reg": 5.0,
    "random_strength": 1.0,
}


def categorical_columns(
    frame: pd.DataFrame,
) -> list[str]:

    return [
        column
        for column in CATEGORICAL_FEATURES
        if column in frame.columns
    ]


def transformed_target(
    frame: pd.DataFrame,
) -> np.ndarray:

    rate_per_mile = (
        frame[TARGET].to_numpy(dtype=float)
        / frame["distance"].to_numpy(dtype=float)
    )

    return np.log1p(rate_per_mile)


def inverse_transformed_target(
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

    return np.maximum(
        rate,
        1.0,
    )


def build_transformed_model(
    iterations: int,
) -> CatBoostRegressor:

    return CatBoostRegressor(
        iterations=iterations,
        loss_function="MAE",
        eval_metric="MAE",
        random_seed=RANDOM_SEED,
        verbose=False,
        allow_writing_files=False,
        thread_count=-1,
        **TRANSFORMED_PARAMS,
    )


def build_raw_model(
    iterations: int,
) -> CatBoostRegressor:

    return CatBoostRegressor(
        iterations=iterations,
        loss_function="MAE",
        eval_metric="MAE",
        random_seed=RANDOM_SEED,
        verbose=False,
        allow_writing_files=False,
        thread_count=-1,
        **RAW_PARAMS,
    )


def calibrate_iterations(
    train: pd.DataFrame,
) -> tuple[int, int, dict]:

    calibration_train = train[
        train["date"] < pd.Timestamp("2025-10-01")
    ].copy()

    calibration_valid = train[
        (
            train["date"]
            >= pd.Timestamp("2025-10-01")
        )
        & (
            train["date"]
            <= pd.Timestamp("2025-10-31")
        )
    ].copy()

    print(
        f"Calibration train rows : "
        f"{len(calibration_train):,}"
    )

    print(
        f"Calibration valid rows : "
        f"{len(calibration_valid):,}"
    )

    X_train = prepare_features(
        calibration_train,
        include_quote_signal=False,
    )

    X_valid = prepare_features(
        calibration_valid,
        include_quote_signal=False,
    )

    cats = categorical_columns(
        X_train
    )

    #  ---------------------------------- Transformed model calibration  ----------------------------------

    print()
    print(
        "Calibrating transformed model..."
    )

    transformed_model = (
        build_transformed_model(
            iterations=1800,
        )
    )

    transformed_model.fit(
        X_train,
        transformed_target(
            calibration_train
        ),
        cat_features=cats,
        eval_set=(
            X_valid,
            transformed_target(
                calibration_valid
            ),
        ),
        early_stopping_rounds=180,
        use_best_model=True,
        verbose=False,
    )

    transformed_iterations = (
        transformed_model.get_best_iteration()
        + 1
    )

    transformed_predictions = (
        inverse_transformed_target(
            transformed_model.predict(
                X_valid
            ),
            calibration_valid,
        )
    )

    #  ---------------------------------- Raw-rate model calibration  ----------------------------------

    print(
        "Calibrating raw-rate model..."
    )

    raw_model = build_raw_model(
        iterations=1600,
    )

    raw_model.fit(
        X_train,
        calibration_train[
            TARGET
        ].to_numpy(dtype=float),
        cat_features=cats,
        eval_set=(
            X_valid,
            calibration_valid[
                TARGET
            ].to_numpy(dtype=float),
        ),
        early_stopping_rounds=180,
        use_best_model=True,
        verbose=False,
    )

    raw_iterations = (
        raw_model.get_best_iteration()
        + 1
    )

    raw_predictions = np.maximum(
        raw_model.predict(X_valid),
        1.0,
    )

    blended_predictions = (
        TRANSFORMED_WEIGHT
        * transformed_predictions
        + RAW_WEIGHT
        * raw_predictions
    )

    actual = calibration_valid[
        TARGET
    ].to_numpy(dtype=float)

    transformed_mae = mean_absolute_error(
        actual,
        transformed_predictions,
    )

    raw_mae = mean_absolute_error(
        actual,
        raw_predictions,
    )

    blend_mae = mean_absolute_error(
        actual,
        blended_predictions,
    )

    calibration_metrics = {
        "transformed_mae": float(
            transformed_mae
        ),
        "raw_mae": float(
            raw_mae
        ),
        "blend_mae": float(
            blend_mae
        ),
    }

    return (
        max(transformed_iterations, 50),
        max(raw_iterations, 50),
        calibration_metrics,
    )


def city_coordinate_lookup(
    train: pd.DataFrame,
) -> pd.DataFrame:

    pickup = pd.DataFrame(
        {
            "city": train["pickup"],
            "lat": train["pickup_lat"],
            "lon": train["pickup_lon"],
        }
    )

    delivery = pd.DataFrame(
        {
            "city": train["delivery"],
            "lat": train["delivery_lat"],
            "lon": train["delivery_lon"],
        }
    )

    coordinates = pd.concat(
        [pickup, delivery],
        ignore_index=True,
    )

    return (
        coordinates
        .groupby("city")[
            ["lat", "lon"]
        ]
        .median()
    )


def prepare_december_frame(
    december: pd.DataFrame,
    train: pd.DataFrame,
) -> pd.DataFrame:

    frame = december.drop(
        columns=["predicted_rate"],
        errors="ignore",
    ).copy()

    coordinates = city_coordinate_lookup(
        train
    )

    frame["pickup_lat"] = (
        frame["pickup"]
        .map(coordinates["lat"])
    )

    frame["pickup_lon"] = (
        frame["pickup"]
        .map(coordinates["lon"])
    )

    frame["delivery_lat"] = (
        frame["delivery"]
        .map(coordinates["lat"])
    )

    frame["delivery_lon"] = (
        frame["delivery"]
        .map(coordinates["lon"])
    )

    frame["market_index"] = np.nan

    frame["quote_signal"] = np.nan

    return frame


def save_feature_importance(
    model: CatBoostRegressor,
    feature_names: list[str],
    path: Path,
) -> None:

    importance = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": (
                model.get_feature_importance()
            ),
        }
    ).sort_values(
        "importance",
        ascending=False,
    )

    importance.to_csv(
        path,
        index=False,
    )


def main() -> None:

    ARTIFACTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 90)
    print("FINAL MODEL TRAINING")
    print("=" * 90)

    train = pd.read_csv(
        TRAIN_PATH
    )

    validation = pd.read_csv(
        VALIDATION_PATH
    )

    template = pd.read_csv(
        TEMPLATE_PATH
    )

    december = pd.read_csv(
        DECEMBER_PATH
    )

    train["date"] = pd.to_datetime(
        train["date"],
        errors="raise",
    )

    validation["date"] = pd.to_datetime(
        validation["date"],
        errors="raise",
    )

    december["date"] = pd.to_datetime(
        december["date"],
        errors="raise",
    )

    print(
        f"Training rows   : {len(train):,}"
    )

    print(
        f"Validation rows : {len(validation):,}"
    )

    #  ---------------------------------- 1. Calibrate final training iterations  ----------------------------------

    print()
    print("=" * 90)
    print("1. ITERATION CALIBRATION")
    print("=" * 90)

    (
        transformed_iterations,
        raw_iterations,
        calibration_metrics,
    ) = calibrate_iterations(train)

    print()
    print(
        f"Transformed iterations : "
        f"{transformed_iterations}"
    )

    print(
        f"Raw-rate iterations    : "
        f"{raw_iterations}"
    )

    print(
        f"October transformed MAE: "
        f"${calibration_metrics['transformed_mae']:,.2f}"
    )

    print(
        f"October raw MAE        : "
        f"${calibration_metrics['raw_mae']:,.2f}"
    )

    print(
        f"October blend MAE      : "
        f"${calibration_metrics['blend_mae']:,.2f}"
    )

    #  ---------------------------------- 2. Prepare all development and validation features  ----------------------------------
    print()
    print("=" * 90)
    print("2. FULL-DATA TRAINING")
    print("=" * 90)

    X_full = prepare_features(
        train,
        include_quote_signal=False,
    )

    X_validation = prepare_features(
        validation,
        include_quote_signal=False,
    )

    X_validation = X_validation.reindex(
        columns=X_full.columns
    )

    cats = categorical_columns(
        X_full
    )

    #  ---------------------------------- 3. Fit transformed model on ALL Jan-Oct rows  ----------------------------------

    print(
        "Training final transformed model..."
    )

    transformed_model = (
        build_transformed_model(
            transformed_iterations
        )
    )

    transformed_model.fit(
        X_full,
        transformed_target(train),
        cat_features=cats,
        verbose=False,
    )

    transformed_validation = (
        inverse_transformed_target(
            transformed_model.predict(
                X_validation
            ),
            validation,
        )
    )

    #  ---------------------------------- 4. Fit raw-rate model on ALL Jan-Oct rows  ----------------------------------

    print(
        "Training final raw-rate model..."
    )

    raw_model = build_raw_model(
        raw_iterations
    )

    raw_model.fit(
        X_full,
        train[TARGET].to_numpy(
            dtype=float
        ),
        cat_features=cats,
        verbose=False,
    )

    raw_validation = np.maximum(
        raw_model.predict(
            X_validation
        ),
        1.0,
    )

    #  ---------------------------------- 5. Final ensemble  ----------------------------------

    final_validation = (
        TRANSFORMED_WEIGHT
        * transformed_validation
        + RAW_WEIGHT
        * raw_validation
    )

    final_validation = np.maximum(
        final_validation,
        1.0,
    )

    #  ---------------------------------- 6. Fill official prediction template  ----------------------------------

    print()
    print("=" * 90)
    print("3. VALIDATION PREDICTIONS")
    print("=" * 90)

    if len(validation) != 12_000:
        raise ValueError(
            "validation.csv must contain "
            "exactly 12,000 rows."
        )

    if len(template) != 12_000:
        raise ValueError(
            "Prediction template must contain "
            "exactly 12,000 rows."
        )

    if set(template["load_id"]) != set(
        validation["load_id"]
    ):
        raise ValueError(
            "Template IDs do not match "
            "validation IDs."
        )

    prediction_lookup = pd.Series(
        final_validation,
        index=validation["load_id"],
    )

    output = template[
        ["load_id"]
    ].copy()

    output["predicted_rate"] = (
        output["load_id"]
        .map(prediction_lookup)
    )

    if output[
        "predicted_rate"
    ].isna().any():
        raise ValueError(
            "Some validation predictions "
            "are missing."
        )

    if (
        output["predicted_rate"] <= 0
    ).any():
        raise ValueError(
            "All predictions must be positive."
        )

    output.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print(
        f"Saved: {OUTPUT_PATH}"
    )

    print(
        f"Prediction range: "
        f"${output['predicted_rate'].min():,.2f} "
        f"-> "
        f"${output['predicted_rate'].max():,.2f}"
    )

    print(
        f"Prediction mean : "
        f"${output['predicted_rate'].mean():,.2f}"
    )

    print(
        f"Prediction median: "
        f"${output['predicted_rate'].median():,.2f}"
    )

    #  ---------------------------------- 7. December fixed-scenario predictions  ----------------------------------

    print()
    print("=" * 90)
    print("4. DECEMBER CHART PREDICTIONS")
    print("=" * 90)

    december_model_frame = (
        prepare_december_frame(
            december,
            train,
        )
    )

    X_december = prepare_features(
        december_model_frame,
        include_quote_signal=False,
    )

    X_december = X_december.reindex(
        columns=X_full.columns
    )

    transformed_december = (
        inverse_transformed_target(
            transformed_model.predict(
                X_december
            ),
            december_model_frame,
        )
    )

    raw_december = np.maximum(
        raw_model.predict(
            X_december
        ),
        1.0,
    )

    final_december = (
        TRANSFORMED_WEIGHT
        * transformed_december
        + RAW_WEIGHT
        * raw_december
    )

    final_december = np.maximum(
        final_december,
        1.0,
    )

    december_output = december.copy()

    december_output[
        "predicted_rate"
    ] = final_december

    # Keep a copy of the original blank file.
    december_backup = (
        DATA_DIR
        / "december_chart_inputs_original.csv"
    )

    if not december_backup.exists():
        december.to_csv(
            december_backup,
            index=False,
        )

    december_output.to_csv(
        DECEMBER_PATH,
        index=False,
        date_format="%Y-%m-%d",
    )

    print(
        f"Completed: {DECEMBER_PATH}"
    )

    print(
        f"December prediction range: "
        f"${final_december.min():,.2f} "
        f"-> "
        f"${final_december.max():,.2f}"
    )

    #  ---------------------------------- 8. Save models and metadata  ----------------------------------

    print()
    print("=" * 90)
    print("5. SAVING ARTIFACTS")
    print("=" * 90)

    transformed_model_path = (
        ARTIFACTS_DIR
        / "final_transformed_model.cbm"
    )

    raw_model_path = (
        ARTIFACTS_DIR
        / "final_raw_model.cbm"
    )

    transformed_model.save_model(
        str(transformed_model_path)
    )

    raw_model.save_model(
        str(raw_model_path)
    )

    save_feature_importance(
        transformed_model,
        list(X_full.columns),
        ARTIFACTS_DIR
        / "transformed_feature_importance.csv",
    )

    save_feature_importance(
        raw_model,
        list(X_full.columns),
        ARTIFACTS_DIR
        / "raw_feature_importance.csv",
    )

    metadata = {
        "training_rows": int(
            len(train)
        ),
        "training_start": str(
            train["date"].min().date()
        ),
        "training_end": str(
            train["date"].max().date()
        ),
        "validation_rows": int(
            len(validation)
        ),
        "target_strategy": (
            "75% log1p(rate_per_mile) "
            "+ 25% raw_rate"
        ),
        "quote_signal_used": False,
        "transformed_weight": (
            TRANSFORMED_WEIGHT
        ),
        "raw_weight": RAW_WEIGHT,
        "transformed_iterations": int(
            transformed_iterations
        ),
        "raw_iterations": int(
            raw_iterations
        ),
        "calibration": (
            calibration_metrics
        ),
        "random_seed": RANDOM_SEED,
    }

    metadata_path = (
        ARTIFACTS_DIR
        / "training_metadata.json"
    )

    with open(
        metadata_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            indent=2,
        )

    print(
        f"Saved: {transformed_model_path}"
    )

    print(
        f"Saved: {raw_model_path}"
    )

    print(
        f"Saved: {metadata_path}"
    )

    print()
    print("=" * 90)
    print("FINAL TRAINING COMPLETE")
    print("=" * 90)


if __name__ == "__main__":
    main()
