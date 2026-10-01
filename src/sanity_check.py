from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

TRAIN_PATH = ROOT / "data" / "train_test.csv"
VALIDATION_PATH = ROOT / "data" / "validation.csv"
PREDICTIONS_PATH = ROOT / "validation_predictions.csv"
DECEMBER_PATH = ROOT / "data" / "december_chart_inputs.csv"

TRANSFORMED_IMPORTANCE_PATH = (
    ROOT / "artifacts" / "transformed_feature_importance.csv"
)

RAW_IMPORTANCE_PATH = (
    ROOT / "artifacts" / "raw_feature_importance.csv"
)


def section(title: str) -> None:
    print()
    print("=" * 95)
    print(title)
    print("=" * 95)


def main() -> None:

    train = pd.read_csv(TRAIN_PATH)
    validation = pd.read_csv(VALIDATION_PATH)
    predictions = pd.read_csv(PREDICTIONS_PATH)
    december = pd.read_csv(DECEMBER_PATH)

    transformed_importance = pd.read_csv(
        TRANSFORMED_IMPORTANCE_PATH
    )

    raw_importance = pd.read_csv(
        RAW_IMPORTANCE_PATH
    )

    merged = validation.merge(
        predictions,
        on="load_id",
        validate="one_to_one",
    )

    # ------------------------------------------------------------------
    section("1. FINAL PREDICTION DISTRIBUTION")

    train_rate = train["posted_rate"]
    predicted_rate = merged["predicted_rate"]

    summary = pd.DataFrame(
        {
            "train_actual": [
                train_rate.min(),
                train_rate.quantile(0.01),
                train_rate.quantile(0.05),
                train_rate.median(),
                train_rate.mean(),
                train_rate.quantile(0.95),
                train_rate.quantile(0.99),
                train_rate.max(),
            ],
            "final_prediction": [
                predicted_rate.min(),
                predicted_rate.quantile(0.01),
                predicted_rate.quantile(0.05),
                predicted_rate.median(),
                predicted_rate.mean(),
                predicted_rate.quantile(0.95),
                predicted_rate.quantile(0.99),
                predicted_rate.max(),
            ],
        },
        index=[
            "min",
            "p01",
            "p05",
            "median",
            "mean",
            "p95",
            "p99",
            "max",
        ],
    )

    print(
        summary.round(2).to_string()
    )

    # ------------------------------------------------------------------
    section("2. PREDICTED RATE PER MILE")

    merged["predicted_rpm"] = (
        merged["predicted_rate"]
        / merged["distance"]
    )

    train["actual_rpm"] = (
        train["posted_rate"]
        / train["distance"]
    )

    rpm_summary = pd.DataFrame(
        {
            "train_actual_rpm": [
                train["actual_rpm"].quantile(0.01),
                train["actual_rpm"].median(),
                train["actual_rpm"].mean(),
                train["actual_rpm"].quantile(0.99),
            ],
            "validation_predicted_rpm": [
                merged["predicted_rpm"].quantile(0.01),
                merged["predicted_rpm"].median(),
                merged["predicted_rpm"].mean(),
                merged["predicted_rpm"].quantile(0.99),
            ],
        },
        index=[
            "p01",
            "median",
            "mean",
            "p99",
        ],
    )

    print(
        rpm_summary.round(3).to_string()
    )

    # ------------------------------------------------------------------
    section("3. EXTREME FINAL PREDICTIONS")

    columns = [
        "load_id",
        "pickup",
        "delivery",
        "distance",
        "equipment",
        "weight",
        "date",
        "predicted_rate",
        "predicted_rpm",
    ]

    print("Lowest 10 predictions:")
    print(
        merged
        .nsmallest(10, "predicted_rate")[columns]
        .round(2)
        .to_string(index=False)
    )

    print()
    print("Highest 10 predictions:")
    print(
        merged
        .nlargest(10, "predicted_rate")[columns]
        .round(2)
        .to_string(index=False)
    )

    # ------------------------------------------------------------------
    section("4. LEXINGTON -> FORT WAYNE HISTORICAL CHECK")

    exact_lane = train[
        (train["pickup"] == "Lexington")
        & (train["delivery"] == "Fort Wayne")
    ].copy()

    exact_lane = exact_lane[
        exact_lane["equipment"] == "Dry Van"
    ]

    print(
        f"Exact historical lane rows: "
        f"{len(exact_lane):,}"
    )

    if len(exact_lane) > 0:

        exact_lane["rpm"] = (
            exact_lane["posted_rate"]
            / exact_lane["distance"]
        )

        print(
            exact_lane[
                [
                    "date",
                    "distance",
                    "weight",
                    "posted_rate",
                    "rpm",
                ]
            ]
            .sort_values("date")
            .tail(20)
            .round(2)
            .to_string(index=False)
        )

        print()
        print(
            f"Historical lane median rate: "
            f"${exact_lane['posted_rate'].median():,.2f}"
        )

        print(
            f"Historical lane median RPM : "
            f"${exact_lane['rpm'].median():,.3f}"
        )

    # ------------------------------------------------------------------
    section("5. COMPARABLE SHORT-HAUL DRY VAN LOADS")

    comparable = train[
        (train["equipment"] == "Dry Van")
        & (train["distance"].between(300, 420))
        & (train["weight"].abs().between(28000, 36000))
    ].copy()

    print(
        f"Comparable historical rows: "
        f"{len(comparable):,}"
    )

    if len(comparable) > 0:

        comparable["rpm"] = (
            comparable["posted_rate"]
            / comparable["distance"]
        )

        print(
            comparable[
                ["posted_rate", "rpm"]
            ]
            .describe(
                percentiles=[
                    0.05,
                    0.25,
                    0.50,
                    0.75,
                    0.95,
                ]
            )
            .round(2)
            .to_string()
        )

    # ------------------------------------------------------------------
    section("6. DECEMBER FIXED-SCENARIO CHECK")

    print(
        december[
            ["date", "predicted_rate"]
        ]
        .round(2)
        .to_string(index=False)
    )

    print()
    print(
        f"December mean   : "
        f"${december['predicted_rate'].mean():,.2f}"
    )

    print(
        f"December median : "
        f"${december['predicted_rate'].median():,.2f}"
    )

    print(
        f"December min    : "
        f"${december['predicted_rate'].min():,.2f}"
    )

    print(
        f"December max    : "
        f"${december['predicted_rate'].max():,.2f}"
    )

    # ------------------------------------------------------------------
    section("7. TOP TRANSFORMED-MODEL FEATURES")

    print(
        transformed_importance
        .head(15)
        .round(3)
        .to_string(index=False)
    )

    # ------------------------------------------------------------------
    section("8. TOP RAW-MODEL FEATURES")

    print(
        raw_importance
        .head(15)
        .round(3)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
