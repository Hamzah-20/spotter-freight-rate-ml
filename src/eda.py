from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
REPORTS_DIR = PROJECT_ROOT / "reports"

TRAIN_PATH = DATA_DIR / "train_test.csv"
VALIDATION_PATH = DATA_DIR / "validation.csv"

TARGET = "posted_rate"


def section(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(TRAIN_PATH)
    validation = pd.read_csv(VALIDATION_PATH)

    train["date"] = pd.to_datetime(train["date"], errors="raise")
    validation["date"] = pd.to_datetime(validation["date"], errors="raise")

    lines = []

    def log(message=""):
        print(message)
        lines.append(str(message))

    section("1. DATASET OVERVIEW")
    log(f"Train shape      : {train.shape}")
    log(f"Validation shape : {validation.shape}")
    log()
    log(f"Train date range : {train['date'].min().date()} -> {train['date'].max().date()}")
    log(
        f"Validation dates : "
        f"{validation['date'].min().date()} -> "
        f"{validation['date'].max().date()}"
    )

    section("2. COLUMN CHECK")
    log(f"Train columns ({len(train.columns)}):")
    log(", ".join(train.columns))
    log()
    log(f"Validation columns ({len(validation.columns)}):")
    log(", ".join(validation.columns))

    train_features = set(train.columns) - {TARGET}
    validation_features = set(validation.columns)

    log()
    log(f"Target only in train: {TARGET in train.columns}")
    log(f"Feature schemas match: {train_features == validation_features}")

    section("3. MISSING VALUES")
    missing = pd.DataFrame(
        {
            "train_missing": train.isna().sum(),
            "train_missing_pct": train.isna().mean() * 100,
            "validation_missing": validation.isna().sum(),
            "validation_missing_pct": validation.isna().mean() * 100,
        }
    ).fillna(0)

    missing = missing[
        (missing["train_missing"] > 0)
        | (missing["validation_missing"] > 0)
    ]

    if missing.empty:
        log("No missing values found.")
    else:
        log(missing.to_string())

    missing.to_csv(REPORTS_DIR / "missing_values.csv")

    section("4. IDENTIFIER QUALITY")
    log(f"Duplicate train load_id      : {train['load_id'].duplicated().sum()}")
    log(
        f"Duplicate validation load_id : "
        f"{validation['load_id'].duplicated().sum()}"
    )
    log(f"Missing train load_id        : {train['load_id'].isna().sum()}")
    log(
        f"Missing validation load_id   : "
        f"{validation['load_id'].isna().sum()}"
    )

    section("5. NUMERIC DATA QUALITY")

    log(f"Negative train weights       : {(train['weight'] < 0).sum()}")
    log(
        f"Negative validation weights  : "
        f"{(validation['weight'] < 0).sum()}"
    )
    log(f"Zero train weights           : {(train['weight'] == 0).sum()}")
    log(
        f"Zero validation weights      : "
        f"{(validation['weight'] == 0).sum()}"
    )

    numeric_columns = [
        "distance",
        "weight",
        "market_index",
        "quote_signal",
        TARGET,
    ]

    numeric_summary = train[numeric_columns].describe(
        percentiles=[0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]
    ).T

    log()
    log(numeric_summary.to_string())
    numeric_summary.to_csv(REPORTS_DIR / "numeric_summary.csv")

    section("6. TARGET DISTRIBUTION")
    target = train[TARGET]

    log(f"Target minimum : ${target.min():,.2f}")
    log(f"Target maximum : ${target.max():,.2f}")
    log(f"Target mean    : ${target.mean():,.2f}")
    log(f"Target median  : ${target.median():,.2f}")
    log(f"Target P01     : ${target.quantile(0.01):,.2f}")
    log(f"Target P99     : ${target.quantile(0.99):,.2f}")

    section("7. CATEGORICAL CARDINALITY")

    categorical_columns = ["pickup", "delivery", "equipment"]

    for column in categorical_columns:
        log(
            f"{column:12s} | "
            f"train unique={train[column].nunique():3d} | "
            f"validation unique={validation[column].nunique():3d}"
        )

    routes = train["pickup"].astype(str) + " -> " + train["delivery"].astype(str)
    log(f"Unique train routes: {routes.nunique()}")

    section("8. UNSEEN CATEGORIES")

    unseen_results = []

    for column in ["pickup", "delivery", "equipment"]:
        train_values = set(train[column].dropna().astype(str))
        validation_values = set(validation[column].dropna().astype(str))

        unseen = sorted(validation_values - train_values)

        log(f"{column}: {len(unseen)} unseen categories")

        if unseen:
            log("  " + ", ".join(unseen))

        for value in unseen:
            unseen_results.append(
                {
                    "column": column,
                    "unseen_value": value,
                }
            )

    pd.DataFrame(
        unseen_results,
        columns=["column", "unseen_value"],
    ).to_csv(
        REPORTS_DIR / "unseen_categories.csv",
        index=False,
    )

    section("9. MONTHLY TARGET BEHAVIOR")

    train["month"] = train["date"].dt.to_period("M").astype(str)

    monthly_summary = (
        train.groupby("month")[TARGET]
        .agg(["count", "mean", "median", "std"])
        .reset_index()
    )

    log(monthly_summary.to_string(index=False))

    monthly_summary.to_csv(
        REPORTS_DIR / "monthly_target_summary.csv",
        index=False,
    )

    section("10. QUOTE SIGNAL TEMPORAL STABILITY")

    monthly_correlations = []

    for month, group in train.groupby("month"):
        correlation = group["quote_signal"].corr(group[TARGET])

        monthly_correlations.append(
            {
                "month": month,
                "quote_signal_target_correlation": correlation,
            }
        )

        log(f"{month}: {correlation:.4f}")

    pd.DataFrame(monthly_correlations).to_csv(
        REPORTS_DIR / "quote_signal_monthly_correlation.csv",
        index=False,
    )

    section("11. EQUIPMENT DISTRIBUTION")

    equipment_summary = (
        train["equipment"]
        .value_counts(dropna=False)
        .rename_axis("equipment")
        .reset_index(name="count")
    )

    equipment_summary["percentage"] = (
        equipment_summary["count"] / len(train) * 100
    )

    log(equipment_summary.to_string(index=False))

    equipment_summary.to_csv(
        REPORTS_DIR / "equipment_summary.csv",
        index=False,
    )

    section("12. AUDIT COMPLETE")

    log("Created:")
    log("  reports/eda_summary.txt")
    log("  reports/missing_values.csv")
    log("  reports/numeric_summary.csv")
    log("  reports/unseen_categories.csv")
    log("  reports/monthly_target_summary.csv")
    log("  reports/quote_signal_monthly_correlation.csv")
    log("  reports/equipment_summary.csv")

    with open(
        REPORTS_DIR / "eda_summary.txt",
        "w",
        encoding="utf-8",
    ) as file:
        file.write("\n".join(lines))


if __name__ == "__main__":
    main()
