from __future__ import annotations

import numpy as np
import pandas as pd


CATEGORICAL_FEATURES = [
    "pickup",
    "delivery",
    "equipment",
    "lane",
]


def prepare_features(
    frame: pd.DataFrame,
    include_quote_signal: bool = True,
) -> pd.DataFrame:
    """Create model-ready features without modifying the original DataFrame."""

    data = frame.copy()

    data["date"] = pd.to_datetime(data["date"], errors="raise")

    # ------------------------------------------------------------------
    # Data-quality features
    # ------------------------------------------------------------------
    data["weight_missing"] = data["weight"].isna().astype(int)
    data["weight_was_negative"] = (data["weight"] < 0).fillna(False).astype(int)

    # Negative freight weights are physically invalid.
    # Preserve that information in a flag, then use the magnitude.
    data["weight"] = data["weight"].abs()

    data["market_index_missing"] = data["market_index"].isna().astype(int)

    # ------------------------------------------------------------------
    # Route / categorical features
    # ------------------------------------------------------------------
    for column in ["pickup", "delivery", "equipment"]:
        data[column] = data[column].fillna("__MISSING__").astype(str)

    data["lane"] = data["pickup"] + " -> " + data["delivery"]

    # ------------------------------------------------------------------
    # Geographic features
    # ------------------------------------------------------------------
    data["lat_delta"] = data["delivery_lat"] - data["pickup_lat"]
    data["lon_delta"] = data["delivery_lon"] - data["pickup_lon"]

    data["geo_distance_proxy"] = np.sqrt(
        data["lat_delta"] ** 2 + data["lon_delta"] ** 2
    )

    # ------------------------------------------------------------------
    # Time features
    # ------------------------------------------------------------------
    data["year"] = data["date"].dt.year
    data["month"] = data["date"].dt.month
    data["day"] = data["date"].dt.day
    data["day_of_week"] = data["date"].dt.dayofweek
    data["day_of_year"] = data["date"].dt.dayofyear
    data["week_of_year"] = data["date"].dt.isocalendar().week.astype(int)

    data["month_sin"] = np.sin(2 * np.pi * data["month"] / 12)
    data["month_cos"] = np.cos(2 * np.pi * data["month"] / 12)

    data["dow_sin"] = np.sin(2 * np.pi * data["day_of_week"] / 7)
    data["dow_cos"] = np.cos(2 * np.pi * data["day_of_week"] / 7)

    data["doy_sin"] = np.sin(2 * np.pi * data["day_of_year"] / 365.25)
    data["doy_cos"] = np.cos(2 * np.pi * data["day_of_year"] / 365.25)

    # ------------------------------------------------------------------
    # Freight-specific numeric features
    # ------------------------------------------------------------------
    data["distance_log"] = np.log1p(data["distance"].clip(lower=0))
    data["weight_log"] = np.log1p(data["weight"].clip(lower=0))

    data["weight_per_mile"] = data["weight"] / data["distance"].replace(0, np.nan)

    # Experimentally compare this unstable temporal feature.
    if not include_quote_signal:
        data = data.drop(columns=["quote_signal"])

    # IDs and raw timestamps must not become predictive shortcuts.
    data = data.drop(
        columns=["load_id", "date", "posted_rate"],
        errors="ignore",
    )

    return data
