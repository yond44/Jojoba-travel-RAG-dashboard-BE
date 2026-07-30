from __future__ import annotations
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from bson import ObjectId
from motor.motor_asyncio import AsyncIOMotorDatabase

from src.utils.log import logger


VALID_STATUSES = ["Completed", "Confirmed"]
SINGLE_BOOKER_FREQUENCY = 1
CATEGORICAL_FEATURES = ["review_sentiment", "segment", "acquisition_channel"]


async def build_churn_features(
    db: AsyncIOMotorDatabase,
    customer_ids: list[str] | None = None,
    snapshot: datetime | None = None,
) -> pd.DataFrame:

    snapshot = snapshot or datetime.now(timezone.utc)

    snap_ts = pd.Timestamp(snapshot).tz_localize(None) \
        if pd.Timestamp(snapshot).tzinfo is None \
        else pd.Timestamp(snapshot).tz_convert("UTC").tz_localize(None)

    cust_filter: dict = {}
    if customer_ids:
        try:
            cust_filter = {"customer_id": {
                "$in": [ObjectId(c) for c in customer_ids]}}
        except Exception as exc:
            raise ValueError(
                f"customer_ids contains an invalid ObjectId: {exc}") from exc

    rfm_pipeline = [
        {"$match": {
            **cust_filter,
            "status": {"$in": VALID_STATUSES},
            "booking_date": {"$lte": snapshot},
        }},
        {"$group": {
            "_id": "$customer_id",
            "last_booking":  {"$max": "$booking_date"},
            "first_booking": {"$min": "$booking_date"},
            "frequency":     {"$sum": 1},
            "monetary_total": {"$sum": "$total_price_idr"},
            "monetary_avg":   {"$avg": "$total_price_idr"},
            "pax_avg":        {"$avg": "$pax_count"},
            "nights_avg":     {"$avg": "$nights"},
        }},
    ]

    rfm_rows = await db["bookings"].aggregate(rfm_pipeline).to_list(None)
    if not rfm_rows:
        logger.warning("No valid bookings found for filter %s", cust_filter)
        return pd.DataFrame()

    feat = pd.DataFrame(rfm_rows).rename(columns={"_id": "customer_id"})
    feat = feat.set_index("customer_id")

    cancel_pipeline = [
        {"$match": {**cust_filter, "booking_date": {"$lte": snapshot}}},
        {"$group": {
            "_id": "$customer_id",
            "cancel_rate": {"$avg": {
                "$cond": [{"$eq": ["$status", "Cancelled"]}, 1, 0]}},
        }},
    ]
    cancel_rows = await db["bookings"].aggregate(cancel_pipeline).to_list(None)
    cancel = pd.DataFrame(cancel_rows).rename(
        columns={"_id": "customer_id"}).set_index("customer_id")
    feat = feat.join(cancel)
    feat["cancel_rate"] = feat["cancel_rate"].fillna(0)


    review_pipeline = [
        {"$match": {**cust_filter, "review_date": {"$lte": snapshot}}},
        {"$sort": {"review_date": 1, "_id": 1}},  
        {"$group": {
            "_id": "$customer_id",
            "review_rating":    {"$last": "$rating"},
            "review_sentiment": {"$last": "$sentiment"},
            "aspect_value":     {"$avg": "$aspects.value_for_money"},
        }},
    ]
    review_rows = await db["reviews"].aggregate(review_pipeline).to_list(None)
    if review_rows:
        reviews = pd.DataFrame(review_rows).rename(
            columns={"_id": "customer_id"}).set_index("customer_id")
        feat = feat.join(reviews)
    else:
        feat[["review_rating", "review_sentiment", "aspect_value"]] = np.nan

    profile_rows = await db["customers"].find(
        {"_id": {"$in": list(feat.index)}},
        {"age": 1, "segment": 1, "acquisition_channel": 1},
    ).to_list(None)
    profile = pd.DataFrame(profile_rows).rename(
        columns={"_id": "customer_id"}).set_index("customer_id")
    feat = feat.join(profile)

    for col in ("last_booking", "first_booking"):
        feat[col] = pd.to_datetime(feat[col]).dt.tz_localize(None)

    feat["recency_days"] = (snap_ts - feat["last_booking"]).dt.days
    feat["tenure_days"] = (snap_ts - feat["first_booking"]).dt.days

    feat["avg_interval"] = np.where(
        feat["frequency"] > 1,
        (feat["last_booking"] - feat["first_booking"]).dt.days
        / (feat["frequency"] - 1),
        np.nan)

    feat["recency_ratio"] = (
        feat["recency_days"] / feat["avg_interval"].replace(0, np.nan)
    ).replace([np.inf, -np.inf], np.nan)

    feat["has_review"] = feat["review_rating"].notna().astype(int)
    feat["review_sentiment"] = feat["review_sentiment"].fillna("No Review")
    feat["track"] = np.where(
        feat["frequency"] == SINGLE_BOOKER_FREQUENCY, "single", "repeat")

    feat.index = feat.index.map(str)
    feat.index.name = "customer_id"

    logger.info("Churn features built: %d customers (snapshot=%s)",
                len(feat), snap_ts.date())
    return feat


def align_features(feat: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    encoded = pd.get_dummies(feat, columns=CATEGORICAL_FEATURES)

    unseen = [c for c in encoded.columns
              if c not in feature_columns
              and any(c.startswith(f"{cat}_") for cat in CATEGORICAL_FEATURES)]
    if unseen:
        logger.warning("Categorical values with no matching training dummy "
                       "(will be zeroed, biasing predictions): %s", unseen)

    missing = [c for c in feature_columns
               if c not in encoded.columns
               and not any(c.startswith(f"{cat}_") for cat in CATEGORICAL_FEATURES)]
    if missing:
        raise ValueError(
            f"align_features: expected feature column(s) missing from "
            f"built features (typo or upstream schema change?): {missing}")

    return encoded.reindex(columns=feature_columns, fill_value=0)
