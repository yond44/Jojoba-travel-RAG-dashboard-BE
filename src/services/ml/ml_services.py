from __future__ import annotations

from datetime  import datetime, timezone
from typing import Literal

import numpy as np
import pandas as pd
from motor.motor_asyncio import AsyncIOMotorDatabase

 
from src.model.ml_schemas import (
    ChurnPrediction,
    ChurnPredictionResponse,
    ForecastPeriod,
    ForecastResponse,
    SegmentResponse,
)
from src.services.ml.artifact_loader import get_artifacts
from src.services.ml.feature_builder import align_features, build_churn_features
from src.utils.log import logger


class CustomerHistoryNotFoundError(Exception):
    """Tidak satu pun customer yang diminta punya riwayat booking valid."""
    
class ForecastUnavailableError(Exception):
    """Collection ml_forecast_results kosong — batch pipeline belum jalan."""
    
    
def _risk_bucket(p: float) -> str:
    if p >= 0.75:
        return "Very High"
    if p >= 0.50:
        return "High"
    if p >= 0.25:
        return "Medium"
    return "Low"


async def predict_churn(
    db: AsyncIOMotorDatabase,
    customer_ids: list[str],
) -> ChurnPredictionResponse:
    artifacts = get_artifacts()
    
    feat = await build_churn_features(db, customer_ids=customer_ids)
    
    if feat.empty:
        raise CustomerHistoryNotFoundError(
            f"Tidak ada riwayat booking valid untuk {len(customer_ids)} "
            f"customer yang diminta."
        )
        
    not_found = sorted(set(customer_ids) - set(feat.index))
    
    tracks = feat["track"]
    
    X =align_features(feat, artifacts.churn_features)
    probas = artifacts.churn_model.predict_proba(X)[:, 1]
    
    scored_at = datetime.now(timezone.utc)
    predictions = [
        ChurnPrediction(
            customer_id=cid,
            churn_proba=round(float(p), 4),
            risk_bucket = _risk_bucket(float(p)),
            track=tracks.loc[cid],
            scored_at=scored_at
        )
        for cid, p in zip(feat.index, probas)
    ]
    
    logger.info("Churn scored: %d ok, %d not found",
                len(predictions), len(not_found))
    
    return ChurnPredictionResponse(
        predictions=predictions,
        not_found_customer_ids=not_found,
        model_name=type(artifacts.churn_model).__name__
    )
    
    
async def get_forecast(
    db: AsyncIOMotorDatabase,
    horizon: Literal["weekly", "monthly", "yearly"],
)-> ForecastResponse:
    source_horizon = "monthly" if horizon == "yearly" else horizon
 
    docs = await (
        db["ml_forecast_results"]
        .find({"horizon": source_horizon})
        .sort("period", 1)
        .to_list(None)
    )
    if not docs:
        raise ForecastUnavailableError(
            "ml_forecast_results kosong — jalankan batch pipeline dulu.")
 
    if horizon == "yearly":
        docs = docs[:12]
 
    periods = [
        ForecastPeriod(period=d["period"], forecast_idr=d["forecast_idr"])
        for d in docs
    ]
    return ForecastResponse(
        horizon=horizon,
        periods=periods,
        total_idr=float(sum(p.forecast_idr for p in periods)),
        model_mape_pct=float(docs[0]["model_mape_pct"]),
        generated_at=datetime.fromisoformat(docs[0]["generated_at"]),
    )
 
 
 
 
async def get_customer_segment(
    db: AsyncIOMotorDatabase,
    customer_id: str,
) -> SegmentResponse:

    artifacts = get_artifacts()
 
    feat = await build_churn_features(db, customer_ids=[customer_id])
    if feat.empty:
        raise CustomerHistoryNotFoundError(
            f"Customer {customer_id} tidak punya riwayat booking valid.")
 
    row = feat.iloc[0]
 
    feature_order = artifacts.kmeans["features"]
    rfm = pd.DataFrame(
        [[row[c] for c in feature_order]], columns=feature_order)
 
    X_scaled = artifacts.kmeans["scaler"].transform(np.log1p(rfm))
    cluster = int(artifacts.kmeans["kmeans"].predict(X_scaled)[0])
 
    return SegmentResponse(
        customer_id=str(feat.index[0]),
        cluster=cluster,
        recency_days=int(row["recency_days"]),
        frequency=int(row["frequency"]),
        monetary_total=float(row["monetary_total"]),
        scored_at=datetime.now(timezone.utc),
    )