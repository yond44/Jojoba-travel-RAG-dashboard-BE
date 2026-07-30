from __future__ import annotations

from datetime import date
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query, status
from motor.motor_asyncio import AsyncIOMotorDatabase

from src.config.database import get_db
from src.model.ml_schemas import (
    ChurnPredictionRequest,
    ChurnPredictionResponse,
    ForecastResponse,
    RevenueResponse,
    SegmentResponse,
)
from src.services.ml.ml_services import (
    CustomerHistoryNotFoundError,
    ForecastUnavailableError,
    get_customer_segment,
    get_forecast,
    predict_churn,
)
from src.services.ml.revenue_resolver import resolve_revenue

router = APIRouter(prefix="/api/v1", tags=["ml"])


@router.post("/predict/churn", response_model=ChurnPredictionResponse)
async def predict_churn_route(
    payload: ChurnPredictionRequest,
    db: AsyncIOMotorDatabase = Depends(get_db),
) -> ChurnPredictionResponse:
    try:
        return await predict_churn(db, customer_ids=payload.customer_ids)
    except ValueError as exc:  # ObjectId tidak valid dari feature_builder
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc)) from exc
    except CustomerHistoryNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc


@router.get("/revenue", response_model=RevenueResponse)
async def revenue_route(
    start: date = Query(description="Awal rentang, YYYY-MM-DD"),
    end: date = Query(description="Akhir rentang, YYYY-MM-DD (inklusif)"),
    db: AsyncIOMotorDatabase = Depends(get_db),
) -> RevenueResponse:
    try:
        result = await resolve_revenue(db, start, end)
        return RevenueResponse.model_validate(result)
    except ValueError as exc: 
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc)) from exc
    except RuntimeError as exc:  
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc)) from exc


@router.get("/forecast", response_model=ForecastResponse)
async def get_forecast_route(
    horizon: Literal["daily", "weekly", "monthly", "yearly"] = "monthly",
    db: AsyncIOMotorDatabase = Depends(get_db),
) -> ForecastResponse:
    """Legacy: forecast per-horizon mentah. Untuk pertanyaan berbasis
    tanggal, gunakan /revenue — endpoint ini dipertahankan untuk grafik
    dashboard yang memang butuh deret per-horizon apa adanya."""
    try:
        return await get_forecast(db, horizon=horizon)
    except ForecastUnavailableError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc)) from exc


@router.get("/segments/{customer_id}", response_model=SegmentResponse)
async def get_customer_segment_route(
    customer_id: str,
    db: AsyncIOMotorDatabase = Depends(get_db),
) -> SegmentResponse:
    try:
        return await get_customer_segment(db, customer_id=customer_id)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc)) from exc
    except CustomerHistoryNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
