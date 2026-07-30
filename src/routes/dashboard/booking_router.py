from __future__ import annotations
from datetime import date

from fastapi import Depends, APIRouter, HTTPException
from motor.motor_asyncio import AsyncIOMotorDatabase
from pymongo.errors import PyMongoError

from src.utils.log import logger
from src.model.revenue_schemas import SuccessResponse
from src.config.database import get_db
from src.services.booking.booking import (
    getBookingStatusSummary,
    getBookingVolume,
    getLeadTimeStats,
    getCancellationRate,
    getPaymentGatewayPerformance,
)


router = APIRouter(prefix="/api/v1/bookings", tags=["bookings"])


@router.get("/status-summary", response_model=SuccessResponse)
async def getBookingStatusSummaryRoute(
    start_date: date,
    end_date: date,
    db: AsyncIOMotorDatabase = Depends(get_db),
):
    if start_date > end_date:
        raise HTTPException(status_code=400, detail="start_date must be before end_date")

    try:
        data = await getBookingStatusSummary(db, start_date, end_date)
    except PyMongoError as e:
        logger.error(f"DB error in getBookingStatusSummaryRoute: {e}")
        raise HTTPException(status_code=503, detail="Database unavailable")
    except Exception as e:
        logger.exception("Unexpected error in getBookingStatusSummaryRoute")
        raise HTTPException(status_code=500, detail="Internal server error")

    return SuccessResponse(status=200, respond=data)


@router.get("/volume", response_model=SuccessResponse)
async def getBookingVolumeRoute(
    start_date: date,
    end_date: date,
    db: AsyncIOMotorDatabase = Depends(get_db),
):
    if start_date > end_date:
        raise HTTPException(status_code=400, detail="start_date must be before end_date")

    try:
        data = await getBookingVolume(db, start_date, end_date)
    except PyMongoError as e:
        logger.error(f"DB error in getBookingVolumeRoute: {e}")
        raise HTTPException(status_code=503, detail="Database unavailable")
    except Exception as e:
        logger.exception("Unexpected error in getBookingVolumeRoute")
        raise HTTPException(status_code=500, detail="Internal server error")

    return SuccessResponse(status=200, respond=data)


@router.get("/lead-time", response_model=SuccessResponse)
async def getLeadTimeStatsRoute(
    start_date: date,
    end_date: date,
    db: AsyncIOMotorDatabase = Depends(get_db),
):
    if start_date > end_date:
        raise HTTPException(status_code=400, detail="start_date must be before end_date")

    try:
        data = await getLeadTimeStats(db, start_date, end_date)
    except PyMongoError as e:
        logger.error(f"DB error in getLeadTimeStatsRoute: {e}")
        raise HTTPException(status_code=503, detail="Database unavailable")
    except Exception as e:
        logger.exception("Unexpected error in getLeadTimeStatsRoute")
        raise HTTPException(status_code=500, detail="Internal server error")

    return SuccessResponse(status=200, respond=data)


@router.get("/cancellation-rate", response_model=SuccessResponse)
async def getCancellationRateRoute(
    start_date: date,
    end_date: date,
    db: AsyncIOMotorDatabase = Depends(get_db),
):
    if start_date > end_date:
        raise HTTPException(status_code=400, detail="start_date must be before end_date")

    try:
        data = await getCancellationRate(db, start_date, end_date)
    except PyMongoError as e:
        logger.error(f"DB error in getCancellationRateRoute: {e}")
        raise HTTPException(status_code=503, detail="Database unavailable")
    except Exception as e:
        logger.exception("Unexpected error in getCancellationRateRoute")
        raise HTTPException(status_code=500, detail="Internal server error")

    return SuccessResponse(status=200, respond=data)


@router.get("/payment-gateway", response_model=SuccessResponse)
async def getPaymentGatewayPerformanceRoute(
    start_date: date,
    end_date: date,
    db: AsyncIOMotorDatabase = Depends(get_db),
):
    if start_date > end_date:
        raise HTTPException(status_code=400, detail="start_date must be before end_date")

    try:
        data = await getPaymentGatewayPerformance(db, start_date, end_date)
    except PyMongoError as e:
        logger.error(f"DB error in getPaymentGatewayPerformanceRoute: {e}")
        raise HTTPException(status_code=503, detail="Database unavailable")
    except Exception as e:
        logger.exception("Unexpected error in getPaymentGatewayPerformanceRoute")
        raise HTTPException(status_code=500, detail="Internal server error")

    return SuccessResponse(status=200, respond=data)
