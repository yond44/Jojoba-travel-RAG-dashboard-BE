from __future__ import annotations
from datetime import date

from fastapi import Depends, APIRouter, HTTPException, status
from motor.motor_asyncio import AsyncIOMotorDatabase
from pymongo.errors import PyMongoError
from typing import Literal

from src.utils.log import logger
from src.model.revenue_schemas import SuccessResponse


from src.config.database import get_db
from src.services.revenue.revenue import (
    getCurrentRevenue,
    getPeriodRevenue,
    getDailyRevenue,
    getRevenueByDestination,
    getRevenueByPackageType,
    getRevenueByChannel,
    getRevenueByPaymentMethod,
    getRevenueBySegment,
    getAgentPerformance,
    getCustomerTypeRevenue,
    getDiscountImpact,
    getPaymentStatusSummary,
)


router = APIRouter(prefix="/api/v1", tags=["revenue"])


@router.get("/current", response_model=SuccessResponse)
async def current_revenue(db: AsyncIOMotorDatabase = Depends(get_db)):
    try:
        data = await getCurrentRevenue(db)
    except PyMongoError as e:
        logger.error(f"DB error in current_revenue: {e}")
        raise HTTPException(status_code=503, detail="Database unavailable")
    except Exception as e:
        logger.exception("Unexpected error in current_revenue")
        raise HTTPException(status_code=500, detail="Internal server error")

    return SuccessResponse(status=200, respond=data)


@router.get("/period", response_model=SuccessResponse)
async def getPeriodRevenueRoute(
    start_date: date,
    end_date: date,
    granularity: Literal["weekly", "monthly", "annually"] = "monthly",
    db: AsyncIOMotorDatabase = Depends(get_db),
):
    if start_date > end_date:
        raise HTTPException(status_code=400, detail="start_date must be before end_date")

    try:
        data = await getPeriodRevenue(db, start_date, end_date, granularity)
    except PyMongoError as e:
        logger.error(f"DB error in getPeriodRevenueRoute: {e}")
        raise HTTPException(status_code=503, detail="Database unavailable")
    except Exception as e:
        logger.exception("Unexpected error in getPeriodRevenueRoute")
        raise HTTPException(status_code=500, detail="Internal server error")

    return SuccessResponse(status=200, respond=data)


@router.get("/daily", response_model=SuccessResponse)
async def getDailyRevenueRoute(
    start_date: date,
    end_date: date,
    db: AsyncIOMotorDatabase = Depends(get_db),
):
    if start_date > end_date:
        raise HTTPException(status_code=400, detail="start_date must be before end_date")

    try:
        data = await getDailyRevenue(db, start_date, end_date)
    except PyMongoError as e:
        logger.error(f"DB error in getDailyRevenueRoute: {e}")
        raise HTTPException(status_code=503, detail="Database unavailable")
    except Exception as e:
        logger.exception("Unexpected error in getDailyRevenueRoute")
        raise HTTPException(status_code=500, detail="Internal server error")

    return SuccessResponse(status=200, respond=data)


@router.get("/by-destination", response_model=SuccessResponse)
async def getRevenueByDestinationRoute(
    start_date: date,
    end_date: date,
    db: AsyncIOMotorDatabase = Depends(get_db),
):
    if start_date > end_date:
        raise HTTPException(status_code=400, detail="start_date must be before end_date")

    try:
        data = await getRevenueByDestination(db, start_date, end_date)
    except PyMongoError as e:
        logger.error(f"DB error in getRevenueByDestinationRoute: {e}")
        raise HTTPException(status_code=503, detail="Database unavailable")
    except Exception as e:
        logger.exception("Unexpected error in getRevenueByDestinationRoute")
        raise HTTPException(status_code=500, detail="Internal server error")

    return SuccessResponse(status=200, respond=data)


@router.get("/by-package-type", response_model=SuccessResponse)
async def getRevenueByPackageTypeRoute(
    start_date: date,
    end_date: date,
    db: AsyncIOMotorDatabase = Depends(get_db),
):
    if start_date > end_date:
        raise HTTPException(status_code=400, detail="start_date must be before end_date")

    try:
        data = await getRevenueByPackageType(db, start_date, end_date)
    except PyMongoError as e:
        logger.error(f"DB error in getRevenueByPackageTypeRoute: {e}")
        raise HTTPException(status_code=503, detail="Database unavailable")
    except Exception as e:
        logger.exception("Unexpected error in getRevenueByPackageTypeRoute")
        raise HTTPException(status_code=500, detail="Internal server error")

    return SuccessResponse(status=200, respond=data)


@router.get("/by-channel", response_model=SuccessResponse)
async def getRevenueByChannelRoute(
    start_date: date,
    end_date: date,
    db: AsyncIOMotorDatabase = Depends(get_db),
):
    if start_date > end_date:
        raise HTTPException(status_code=400, detail="start_date must be before end_date")

    try:
        data = await getRevenueByChannel(db, start_date, end_date)
    except PyMongoError as e:
        logger.error(f"DB error in getRevenueByChannelRoute: {e}")
        raise HTTPException(status_code=503, detail="Database unavailable")
    except Exception as e:
        logger.exception("Unexpected error in getRevenueByChannelRoute")
        raise HTTPException(status_code=500, detail="Internal server error")

    return SuccessResponse(status=200, respond=data)


@router.get("/by-payment-method", response_model=SuccessResponse)
async def getRevenueByPaymentMethodRoute(
    start_date: date,
    end_date: date,
    db: AsyncIOMotorDatabase = Depends(get_db),
):
    if start_date > end_date:
        raise HTTPException(status_code=400, detail="start_date must be before end_date")

    try:
        data = await getRevenueByPaymentMethod(db, start_date, end_date)
    except PyMongoError as e:
        logger.error(f"DB error in getRevenueByPaymentMethodRoute: {e}")
        raise HTTPException(status_code=503, detail="Database unavailable")
    except Exception as e:
        logger.exception("Unexpected error in getRevenueByPaymentMethodRoute")
        raise HTTPException(status_code=500, detail="Internal server error")

    return SuccessResponse(status=200, respond=data)


@router.get("/by-segment", response_model=SuccessResponse)
async def getRevenueBySegmentRoute(
    start_date: date,
    end_date: date,
    db: AsyncIOMotorDatabase = Depends(get_db),
):
    if start_date > end_date:
        raise HTTPException(status_code=400, detail="start_date must be before end_date")

    try:
        data = await getRevenueBySegment(db, start_date, end_date)
    except PyMongoError as e:
        logger.error(f"DB error in getRevenueBySegmentRoute: {e}")
        raise HTTPException(status_code=503, detail="Database unavailable")
    except Exception as e:
        logger.exception("Unexpected error in getRevenueBySegmentRoute")
        raise HTTPException(status_code=500, detail="Internal server error")

    return SuccessResponse(status=200, respond=data)


@router.get("/agent-performance", response_model=SuccessResponse)
async def getAgentPerformanceRoute(
    start_date: date,
    end_date: date,
    db: AsyncIOMotorDatabase = Depends(get_db),
):
    if start_date > end_date:
        raise HTTPException(status_code=400, detail="start_date must be before end_date")

    try:
        data = await getAgentPerformance(db, start_date, end_date)
    except PyMongoError as e:
        logger.error(f"DB error in getAgentPerformanceRoute: {e}")
        raise HTTPException(status_code=503, detail="Database unavailable")
    except Exception as e:
        logger.exception("Unexpected error in getAgentPerformanceRoute")
        raise HTTPException(status_code=500, detail="Internal server error")

    return SuccessResponse(status=200, respond=data)


@router.get("/customer-type", response_model=SuccessResponse)
async def getCustomerTypeRevenueRoute(
    start_date: date,
    end_date: date,
    db: AsyncIOMotorDatabase = Depends(get_db),
):
    if start_date > end_date:
        raise HTTPException(status_code=400, detail="start_date must be before end_date")

    try:
        data = await getCustomerTypeRevenue(db, start_date, end_date)
    except PyMongoError as e:
        logger.error(f"DB error in getCustomerTypeRevenueRoute: {e}")
        raise HTTPException(status_code=503, detail="Database unavailable")
    except Exception as e:
        logger.exception("Unexpected error in getCustomerTypeRevenueRoute")
        raise HTTPException(status_code=500, detail="Internal server error")

    return SuccessResponse(status=200, respond=data)


@router.get("/discount-impact", response_model=SuccessResponse)
async def getDiscountImpactRoute(
    start_date: date,
    end_date: date,
    db: AsyncIOMotorDatabase = Depends(get_db),
):
    if start_date > end_date:
        raise HTTPException(status_code=400, detail="start_date must be before end_date")

    try:
        data = await getDiscountImpact(db, start_date, end_date)
    except PyMongoError as e:
        logger.error(f"DB error in getDiscountImpactRoute: {e}")
        raise HTTPException(status_code=503, detail="Database unavailable")
    except Exception as e:
        logger.exception("Unexpected error in getDiscountImpactRoute")
        raise HTTPException(status_code=500, detail="Internal server error")

    return SuccessResponse(status=200, respond=data)


@router.get("/payment-status", response_model=SuccessResponse)
async def getPaymentStatusSummaryRoute(
    start_date: date,
    end_date: date,
    db: AsyncIOMotorDatabase = Depends(get_db),
):
    if start_date > end_date:
        raise HTTPException(status_code=400, detail="start_date must be before end_date")

    try:
        data = await getPaymentStatusSummary(db, start_date, end_date)
    except PyMongoError as e:
        logger.error(f"DB error in getPaymentStatusSummaryRoute: {e}")
        raise HTTPException(status_code=503, detail="Database unavailable")
    except Exception as e:
        logger.exception("Unexpected error in getPaymentStatusSummaryRoute")
        raise HTTPException(status_code=500, detail="Internal server error")

    return SuccessResponse(status=200, respond=data)