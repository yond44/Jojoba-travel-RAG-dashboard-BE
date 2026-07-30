from __future__ import annotations
from datetime import date
from typing import Literal

from fastapi import Depends, APIRouter, HTTPException
from motor.motor_asyncio import AsyncIOMotorDatabase
from pymongo.errors import PyMongoError

from src.utils.log import logger
from src.model.revenue_schemas import SuccessResponse
from src.config.database import get_db
from src.services.review.review import getSentimentSummary, getRatingTrend, getDestinationRatingSummary


router = APIRouter(prefix="/api/v1/reviews", tags=["reviews"])


@router.get("/sentiment-summary", response_model=SuccessResponse)
async def getSentimentSummaryRoute(
    start_date: date,
    end_date: date,
    db: AsyncIOMotorDatabase = Depends(get_db),
):
    if start_date > end_date:
        raise HTTPException(status_code=400, detail="start_date must be before end_date")

    try:
        data = await getSentimentSummary(db, start_date, end_date)
    except PyMongoError as e:
        logger.error(f"DB error in getSentimentSummaryRoute: {e}")
        raise HTTPException(status_code=503, detail="Database unavailable")
    except Exception as e:
        logger.exception("Unexpected error in getSentimentSummaryRoute")
        raise HTTPException(status_code=500, detail="Internal server error")

    return SuccessResponse(status=200, respond=data)


@router.get("/rating-trend", response_model=SuccessResponse)
async def getRatingTrendRoute(
    start_date: date,
    end_date: date,
    granularity: Literal["weekly", "monthly", "annually"] = "monthly",
    db: AsyncIOMotorDatabase = Depends(get_db),
):
    if start_date > end_date:
        raise HTTPException(status_code=400, detail="start_date must be before end_date")

    try:
        data = await getRatingTrend(db, start_date, end_date, granularity)
    except PyMongoError as e:
        logger.error(f"DB error in getRatingTrendRoute: {e}")
        raise HTTPException(status_code=503, detail="Database unavailable")
    except Exception as e:
        logger.exception("Unexpected error in getRatingTrendRoute")
        raise HTTPException(status_code=500, detail="Internal server error")

    return SuccessResponse(status=200, respond=data)


@router.get("/destination-rating", response_model=SuccessResponse)
async def getDestinationRatingSummaryRoute(
    start_date: date,
    end_date: date,
    limit: int = 10,
    db: AsyncIOMotorDatabase = Depends(get_db),
):
    if start_date > end_date:
        raise HTTPException(status_code=400, detail="start_date must be before end_date")

    try:
        data = await getDestinationRatingSummary(db, start_date, end_date, limit)
    except PyMongoError as e:
        logger.error(f"DB error in getDestinationRatingSummaryRoute: {e}")
        raise HTTPException(status_code=503, detail="Database unavailable")
    except Exception as e:
        logger.exception("Unexpected error in getDestinationRatingSummaryRoute")
        raise HTTPException(status_code=500, detail="Internal server error")

    return SuccessResponse(status=200, respond=data)
