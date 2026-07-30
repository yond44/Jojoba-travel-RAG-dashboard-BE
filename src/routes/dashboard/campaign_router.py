from __future__ import annotations
from datetime import date

from fastapi import Depends, APIRouter, HTTPException
from motor.motor_asyncio import AsyncIOMotorDatabase
from pymongo.errors import PyMongoError

from src.utils.log import logger
from src.model.revenue_schemas import SuccessResponse
from src.config.database import get_db
from src.services.campaign.campaign import (
    getCampaignPerformance,
    getCampaignTypeSummary,
    getCampaignFunnel,
)


router = APIRouter(prefix="/api/v1/campaigns", tags=["campaigns"])


@router.get("/performance", response_model=SuccessResponse)
async def getCampaignPerformanceRoute(
    start_date: date,
    end_date: date,
    limit: int = 20,
    db: AsyncIOMotorDatabase = Depends(get_db),
):
    if start_date > end_date:
        raise HTTPException(status_code=400, detail="start_date must be before end_date")

    try:
        data = await getCampaignPerformance(db, start_date, end_date, limit)
    except PyMongoError as e:
        logger.error(f"DB error in getCampaignPerformanceRoute: {e}")
        raise HTTPException(status_code=503, detail="Database unavailable")
    except Exception as e:
        logger.exception("Unexpected error in getCampaignPerformanceRoute")
        raise HTTPException(status_code=500, detail="Internal server error")

    return SuccessResponse(status=200, respond=data)


@router.get("/type-summary", response_model=SuccessResponse)
async def getCampaignTypeSummaryRoute(
    start_date: date,
    end_date: date,
    db: AsyncIOMotorDatabase = Depends(get_db),
):
    if start_date > end_date:
        raise HTTPException(status_code=400, detail="start_date must be before end_date")

    try:
        data = await getCampaignTypeSummary(db, start_date, end_date)
    except PyMongoError as e:
        logger.error(f"DB error in getCampaignTypeSummaryRoute: {e}")
        raise HTTPException(status_code=503, detail="Database unavailable")
    except Exception as e:
        logger.exception("Unexpected error in getCampaignTypeSummaryRoute")
        raise HTTPException(status_code=500, detail="Internal server error")

    return SuccessResponse(status=200, respond=data)


@router.get("/funnel", response_model=SuccessResponse)
async def getCampaignFunnelRoute(
    start_date: date,
    end_date: date,
    db: AsyncIOMotorDatabase = Depends(get_db),
):
    if start_date > end_date:
        raise HTTPException(status_code=400, detail="start_date must be before end_date")

    try:
        data = await getCampaignFunnel(db, start_date, end_date)
    except PyMongoError as e:
        logger.error(f"DB error in getCampaignFunnelRoute: {e}")
        raise HTTPException(status_code=503, detail="Database unavailable")
    except Exception as e:
        logger.exception("Unexpected error in getCampaignFunnelRoute")
        raise HTTPException(status_code=500, detail="Internal server error")

    return SuccessResponse(status=200, respond=data)
