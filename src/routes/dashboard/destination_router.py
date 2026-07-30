from __future__ import annotations
from datetime import date

from fastapi import Depends, APIRouter, HTTPException
from motor.motor_asyncio import AsyncIOMotorDatabase
from pymongo.errors import PyMongoError

from src.utils.log import logger
from src.model.revenue_schemas import SuccessResponse
from src.config.database import get_db
from src.services.destination.destination import getDestinationPopularityVsBookings, getRegionSummary


router = APIRouter(prefix="/api/v1/destinations", tags=["destinations"])


@router.get("/popularity-vs-bookings", response_model=SuccessResponse)
async def getDestinationPopularityVsBookingsRoute(
    start_date: date,
    end_date: date,
    db: AsyncIOMotorDatabase = Depends(get_db),
):
    if start_date > end_date:
        raise HTTPException(status_code=400, detail="start_date must be before end_date")

    try:
        data = await getDestinationPopularityVsBookings(db, start_date, end_date)
    except PyMongoError as e:
        logger.error(f"DB error in getDestinationPopularityVsBookingsRoute: {e}")
        raise HTTPException(status_code=503, detail="Database unavailable")
    except Exception as e:
        logger.exception("Unexpected error in getDestinationPopularityVsBookingsRoute")
        raise HTTPException(status_code=500, detail="Internal server error")

    return SuccessResponse(status=200, respond=data)


@router.get("/region-summary", response_model=SuccessResponse)
async def getRegionSummaryRoute(db: AsyncIOMotorDatabase = Depends(get_db)):
    try:
        data = await getRegionSummary(db)
    except PyMongoError as e:
        logger.error(f"DB error in getRegionSummaryRoute: {e}")
        raise HTTPException(status_code=503, detail="Database unavailable")
    except Exception as e:
        logger.exception("Unexpected error in getRegionSummaryRoute")
        raise HTTPException(status_code=500, detail="Internal server error")

    return SuccessResponse(status=200, respond=data)
