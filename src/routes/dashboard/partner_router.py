from __future__ import annotations

from fastapi import Depends, APIRouter, HTTPException
from motor.motor_asyncio import AsyncIOMotorDatabase
from pymongo.errors import PyMongoError

from src.utils.log import logger
from src.model.revenue_schemas import SuccessResponse
from src.config.database import get_db
from src.services.partner.partner import getTopHotels, getTopFlightRoutes


router = APIRouter(prefix="/api/v1/partners", tags=["partners"])


@router.get("/hotels/top", response_model=SuccessResponse)
async def getTopHotelsRoute(
    limit: int = 10,
    db: AsyncIOMotorDatabase = Depends(get_db),
):
    try:
        data = await getTopHotels(db, limit)
    except PyMongoError as e:
        logger.error(f"DB error in getTopHotelsRoute: {e}")
        raise HTTPException(status_code=503, detail="Database unavailable")
    except Exception as e:
        logger.exception("Unexpected error in getTopHotelsRoute")
        raise HTTPException(status_code=500, detail="Internal server error")

    return SuccessResponse(status=200, respond=data)


@router.get("/flights/top", response_model=SuccessResponse)
async def getTopFlightRoutesRoute(
    limit: int = 10,
    db: AsyncIOMotorDatabase = Depends(get_db),
):
    try:
        data = await getTopFlightRoutes(db, limit)
    except PyMongoError as e:
        logger.error(f"DB error in getTopFlightRoutesRoute: {e}")
        raise HTTPException(status_code=503, detail="Database unavailable")
    except Exception as e:
        logger.exception("Unexpected error in getTopFlightRoutesRoute")
        raise HTTPException(status_code=500, detail="Internal server error")

    return SuccessResponse(status=200, respond=data)
