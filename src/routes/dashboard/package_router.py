from __future__ import annotations

from fastapi import Depends, APIRouter, HTTPException
from motor.motor_asyncio import AsyncIOMotorDatabase
from pymongo.errors import PyMongoError

from src.utils.log import logger
from src.model.revenue_schemas import SuccessResponse
from src.config.database import get_db
from src.services.package.package import getTopPackages, getPackageTypeSummary


router = APIRouter(prefix="/api/v1/packages", tags=["packages"])


@router.get("/top", response_model=SuccessResponse)
async def getTopPackagesRoute(
    limit: int = 10,
    db: AsyncIOMotorDatabase = Depends(get_db),
):
    try:
        data = await getTopPackages(db, limit)
    except PyMongoError as e:
        logger.error(f"DB error in getTopPackagesRoute: {e}")
        raise HTTPException(status_code=503, detail="Database unavailable")
    except Exception as e:
        logger.exception("Unexpected error in getTopPackagesRoute")
        raise HTTPException(status_code=500, detail="Internal server error")

    return SuccessResponse(status=200, respond=data)


@router.get("/type-summary", response_model=SuccessResponse)
async def getPackageTypeSummaryRoute(db: AsyncIOMotorDatabase = Depends(get_db)):
    try:
        data = await getPackageTypeSummary(db)
    except PyMongoError as e:
        logger.error(f"DB error in getPackageTypeSummaryRoute: {e}")
        raise HTTPException(status_code=503, detail="Database unavailable")
    except Exception as e:
        logger.exception("Unexpected error in getPackageTypeSummaryRoute")
        raise HTTPException(status_code=500, detail="Internal server error")

    return SuccessResponse(status=200, respond=data)
