from __future__ import annotations
from datetime import date

from fastapi import Depends, APIRouter, HTTPException
from motor.motor_asyncio import AsyncIOMotorDatabase
from pymongo.errors import PyMongoError

from src.utils.log import logger
from src.model.revenue_schemas import SuccessResponse
from src.config.database import get_db
from src.services.customer.customer import (
    getSegmentDistribution,
    getAgeGroupDistribution,
    getCityDistribution,
    getAcquisitionChannelDistribution,
    getTopCustomers,
    getNewCustomersByDay,
    getRepeatCustomerRatio,
)


router = APIRouter(prefix="/api/v1/customers", tags=["customers"])


@router.get("/segment-distribution", response_model=SuccessResponse)
async def getSegmentDistributionRoute(db: AsyncIOMotorDatabase = Depends(get_db)):
    try:
        data = await getSegmentDistribution(db)
    except PyMongoError as e:
        logger.error(f"DB error in getSegmentDistributionRoute: {e}")
        raise HTTPException(status_code=503, detail="Database unavailable")
    except Exception as e:
        logger.exception("Unexpected error in getSegmentDistributionRoute")
        raise HTTPException(status_code=500, detail="Internal server error")

    return SuccessResponse(status=200, respond=data)


@router.get("/age-group-distribution", response_model=SuccessResponse)
async def getAgeGroupDistributionRoute(db: AsyncIOMotorDatabase = Depends(get_db)):
    try:
        data = await getAgeGroupDistribution(db)
    except PyMongoError as e:
        logger.error(f"DB error in getAgeGroupDistributionRoute: {e}")
        raise HTTPException(status_code=503, detail="Database unavailable")
    except Exception as e:
        logger.exception("Unexpected error in getAgeGroupDistributionRoute")
        raise HTTPException(status_code=500, detail="Internal server error")

    return SuccessResponse(status=200, respond=data)


@router.get("/city-distribution", response_model=SuccessResponse)
async def getCityDistributionRoute(
    limit: int = 10,
    db: AsyncIOMotorDatabase = Depends(get_db),
):
    try:
        data = await getCityDistribution(db, limit)
    except PyMongoError as e:
        logger.error(f"DB error in getCityDistributionRoute: {e}")
        raise HTTPException(status_code=503, detail="Database unavailable")
    except Exception as e:
        logger.exception("Unexpected error in getCityDistributionRoute")
        raise HTTPException(status_code=500, detail="Internal server error")

    return SuccessResponse(status=200, respond=data)


@router.get("/acquisition-channel", response_model=SuccessResponse)
async def getAcquisitionChannelDistributionRoute(db: AsyncIOMotorDatabase = Depends(get_db)):
    try:
        data = await getAcquisitionChannelDistribution(db)
    except PyMongoError as e:
        logger.error(f"DB error in getAcquisitionChannelDistributionRoute: {e}")
        raise HTTPException(status_code=503, detail="Database unavailable")
    except Exception as e:
        logger.exception("Unexpected error in getAcquisitionChannelDistributionRoute")
        raise HTTPException(status_code=500, detail="Internal server error")

    return SuccessResponse(status=200, respond=data)


@router.get("/top", response_model=SuccessResponse)
async def getTopCustomersRoute(
    limit: int = 10,
    db: AsyncIOMotorDatabase = Depends(get_db),
):
    try:
        data = await getTopCustomers(db, limit)
    except PyMongoError as e:
        logger.error(f"DB error in getTopCustomersRoute: {e}")
        raise HTTPException(status_code=503, detail="Database unavailable")
    except Exception as e:
        logger.exception("Unexpected error in getTopCustomersRoute")
        raise HTTPException(status_code=500, detail="Internal server error")

    return SuccessResponse(status=200, respond=data)


@router.get("/new-by-day", response_model=SuccessResponse)
async def getNewCustomersByDayRoute(
    start_date: date,
    end_date: date,
    db: AsyncIOMotorDatabase = Depends(get_db),
):
    if start_date > end_date:
        raise HTTPException(status_code=400, detail="start_date must be before end_date")

    try:
        data = await getNewCustomersByDay(db, start_date, end_date)
    except PyMongoError as e:
        logger.error(f"DB error in getNewCustomersByDayRoute: {e}")
        raise HTTPException(status_code=503, detail="Database unavailable")
    except Exception as e:
        logger.exception("Unexpected error in getNewCustomersByDayRoute")
        raise HTTPException(status_code=500, detail="Internal server error")

    return SuccessResponse(status=200, respond=data)


@router.get("/repeat-ratio", response_model=SuccessResponse)
async def getRepeatCustomerRatioRoute(db: AsyncIOMotorDatabase = Depends(get_db)):
    try:
        data = await getRepeatCustomerRatio(db)
    except PyMongoError as e:
        logger.error(f"DB error in getRepeatCustomerRatioRoute: {e}")
        raise HTTPException(status_code=503, detail="Database unavailable")
    except Exception as e:
        logger.exception("Unexpected error in getRepeatCustomerRatioRoute")
        raise HTTPException(status_code=500, detail="Internal server error")

    return SuccessResponse(status=200, respond=data)
