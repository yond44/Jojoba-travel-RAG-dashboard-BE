from __future__ import annotations

import re

from datetime import datetime, timezone, timedelta, date
from collections import defaultdict

from src.model.customer_schemas import (
    SegmentDistribution,
    AgeGroupDistribution,
    CityDistribution,
    AcquisitionChannelDistribution,
    TopCustomer,
    NewCustomersByDay,
    RepeatCustomerRatio,
    ChurnRiskCustomer,
    CustomerSearchResult
)
MAX_SEARCH_LIMIT = 25
MAX_RISK_LIMIT = 100

def _to_range(start_date: date, end_date: date) -> tuple[datetime, datetime]:
    start_dt = datetime(start_date.year, start_date.month, start_date.day, tzinfo=timezone.utc)
    end_dt = datetime(end_date.year, end_date.month, end_date.day, tzinfo=timezone.utc) + timedelta(days=1)
    return start_dt, end_dt


async def getSegmentDistribution(db) -> list[SegmentDistribution]:
    cursor = db.customers.find({}, {"segment": 1, "total_spent_idr": 1, "_id": 0})

    count_buckets: dict[str, int] = defaultdict(int)
    spent_buckets: dict[str, float] = defaultdict(float)

    async for doc in cursor:
        key = doc.get("segment", "Unknown")
        count_buckets[key] += 1
        spent_buckets[key] += doc.get("total_spent_idr", 0.0)

    return [
        SegmentDistribution(segment=key, customer_count=count_buckets[key], total_spent_idr=spent_buckets[key])
        for key in sorted(count_buckets, key=lambda k: spent_buckets[k], reverse=True)
    ]


async def getAgeGroupDistribution(db) -> list[AgeGroupDistribution]:
    cursor = db.customers.find({}, {"age_group": 1, "_id": 0})

    count_buckets: dict[str, int] = defaultdict(int)

    async for doc in cursor:
        key = doc.get("age_group", "Unknown")
        count_buckets[key] += 1

    return [
        AgeGroupDistribution(age_group=key, customer_count=count_buckets[key])
        for key in sorted(count_buckets)
    ]


async def getCityDistribution(db, limit: int = 10) -> list[CityDistribution]:
    cursor = db.customers.find({}, {"city": 1, "_id": 0})

    count_buckets: dict[str, int] = defaultdict(int)

    async for doc in cursor:
        key = doc.get("city", "Unknown")
        count_buckets[key] += 1

    top_keys = sorted(count_buckets, key=lambda k: count_buckets[k], reverse=True)[:limit]

    return [CityDistribution(city=key, customer_count=count_buckets[key]) for key in top_keys]


async def getAcquisitionChannelDistribution(db) -> list[AcquisitionChannelDistribution]:
    cursor = db.customers.find({}, {"acquisition_channel": 1, "_id": 0})

    count_buckets: dict[str, int] = defaultdict(int)

    async for doc in cursor:
        key = doc.get("acquisition_channel", "Unknown")
        count_buckets[key] += 1

    return [
        AcquisitionChannelDistribution(acquisition_channel=key, customer_count=count_buckets[key])
        for key in sorted(count_buckets, key=lambda k: count_buckets[k], reverse=True)
    ]


async def getTopCustomers(db, limit: int = 10) -> list[TopCustomer]:
    cursor = db.customers.find(
        {},
        {"customer_code": 1, "name": 1, "segment": 1, "total_trips": 1, "total_spent_idr": 1, "_id": 0},
    ).sort("total_spent_idr", -1).limit(limit)

    results: list[TopCustomer] = []
    async for doc in cursor:
        results.append(
            TopCustomer(
                customer_code=doc.get("customer_code", ""),
                name=doc.get("name", ""),
                segment=doc.get("segment", "Unknown"),
                total_trips=doc.get("total_trips", 0),
                total_spent_idr=doc.get("total_spent_idr", 0.0),
            )
        )
    return results


async def getNewCustomersByDay(db, start_date: date, end_date: date) -> list[NewCustomersByDay]:
    start_dt, end_dt = _to_range(start_date, end_date)

    cursor = db.customers.find(
        {"join_date": {"$gte": start_dt, "$lt": end_dt}},
        {"join_date": 1, "_id": 0},
    )

    buckets: dict[date, int] = defaultdict(int)

    async for doc in cursor:
        join_date: datetime = doc["join_date"]
        buckets[join_date.date()] += 1

    all_days = [start_date + timedelta(days=i) for i in range((end_date - start_date).days + 1)]

    return [
        NewCustomersByDay(date=day, new_customer_count=buckets.get(day, 0))
        for day in all_days
    ]


async def getRepeatCustomerRatio(db) -> RepeatCustomerRatio:
    total_repeat = await db.customers.count_documents({"is_repeat_customer": True})
    total_new = await db.customers.count_documents({"is_repeat_customer": False})

    total = total_repeat + total_new
    repeat_rate = (total_repeat / total) if total else 0.0

    return RepeatCustomerRatio(
        repeat_customer_count=total_repeat,
        new_customer_count=total_new,
        repeat_rate=repeat_rate,
    )
    
    
    
# ---------- Pencarian pelanggan ----------
async def searchCustomers(db, query: str, limit: int = 10) -> list[CustomerSearchResult]:
    cleaned = (query or "").strip()
    if len(cleaned) < 2:
        return []
 
    pattern = re.escape(cleaned)
    cursor = db.customers.find(
        {
            "$or": [
                {"name": {"$regex": pattern, "$options": "i"}},
                {"customer_code": {"$regex": pattern, "$options": "i"}},
            ]
        },
        {"name": 1, "customer_code": 1, "segment": 1, "total_spent_idr": 1,
         "total_trips": 1},
    ).limit(min(limit, MAX_SEARCH_LIMIT))
 
    results: list[CustomerSearchResult] = []
    async for doc in cursor:
        results.append(
            CustomerSearchResult(
                customer_id=str(doc["_id"]),
                customer_code=doc.get("customer_code", ""),
                name=doc.get("name", ""),
                segment=doc.get("segment", "Unknown"),
                total_trips=doc.get("total_trips", 0),
                total_spent_idr=doc.get("total_spent_idr", 0.0),
            )
        )
    return results
 
 
# ---------- Daftar pelanggan berisiko ----------
async def getChurnRiskList(db, bucket: str | None = None,
                           limit: int = 25) -> list[ChurnRiskCustomer]:
    pipeline: list[dict] = []
    if bucket:
        pipeline.append({"$match": {"risk_bucket": bucket}})
 
    pipeline.extend([
        {"$sort": {"churn_proba": -1, "monetary_total": -1}},
        {"$limit": min(limit, MAX_RISK_LIMIT)},
        {"$addFields": {
            "customer_object_id": {
                "$convert": {
                    "input": "$customer_id",
                    "to": "objectId",
                    "onError": None,
                    "onNull": None,
                }
            }
        }},
        {"$lookup": {
            "from": "customers",
            "localField": "customer_object_id",
            "foreignField": "_id",
            "as": "profile",
        }},
        {"$unwind": {"path": "$profile", "preserveNullAndEmptyArrays": True}},
    ])
 
    results: list[ChurnRiskCustomer] = []
    async for doc in db.ml_churn_scores.aggregate(pipeline):
        profile = doc.get("profile") or {}
        results.append(
            ChurnRiskCustomer(
                customer_id=str(doc.get("customer_id", "")),
                customer_code=profile.get("customer_code", ""),
                name=profile.get("name", "Tanpa nama"),
                segment=profile.get("segment", "Unknown"),
                churn_proba=float(doc.get("churn_proba", 0.0)),
                risk_bucket=doc.get("risk_bucket", "Low"),
                track=doc.get("track", "single"),
                monetary_total=float(doc.get("monetary_total", 0.0)),
                tenure_days=int(doc.get("tenure_days", 0)),
                scored_at=str(doc.get("scored_at", "")),
            )
        )
    return results
 
