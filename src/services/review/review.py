from __future__ import annotations

from datetime import datetime, timezone, timedelta, date

from src.model.review_schemas import SentimentSummary, RatingTrend, DestinationRatingSummary


def _to_range(start_date: date, end_date: date) -> tuple[datetime, datetime]:
    start_dt = datetime(start_date.year, start_date.month, start_date.day, tzinfo=timezone.utc)
    end_dt = datetime(end_date.year, end_date.month, end_date.day, tzinfo=timezone.utc) + timedelta(days=1)
    return start_dt, end_dt


async def getSentimentSummary(db, start_date: date, end_date: date) -> list[SentimentSummary]:
    start_dt, end_dt = _to_range(start_date, end_date)

    pipeline = [
        {"$match": {"review_date": {"$gte": start_dt, "$lt": end_dt}}},
        {
            "$group": {
                "_id": "$sentiment",
                "review_count": {"$sum": 1},
                "avg_rating": {"$avg": "$rating"},
            }
        },
        {"$sort": {"review_count": -1}},
    ]

    cursor = db.reviews.aggregate(pipeline)

    results: list[SentimentSummary] = []
    async for doc in cursor:
        results.append(
            SentimentSummary(
                sentiment=doc["_id"] or "Unknown",
                review_count=doc.get("review_count", 0),
                avg_rating=doc.get("avg_rating", 0.0),
            )
        )
    return results


async def getRatingTrend(
    db,
    start_date: date,
    end_date: date,
    granularity: str = "monthly",
) -> list[RatingTrend]:
    start_dt, end_dt = _to_range(start_date, end_date)

    if granularity == "weekly":
        group_id = {"iso_year": {"$isoWeekYear": "$review_date"}, "iso_week": {"$isoWeek": "$review_date"}}
    elif granularity == "annually":
        group_id = {"year": {"$year": "$review_date"}}
    else:
        group_id = {"year": {"$year": "$review_date"}, "month": {"$month": "$review_date"}}

    pipeline = [
        {"$match": {"review_date": {"$gte": start_dt, "$lt": end_dt}}},
        {
            "$group": {
                "_id": group_id,
                "avg_rating": {"$avg": "$rating"},
                "review_count": {"$sum": 1},
            }
        },
        {"$sort": {"_id": 1}},
    ]

    cursor = db.reviews.aggregate(pipeline)

    results: list[RatingTrend] = []
    async for doc in cursor:
        _id = doc["_id"]
        if granularity == "weekly":
            period = date.fromisocalendar(_id["iso_year"], _id["iso_week"], 1)
        elif granularity == "annually":
            period = date(_id["year"], 1, 1)
        else:
            period = date(_id["year"], _id["month"], 1)

        results.append(
            RatingTrend(
                period=period,
                granularity=granularity,
                avg_rating=doc.get("avg_rating", 0.0),
                review_count=doc.get("review_count", 0),
            )
        )
    return results


async def getDestinationRatingSummary(
    db,
    start_date: date,
    end_date: date,
    limit: int = 10,
) -> list[DestinationRatingSummary]:
    start_dt, end_dt = _to_range(start_date, end_date)

    pipeline = [
        {"$match": {"review_date": {"$gte": start_dt, "$lt": end_dt}}},
        {
            "$group": {
                "_id": "$destination_name",
                "avg_rating": {"$avg": "$rating"},
                "review_count": {"$sum": 1},
            }
        },
        {"$sort": {"avg_rating": -1}},
        {"$limit": limit},
    ]

    cursor = db.reviews.aggregate(pipeline)

    results: list[DestinationRatingSummary] = []
    async for doc in cursor:
        results.append(
            DestinationRatingSummary(
                destination_name=doc["_id"] or "Unknown",
                avg_rating=doc.get("avg_rating", 0.0),
                review_count=doc.get("review_count", 0),
            )
        )
    return results
