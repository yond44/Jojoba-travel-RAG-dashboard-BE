from __future__ import annotations

from datetime import datetime, timezone, timedelta, date

from src.model.destination_schemas import DestinationPopularity, RegionSummary


def _to_range(start_date: date, end_date: date) -> tuple[datetime, datetime]:
    start_dt = datetime(start_date.year, start_date.month, start_date.day, tzinfo=timezone.utc)
    end_dt = datetime(end_date.year, end_date.month, end_date.day, tzinfo=timezone.utc) + timedelta(days=1)
    return start_dt, end_dt


async def getDestinationPopularityVsBookings(
    db,
    start_date: date,
    end_date: date,
) -> list[DestinationPopularity]:
    start_dt, end_dt = _to_range(start_date, end_date)

    popularity_map: dict[str, float] = {}
    destinations_cursor = db.destinations.find({}, {"name": 1, "popularity_score": 1, "_id": 0})
    async for doc in destinations_cursor:
        popularity_map[doc.get("name", "Unknown")] = doc.get("popularity_score", 0.0)

    pipeline = [
        {
            "$match": {
                "payment_status": "Paid",
                "status": "Completed",
                "created_at": {"$gte": start_dt, "$lt": end_dt},
            }
        },
        {
            "$group": {
                "_id": "$destination_name",
                "actual_booking_count": {"$sum": 1},
                "actual_revenue_idr": {"$sum": "$total_price_idr"},
            }
        },
    ]

    booking_stats: dict[str, dict] = {}
    cursor = db.bookings.aggregate(pipeline)
    async for doc in cursor:
        booking_stats[doc["_id"] or "Unknown"] = doc

    all_destinations = set(popularity_map) | set(booking_stats)

    results = [
        DestinationPopularity(
            destination_name=name,
            popularity_score=popularity_map.get(name, 0.0),
            actual_booking_count=booking_stats.get(name, {}).get("actual_booking_count", 0),
            actual_revenue_idr=booking_stats.get(name, {}).get("actual_revenue_idr", 0.0),
        )
        for name in all_destinations
    ]

    return sorted(results, key=lambda r: r.actual_revenue_idr, reverse=True)


async def getRegionSummary(db) -> list[RegionSummary]:
    pipeline = [
        {
            "$group": {
                "_id": "$region",
                "destination_count": {"$sum": 1},
                "avg_rating": {"$avg": "$avg_rating"},
            }
        },
        {"$sort": {"destination_count": -1}},
    ]

    cursor = db.destinations.aggregate(pipeline)

    results: list[RegionSummary] = []
    async for doc in cursor:
        results.append(
            RegionSummary(
                region=doc["_id"] or "Unknown",
                destination_count=doc.get("destination_count", 0),
                avg_rating=doc.get("avg_rating", 0.0),
            )
        )
    return results
