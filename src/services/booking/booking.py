from __future__ import annotations

from datetime import datetime, timezone, timedelta, date

from src.model.booking_schemas import (
    BookingStatusSummary,
    BookingVolumeByDay,
    LeadTimeStats,
    CancellationRate,
    PaymentGatewayPerformance,
)


def _to_range(start_date: date, end_date: date) -> tuple[datetime, datetime]:
    start_dt = datetime(start_date.year, start_date.month, start_date.day, tzinfo=timezone.utc)
    end_dt = datetime(end_date.year, end_date.month, end_date.day, tzinfo=timezone.utc) + timedelta(days=1)
    return start_dt, end_dt


async def getBookingStatusSummary(db, start_date: date, end_date: date) -> list[BookingStatusSummary]:
    start_dt, end_dt = _to_range(start_date, end_date)

    pipeline = [
        {"$match": {"created_at": {"$gte": start_dt, "$lt": end_dt}}},
        {"$group": {"_id": "$status", "booking_count": {"$sum": 1}}},
        {"$sort": {"booking_count": -1}},
    ]

    cursor = db.bookings.aggregate(pipeline)

    results: list[BookingStatusSummary] = []
    async for doc in cursor:
        results.append(
            BookingStatusSummary(status=doc["_id"] or "Unknown", booking_count=doc.get("booking_count", 0))
        )
    return results


async def getBookingVolume(db, start_date: date, end_date: date) -> list[BookingVolumeByDay]:
    start_dt, end_dt = _to_range(start_date, end_date)

    pipeline = [
        {"$match": {"created_at": {"$gte": start_dt, "$lt": end_dt}}},
        {
            "$group": {
                "_id": {"$dateToString": {"format": "%Y-%m-%d", "date": "$created_at"}},
                "booking_count": {"$sum": 1},
            }
        },
    ]

    cursor = db.bookings.aggregate(pipeline)

    counts: dict[date, int] = {}
    async for doc in cursor:
        day = datetime.strptime(doc["_id"], "%Y-%m-%d").date()
        counts[day] = doc.get("booking_count", 0)

    all_days = [start_date + timedelta(days=i) for i in range((end_date - start_date).days + 1)]

    return [BookingVolumeByDay(date=day, booking_count=counts.get(day, 0)) for day in all_days]


async def getLeadTimeStats(db, start_date: date, end_date: date) -> LeadTimeStats:
    start_dt, end_dt = _to_range(start_date, end_date)

    pipeline = [
        {"$match": {"created_at": {"$gte": start_dt, "$lt": end_dt}}},
        {
            "$project": {
                "lead_time_days": {
                    "$divide": [{"$subtract": ["$travel_date", "$booking_date"]}, 1000 * 60 * 60 * 24]
                }
            }
        },
        {
            "$group": {
                "_id": None,
                "avg_lead_time_days": {"$avg": "$lead_time_days"},
                "min_lead_time_days": {"$min": "$lead_time_days"},
                "max_lead_time_days": {"$max": "$lead_time_days"},
            }
        },
    ]

    result = await db.bookings.aggregate(pipeline).to_list(length=1)

    if not result:
        return LeadTimeStats(avg_lead_time_days=0.0, min_lead_time_days=0, max_lead_time_days=0)

    doc = result[0]
    return LeadTimeStats(
        avg_lead_time_days=round(doc.get("avg_lead_time_days") or 0.0, 1),
        min_lead_time_days=int(doc.get("min_lead_time_days") or 0),
        max_lead_time_days=int(doc.get("max_lead_time_days") or 0),
    )


async def getCancellationRate(db, start_date: date, end_date: date) -> CancellationRate:
    start_dt, end_dt = _to_range(start_date, end_date)

    total = await db.bookings.count_documents({"created_at": {"$gte": start_dt, "$lt": end_dt}})
    cancelled = await db.bookings.count_documents(
        {"created_at": {"$gte": start_dt, "$lt": end_dt}, "status": "Cancelled"}
    )

    rate = (cancelled / total) if total else 0.0

    return CancellationRate(total_bookings=total, cancelled_bookings=cancelled, cancellation_rate=rate)


async def getPaymentGatewayPerformance(db, start_date: date, end_date: date) -> list[PaymentGatewayPerformance]:
    start_dt, end_dt = _to_range(start_date, end_date)

    pipeline = [
        {"$match": {"transaction_date": {"$gte": start_dt, "$lt": end_dt}}},
        {
            "$group": {
                "_id": "$payment_method",
                "success_count": {"$sum": {"$cond": [{"$eq": ["$status", "Success"]}, 1, 0]}},
                "failed_count": {"$sum": {"$cond": [{"$eq": ["$status", "Success"]}, 0, 1]}},
                "total_amount_idr": {"$sum": {"$cond": [{"$eq": ["$status", "Success"]}, "$amount_idr", 0]}},
            }
        },
        {"$sort": {"total_amount_idr": -1}},
    ]

    cursor = db.transactions.aggregate(pipeline)

    results: list[PaymentGatewayPerformance] = []
    async for doc in cursor:
        results.append(
            PaymentGatewayPerformance(
                payment_method=doc["_id"] or "Unknown",
                success_count=doc.get("success_count", 0),
                failed_count=doc.get("failed_count", 0),
                total_amount_idr=doc.get("total_amount_idr", 0.0),
            )
        )
    return results
