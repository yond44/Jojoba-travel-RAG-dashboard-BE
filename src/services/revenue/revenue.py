from __future__ import annotations

from datetime import datetime, timezone, timedelta, date
from collections import defaultdict

from src.utils.log import logger
from src.model.revenue_schemas import (
    CurrentRevenueData,
    PeriodRevenueData,
    DailyRevenueData,
    RevenueByDestination,
    RevenueByPackageType,
    RevenueByChannel,
    RevenueByPaymentMethod,
    RevenueBySegment,
    AgentPerformance,
    CustomerTypeRevenue,
    DiscountImpact,
    PaymentStatusSummary,
)


async def getCurrentRevenue(db) -> CurrentRevenueData:
    now = datetime.now(timezone.utc)
    today_start = datetime(now.year, now.month, now.day, tzinfo=timezone.utc)
    seven_days_ago = today_start - timedelta(days=7)
    month_start = datetime(now.year, now.month, 1, tzinfo=timezone.utc)
    year_start = datetime(now.year, 1, 1, tzinfo=timezone.utc)

    pipeline = [
        {"$match": {"payment_status": "Paid", "status": "Completed"}},
        {
            "$facet": {
                "today": [
                    {"$match": {"created_at": {"$gte": today_start}}},
                    {"$group": {"_id": None, "total": {"$sum": "$total_price_idr"}}},
                ],
                "last_seven_days": [
                    {"$match": {"created_at": {"$gte": seven_days_ago}}},
                    {"$group": {"_id": None, "total": {"$sum": "$total_price_idr"}}},
                ],
                "current_month": [
                    {"$match": {"created_at": {"$gte": month_start}}},
                    {"$group": {"_id": None, "total": {"$sum": "$total_price_idr"}}},
                ],
                "this_year": [
                    {"$match": {"created_at": {"$gte": year_start}}},
                    {"$group": {"_id": None, "total": {"$sum": "$total_price_idr"}}},
                ],
            }
        },
    ]

    result = await db.bookings.aggregate(pipeline).to_list(length=1)
    facet = result[0] if result else {}

    def extract(key: str) -> float:
        bucket = facet.get(key, [])
        return bucket[0]["total"] if bucket else 0.0

    return CurrentRevenueData(
        today=extract("today"),
        last_seven_days=extract("last_seven_days"),
        current_month=extract("current_month"),
        this_year=extract("this_year"),
    )


async def getPeriodRevenue(
    db,
    start_date: date,
    end_date: date,
    granularity: str = "monthly",
) -> list[PeriodRevenueData]:
    start_dt = datetime(start_date.year, start_date.month, start_date.day, tzinfo=timezone.utc)
    end_dt = datetime(end_date.year, end_date.month, end_date.day, tzinfo=timezone.utc) + timedelta(days=1)

    cursor = db.bookings.find(
        {
            "payment_status": "Paid",
            "status": "Completed",
            "created_at": {"$gte": start_dt, "$lt": end_dt},
        },
        {"total_price_idr": 1, "created_at": 1, "_id": 0},
    )

    buckets: dict[date, float] = defaultdict(float)

    async for doc in cursor:
        created_at: datetime = doc["created_at"]
        amount: float = doc.get("total_price_idr", 0.0)

        if granularity == "weekly":
            key = created_at.date() - timedelta(days=created_at.weekday())
        elif granularity == "monthly":
            key = date(created_at.year, created_at.month, 1)
        elif granularity == "annually":
            key = date(created_at.year, 1, 1)
        else:
            key = created_at.date()

        buckets[key] += amount

    return [
        PeriodRevenueData(period=period, granularity=granularity, revenue=total)
        for period, total in sorted(buckets.items())
    ]


async def getDailyRevenue(
    db,
    start_date: date,
    end_date: date,
) -> list[DailyRevenueData]:
    start_dt = datetime(start_date.year, start_date.month, start_date.day, tzinfo=timezone.utc)
    end_dt = datetime(end_date.year, end_date.month, end_date.day, tzinfo=timezone.utc) + timedelta(days=1)

    cursor = db.bookings.find(
        {
            "payment_status": "Paid",
            "status": "Completed",
            "created_at": {"$gte": start_dt, "$lt": end_dt},
        },
        {"total_price_idr": 1, "created_at": 1, "_id": 0},
    )

    buckets: dict[date, float] = defaultdict(float)

    async for doc in cursor:
        created_at: datetime = doc["created_at"]
        amount: float = doc.get("total_price_idr", 0.0)
        buckets[created_at.date()] += amount

    all_days = [start_date + timedelta(days=i) for i in range((end_date - start_date).days + 1)]

    return [
        DailyRevenueData(date=day, revenue=buckets.get(day, 0.0))
        for day in all_days
    ]


async def getRevenueByDestination(
    db,
    start_date: date,
    end_date: date,
) -> list[RevenueByDestination]:
    start_dt = datetime(start_date.year, start_date.month, start_date.day, tzinfo=timezone.utc)
    end_dt = datetime(end_date.year, end_date.month, end_date.day, tzinfo=timezone.utc) + timedelta(days=1)

    cursor = db.bookings.find(
        {
            "payment_status": "Paid",
            "status": "Completed",
            "created_at": {"$gte": start_dt, "$lt": end_dt},
        },
        {"total_price_idr": 1, "destination_name": 1, "_id": 0},
    )

    revenue_buckets: dict[str, float] = defaultdict(float)
    count_buckets: dict[str, int] = defaultdict(int)

    async for doc in cursor:
        key = doc.get("destination_name", "Unknown")
        revenue_buckets[key] += doc.get("total_price_idr", 0.0)
        count_buckets[key] += 1

    return [
        RevenueByDestination(destination_name=key, revenue=revenue_buckets[key], booking_count=count_buckets[key])
        for key in sorted(revenue_buckets, key=lambda k: revenue_buckets[k], reverse=True)
    ]


async def getRevenueByPackageType(
    db,
    start_date: date,
    end_date: date,
) -> list[RevenueByPackageType]:
    start_dt = datetime(start_date.year, start_date.month, start_date.day, tzinfo=timezone.utc)
    end_dt = datetime(end_date.year, end_date.month, end_date.day, tzinfo=timezone.utc) + timedelta(days=1)

    cursor = db.bookings.find(
        {
            "payment_status": "Paid",
            "status": "Completed",
            "created_at": {"$gte": start_dt, "$lt": end_dt},
        },
        {"total_price_idr": 1, "package_type": 1, "_id": 0},
    )

    revenue_buckets: dict[str, float] = defaultdict(float)
    count_buckets: dict[str, int] = defaultdict(int)

    async for doc in cursor:
        key = doc.get("package_type", "Unknown")
        revenue_buckets[key] += doc.get("total_price_idr", 0.0)
        count_buckets[key] += 1

    return [
        RevenueByPackageType(package_type=key, revenue=revenue_buckets[key], booking_count=count_buckets[key])
        for key in sorted(revenue_buckets, key=lambda k: revenue_buckets[k], reverse=True)
    ]


async def getRevenueByChannel(
    db,
    start_date: date,
    end_date: date,
) -> list[RevenueByChannel]:
    start_dt = datetime(start_date.year, start_date.month, start_date.day, tzinfo=timezone.utc)
    end_dt = datetime(end_date.year, end_date.month, end_date.day, tzinfo=timezone.utc) + timedelta(days=1)

    cursor = db.bookings.find(
        {
            "payment_status": "Paid",
            "status": "Completed",
            "created_at": {"$gte": start_dt, "$lt": end_dt},
        },
        {"total_price_idr": 1, "channel": 1, "_id": 0},
    )

    revenue_buckets: dict[str, float] = defaultdict(float)
    count_buckets: dict[str, int] = defaultdict(int)

    async for doc in cursor:
        key = doc.get("channel", "Unknown")
        revenue_buckets[key] += doc.get("total_price_idr", 0.0)
        count_buckets[key] += 1

    return [
        RevenueByChannel(channel=key, revenue=revenue_buckets[key], booking_count=count_buckets[key])
        for key in sorted(revenue_buckets, key=lambda k: revenue_buckets[k], reverse=True)
    ]


async def getRevenueByPaymentMethod(
    db,
    start_date: date,
    end_date: date,
) -> list[RevenueByPaymentMethod]:
    start_dt = datetime(start_date.year, start_date.month, start_date.day, tzinfo=timezone.utc)
    end_dt = datetime(end_date.year, end_date.month, end_date.day, tzinfo=timezone.utc) + timedelta(days=1)

    cursor = db.bookings.find(
        {
            "payment_status": "Paid",
            "status": "Completed",
            "created_at": {"$gte": start_dt, "$lt": end_dt},
        },
        {"total_price_idr": 1, "payment_method": 1, "_id": 0},
    )

    revenue_buckets: dict[str, float] = defaultdict(float)
    count_buckets: dict[str, int] = defaultdict(int)

    async for doc in cursor:
        key = doc.get("payment_method", "Unknown")
        revenue_buckets[key] += doc.get("total_price_idr", 0.0)
        count_buckets[key] += 1

    return [
        RevenueByPaymentMethod(payment_method=key, revenue=revenue_buckets[key], booking_count=count_buckets[key])
        for key in sorted(revenue_buckets, key=lambda k: revenue_buckets[k], reverse=True)
    ]


async def getRevenueBySegment(
    db,
    start_date: date,
    end_date: date,
) -> list[RevenueBySegment]:
    start_dt = datetime(start_date.year, start_date.month, start_date.day, tzinfo=timezone.utc)
    end_dt = datetime(end_date.year, end_date.month, end_date.day, tzinfo=timezone.utc) + timedelta(days=1)

    cursor = db.bookings.find(
        {
            "payment_status": "Paid",
            "status": "Completed",
            "created_at": {"$gte": start_dt, "$lt": end_dt},
        },
        {"total_price_idr": 1, "customer_segment": 1, "_id": 0},
    )

    revenue_buckets: dict[str, float] = defaultdict(float)
    count_buckets: dict[str, int] = defaultdict(int)

    async for doc in cursor:
        key = doc.get("customer_segment", "Unknown")
        revenue_buckets[key] += doc.get("total_price_idr", 0.0)
        count_buckets[key] += 1

    return [
        RevenueBySegment(customer_segment=key, revenue=revenue_buckets[key], booking_count=count_buckets[key])
        for key in sorted(revenue_buckets, key=lambda k: revenue_buckets[k], reverse=True)
    ]


async def getAgentPerformance(
    db,
    start_date: date,
    end_date: date,
) -> list[AgentPerformance]:
    start_dt = datetime(start_date.year, start_date.month, start_date.day, tzinfo=timezone.utc)
    end_dt = datetime(end_date.year, end_date.month, end_date.day, tzinfo=timezone.utc) + timedelta(days=1)

    cursor = db.bookings.find(
        {
            "payment_status": "Paid",
            "status": "Completed",
            "created_at": {"$gte": start_dt, "$lt": end_dt},
        },
        {"total_price_idr": 1, "agent_id": 1, "_id": 0},
    )

    revenue_buckets: dict[str, float] = defaultdict(float)
    count_buckets: dict[str, int] = defaultdict(int)

    async for doc in cursor:
        key = str(doc.get("agent_id", "Unknown"))
        revenue_buckets[key] += doc.get("total_price_idr", 0.0)
        count_buckets[key] += 1

    return [
        AgentPerformance(agent_id=key, revenue=revenue_buckets[key], booking_count=count_buckets[key])
        for key in sorted(revenue_buckets, key=lambda k: revenue_buckets[k], reverse=True)
    ]


async def getCustomerTypeRevenue(
    db,
    start_date: date,
    end_date: date,
) -> list[CustomerTypeRevenue]:
    start_dt = datetime(start_date.year, start_date.month, start_date.day, tzinfo=timezone.utc)
    end_dt = datetime(end_date.year, end_date.month, end_date.day, tzinfo=timezone.utc) + timedelta(days=1)

    cursor = db.bookings.find(
        {
            "payment_status": "Paid",
            "status": "Completed",
            "created_at": {"$gte": start_dt, "$lt": end_dt},
        },
        {"total_price_idr": 1, "is_repeat_customer": 1, "_id": 0},
    )

    revenue_buckets: dict[bool, float] = defaultdict(float)
    count_buckets: dict[bool, int] = defaultdict(int)

    async for doc in cursor:
        key = bool(doc.get("is_repeat_customer", False))
        revenue_buckets[key] += doc.get("total_price_idr", 0.0)
        count_buckets[key] += 1

    return [
        CustomerTypeRevenue(is_repeat_customer=key, revenue=revenue_buckets[key], booking_count=count_buckets[key])
        for key in revenue_buckets
    ]


async def getDiscountImpact(
    db,
    start_date: date,
    end_date: date,
) -> DiscountImpact:
    start_dt = datetime(start_date.year, start_date.month, start_date.day, tzinfo=timezone.utc)
    end_dt = datetime(end_date.year, end_date.month, end_date.day, tzinfo=timezone.utc) + timedelta(days=1)

    cursor = db.bookings.find(
        {
            "payment_status": "Paid",
            "status": "Completed",
            "created_at": {"$gte": start_dt, "$lt": end_dt},
        },
        {"total_price_idr": 1, "discount_idr": 1, "_id": 0},
    )

    total_discount = 0.0
    total_revenue = 0.0

    async for doc in cursor:
        total_discount += doc.get("discount_idr", 0.0)
        total_revenue += doc.get("total_price_idr", 0.0)

    gross_revenue = total_revenue + total_discount
    discount_rate = (total_discount / gross_revenue) if gross_revenue else 0.0

    return DiscountImpact(
        total_discount=total_discount,
        total_revenue=total_revenue,
        discount_rate=discount_rate,
    )


async def getPaymentStatusSummary(
    db,
    start_date: date,
    end_date: date,
) -> list[PaymentStatusSummary]:
    start_dt = datetime(start_date.year, start_date.month, start_date.day, tzinfo=timezone.utc)
    end_dt = datetime(end_date.year, end_date.month, end_date.day, tzinfo=timezone.utc) + timedelta(days=1)

    cursor = db.bookings.find(
        {
            "created_at": {"$gte": start_dt, "$lt": end_dt},
        },
        {"total_price_idr": 1, "payment_status": 1, "_id": 0},
    )

    amount_buckets: dict[str, float] = defaultdict(float)
    count_buckets: dict[str, int] = defaultdict(int)

    async for doc in cursor:
        key = doc.get("payment_status", "Unknown")
        amount_buckets[key] += doc.get("total_price_idr", 0.0)
        count_buckets[key] += 1

    return [
        PaymentStatusSummary(payment_status=key, total_amount=amount_buckets[key], booking_count=count_buckets[key])
        for key in sorted(amount_buckets, key=lambda k: amount_buckets[k], reverse=True)
    ]