from __future__ import annotations

from datetime import datetime, timezone, timedelta, date

from src.model.campaign_schemas import (
    CampaignPerformance,
    CampaignTypeSummary,
    CampaignFunnel,
)


def _to_range(start_date: date, end_date: date) -> tuple[datetime, datetime]:
    start_dt = datetime(start_date.year, start_date.month, start_date.day, tzinfo=timezone.utc)
    end_dt = datetime(end_date.year, end_date.month, end_date.day, tzinfo=timezone.utc) + timedelta(days=1)
    return start_dt, end_dt


async def getCampaignPerformance(
    db,
    start_date: date,
    end_date: date,
    limit: int = 20,
) -> list[CampaignPerformance]:
    start_dt, end_dt = _to_range(start_date, end_date)

    cursor = db.campaigns.find(
        {"start_date": {"$gte": start_dt, "$lt": end_dt}},
        {
            "campaign_code": 1,
            "name": 1,
            "type": 1,
            "spend_idr": 1,
            "revenue_generated_idr": 1,
            "roi": 1,
            "conversion_rate": 1,
            "_id": 0,
        },
    ).sort("roi", -1).limit(limit)

    results: list[CampaignPerformance] = []
    async for doc in cursor:
        results.append(
            CampaignPerformance(
                campaign_code=doc.get("campaign_code", ""),
                name=doc.get("name", ""),
                type=doc.get("type", "Unknown"),
                spend_idr=doc.get("spend_idr", 0.0),
                revenue_generated_idr=doc.get("revenue_generated_idr", 0.0),
                roi=doc.get("roi", 0.0),
                conversion_rate=doc.get("conversion_rate", 0.0),
            )
        )
    return results


async def getCampaignTypeSummary(db, start_date: date, end_date: date) -> list[CampaignTypeSummary]:
    start_dt, end_dt = _to_range(start_date, end_date)

    pipeline = [
        {"$match": {"start_date": {"$gte": start_dt, "$lt": end_dt}}},
        {
            "$group": {
                "_id": "$type",
                "total_spend_idr": {"$sum": "$spend_idr"},
                "total_revenue_idr": {"$sum": "$revenue_generated_idr"},
                "avg_roi": {"$avg": "$roi"},
            }
        },
        {"$sort": {"total_revenue_idr": -1}},
    ]

    cursor = db.campaigns.aggregate(pipeline)

    results: list[CampaignTypeSummary] = []
    async for doc in cursor:
        results.append(
            CampaignTypeSummary(
                type=doc["_id"] or "Unknown",
                total_spend_idr=doc.get("total_spend_idr", 0.0),
                total_revenue_idr=doc.get("total_revenue_idr", 0.0),
                avg_roi=doc.get("avg_roi", 0.0),
            )
        )
    return results


async def getCampaignFunnel(db, start_date: date, end_date: date) -> list[CampaignFunnel]:
    start_dt, end_dt = _to_range(start_date, end_date)

    cursor = db.campaigns.find(
        {"start_date": {"$gte": start_dt, "$lt": end_dt}},
        {"campaign_code": 1, "reach": 1, "impressions": 1, "clicks": 1, "conversions": 1, "_id": 0},
    )

    results: list[CampaignFunnel] = []
    async for doc in cursor:
        results.append(
            CampaignFunnel(
                campaign_code=doc.get("campaign_code", ""),
                reach=doc.get("reach", 0),
                impressions=doc.get("impressions", 0),
                clicks=doc.get("clicks", 0),
                conversions=doc.get("conversions", 0),
            )
        )
    return results
