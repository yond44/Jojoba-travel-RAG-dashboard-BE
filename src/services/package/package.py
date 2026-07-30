from __future__ import annotations

from src.model.package_schemas import PackagePerformance, PackageTypeSummary


async def getTopPackages(db, limit: int = 10) -> list[PackagePerformance]:
    cursor = db.packages.find(
        {},
        {
            "package_code": 1,
            "name": 1,
            "destination_name": 1,
            "type": 1,
            "nights": 1,
            "price_per_pax_idr": 1,
            "rating": 1,
            "total_bookings": 1,
            "_id": 0,
        },
    ).sort("total_bookings", -1).limit(limit)

    results: list[PackagePerformance] = []
    async for doc in cursor:
        results.append(
            PackagePerformance(
                package_code=doc.get("package_code", ""),
                name=doc.get("name", ""),
                destination_name=doc.get("destination_name", ""),
                type=doc.get("type", "Unknown"),
                nights=doc.get("nights", 0),
                price_per_pax_idr=doc.get("price_per_pax_idr", 0.0),
                rating=doc.get("rating", 0.0),
                total_bookings=doc.get("total_bookings", 0),
            )
        )
    return results


async def getPackageTypeSummary(db) -> list[PackageTypeSummary]:
    pipeline = [
        {
            "$group": {
                "_id": "$type",
                "package_count": {"$sum": 1},
                "total_bookings": {"$sum": "$total_bookings"},
                "avg_rating": {"$avg": "$rating"},
                "avg_price_per_pax_idr": {"$avg": "$price_per_pax_idr"},
            }
        },
        {"$sort": {"total_bookings": -1}},
    ]

    cursor = db.packages.aggregate(pipeline)

    results: list[PackageTypeSummary] = []
    async for doc in cursor:
        results.append(
            PackageTypeSummary(
                type=doc["_id"] or "Unknown",
                package_count=doc.get("package_count", 0),
                total_bookings=doc.get("total_bookings", 0),
                avg_rating=doc.get("avg_rating", 0.0),
                avg_price_per_pax_idr=doc.get("avg_price_per_pax_idr", 0.0),
            )
        )
    return results
