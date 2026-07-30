from __future__ import annotations

from src.model.partner_schemas import HotelPerformance, FlightPerformance


async def getTopHotels(db, limit: int = 10) -> list[HotelPerformance]:
    cursor = db.hotels.find(
        {},
        {
            "hotel_code": 1,
            "name": 1,
            "destination_name": 1,
            "star_rating": 1,
            "rating": 1,
            "total_reviews": 1,
            "commission_rate": 1,
            "_id": 0,
        },
    ).sort("rating", -1).limit(limit)

    results: list[HotelPerformance] = []
    async for doc in cursor:
        results.append(
            HotelPerformance(
                hotel_code=doc.get("hotel_code", ""),
                name=doc.get("name", ""),
                destination_name=doc.get("destination_name", ""),
                star_rating=doc.get("star_rating", 0),
                rating=doc.get("rating", 0.0),
                total_reviews=doc.get("total_reviews", 0),
                commission_rate=doc.get("commission_rate", 0.0),
            )
        )
    return results


async def getTopFlightRoutes(db, limit: int = 10) -> list[FlightPerformance]:
    cursor = db.flights.find(
        {},
        {
            "flight_code": 1,
            "airline": 1,
            "origin": 1,
            "destination": 1,
            "avg_occupancy_rate": 1,
            "commission_rate": 1,
            "_id": 0,
        },
    ).sort("avg_occupancy_rate", -1).limit(limit)

    results: list[FlightPerformance] = []
    async for doc in cursor:
        results.append(
            FlightPerformance(
                flight_code=doc.get("flight_code", ""),
                airline=doc.get("airline", ""),
                origin=doc.get("origin", ""),
                destination=doc.get("destination", ""),
                avg_occupancy_rate=doc.get("avg_occupancy_rate", 0.0),
                commission_rate=doc.get("commission_rate", 0.0),
            )
        )
    return results
