from __future__ import annotations

from pydantic import BaseModel


class HotelPerformance(BaseModel):
    hotel_code: str
    name: str
    destination_name: str
    star_rating: int
    rating: float
    total_reviews: int
    commission_rate: float


class FlightPerformance(BaseModel):
    flight_code: str
    airline: str
    origin: str
    destination: str
    avg_occupancy_rate: float
    commission_rate: float
