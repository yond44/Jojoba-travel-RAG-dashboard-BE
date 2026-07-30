from __future__ import annotations

from pydantic import BaseModel


class DestinationPopularity(BaseModel):
    destination_name: str
    popularity_score: float
    actual_booking_count: int
    actual_revenue_idr: float


class RegionSummary(BaseModel):
    region: str
    destination_count: int
    avg_rating: float
