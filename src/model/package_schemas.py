from __future__ import annotations

from pydantic import BaseModel


class PackagePerformance(BaseModel):
    package_code: str
    name: str
    destination_name: str
    type: str
    nights: int
    price_per_pax_idr: float
    rating: float
    total_bookings: int


class PackageTypeSummary(BaseModel):
    type: str
    package_count: int
    total_bookings: int
    avg_rating: float
    avg_price_per_pax_idr: float
