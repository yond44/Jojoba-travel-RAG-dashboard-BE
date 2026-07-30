from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel


class SentimentSummary(BaseModel):
    sentiment: str
    review_count: int
    avg_rating: float


class RatingTrend(BaseModel):
    period: date
    granularity: Literal["weekly", "monthly", "annually"]
    avg_rating: float
    review_count: int


class DestinationRatingSummary(BaseModel):
    destination_name: str
    avg_rating: float
    review_count: int
