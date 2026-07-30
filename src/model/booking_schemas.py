from __future__ import annotations

from datetime import date

from pydantic import BaseModel


class BookingStatusSummary(BaseModel):
    status: str
    booking_count: int


class BookingVolumeByDay(BaseModel):
    date: date
    booking_count: int


class LeadTimeStats(BaseModel):
    avg_lead_time_days: float
    min_lead_time_days: int
    max_lead_time_days: int


class CancellationRate(BaseModel):
    total_bookings: int
    cancelled_bookings: int
    cancellation_rate: float


class PaymentGatewayPerformance(BaseModel):
    payment_method: str
    success_count: int
    failed_count: int
    total_amount_idr: float
