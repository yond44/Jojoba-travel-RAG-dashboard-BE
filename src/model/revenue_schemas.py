from __future__ import annotations

from datetime import date
from typing import Literal, Any

from pydantic import BaseModel, Field


class CurrentRevenueData(BaseModel):
    today: float
    last_seven_days: float
    current_month: float
    this_year: float


class DailyRevenueData(BaseModel):
    date: date
    revenue: float


class PeriodRevenueData(BaseModel):
    period: date
    granularity: Literal["weekly", "monthly", "annually"]
    revenue: float


class RevenueByDestination(BaseModel):
    destination_name: str
    revenue: float
    booking_count: int


class RevenueByPackageType(BaseModel):
    package_type: str
    revenue: float
    booking_count: int


class RevenueByChannel(BaseModel):
    channel: str
    revenue: float
    booking_count: int


class RevenueByPaymentMethod(BaseModel):
    payment_method: str
    revenue: float
    booking_count: int


class RevenueBySegment(BaseModel):
    customer_segment: str
    revenue: float
    booking_count: int


class AgentPerformance(BaseModel):
    agent_id: str
    revenue: float
    booking_count: int


class CustomerTypeRevenue(BaseModel):
    is_repeat_customer: bool
    revenue: float
    booking_count: int


class DiscountImpact(BaseModel):
    total_discount: float
    total_revenue: float
    discount_rate: float


class PaymentStatusSummary(BaseModel):
    payment_status: str
    total_amount: float
    booking_count: int


class ErrorResponse(BaseModel):
    status: int
    message: str


class SuccessResponse(BaseModel):
    status: int
    respond: Any