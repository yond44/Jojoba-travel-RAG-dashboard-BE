from __future__ import annotations

from datetime import date

from pydantic import BaseModel


class SegmentDistribution(BaseModel):
    segment: str
    customer_count: int
    total_spent_idr: float


class AgeGroupDistribution(BaseModel):
    age_group: str
    customer_count: int


class CityDistribution(BaseModel):
    city: str
    customer_count: int


class AcquisitionChannelDistribution(BaseModel):
    acquisition_channel: str
    customer_count: int


class TopCustomer(BaseModel):
    customer_code: str
    name: str
    segment: str
    total_trips: int
    total_spent_idr: float


class NewCustomersByDay(BaseModel):
    date: date
    new_customer_count: int


class RepeatCustomerRatio(BaseModel):
    repeat_customer_count: int
    new_customer_count: int
    repeat_rate: float
