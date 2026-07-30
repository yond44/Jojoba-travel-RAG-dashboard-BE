from __future__ import annotations

from pydantic import BaseModel


class CampaignPerformance(BaseModel):
    campaign_code: str
    name: str
    type: str
    spend_idr: float
    revenue_generated_idr: float
    roi: float
    conversion_rate: float


class CampaignTypeSummary(BaseModel):
    type: str
    total_spend_idr: float
    total_revenue_idr: float
    avg_roi: float


class CampaignFunnel(BaseModel):
    campaign_code: str
    reach: int
    impressions: int
    clicks: int
    conversions: int
