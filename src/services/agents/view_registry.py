from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

DASHBOARD_ROOT = "/dashboard"


@dataclass(frozen=True)
class ViewSpec:
    view_id: str
    label: str
    dashboard_path: str
    api_path: str
    keywords: tuple[str, ...]
    tool_id: Optional[str] = None
    accepts_date_range: bool = True
    extra_query_params: Dict[str, str] = field(default_factory=dict)


# ---------- Katalog halaman ----------
VIEW_SPECS: tuple[ViewSpec, ...] = (
    ViewSpec("overview", "Ringkasan Bisnis",
             f"{DASHBOARD_ROOT}/overview", "/api/v1/current",
             ("ringkasan", "overview", "dashboard utama", "halaman depan",
              "ikhtisar"),
             "revenue_current", False),
    ViewSpec("revenue_trend", "Tren Revenue",
             f"{DASHBOARD_ROOT}/revenue/trend", "/api/v1/period",
             ("tren revenue", "revenue bulanan", "grafik revenue",
              "pertumbuhan revenue"),
             "revenue_by_period", True, {"granularity": "monthly"}),
    ViewSpec("revenue_destination", "Revenue per Destinasi",
             f"{DASHBOARD_ROOT}/revenue/destination", "/api/v1/by-destination",
             ("revenue destinasi", "destinasi terlaris", "per destinasi"),
             "revenue_by_destination"),
    ViewSpec("revenue_channel", "Revenue per Kanal",
             f"{DASHBOARD_ROOT}/revenue/channel", "/api/v1/by-channel",
             ("kanal", "channel", "revenue kanal"),
             "revenue_by_channel"),
    ViewSpec("revenue_segment", "Revenue per Segmen",
             f"{DASHBOARD_ROOT}/revenue/segment", "/api/v1/by-segment",
             ("revenue segmen", "segmen pelanggan"),
             "revenue_by_segment"),
    ViewSpec("revenue_discount", "Dampak Diskon",
             f"{DASHBOARD_ROOT}/revenue/discount", "/api/v1/discount-impact",
             ("diskon", "discount", "potongan"),
             "discount_impact"),
    ViewSpec("agent_performance", "Kinerja Agen",
             f"{DASHBOARD_ROOT}/agents", "/api/v1/agent-performance",
             ("agen", "agent", "kinerja agen", "target agen"),
             "agent_performance"),
    ViewSpec("bookings", "Analisis Booking",
             f"{DASHBOARD_ROOT}/bookings", "/api/v1/bookings/volume",
             ("booking", "pesanan", "volume booking"),
             "booking_volume"),
    ViewSpec("booking_cancellation", "Pembatalan Booking",
             f"{DASHBOARD_ROOT}/bookings/cancellation",
             "/api/v1/bookings/cancellation-rate",
             ("pembatalan", "cancel", "batal"),
             "booking_cancellation_rate"),
    ViewSpec("customers", "Analisis Pelanggan",
             f"{DASHBOARD_ROOT}/customers",
             "/api/v1/customers/segment-distribution",
             ("pelanggan", "customer", "segmentasi pelanggan"),
             "customer_segment_distribution", False),
    ViewSpec("customer_top", "Pelanggan Teratas",
             f"{DASHBOARD_ROOT}/customers/top", "/api/v1/customers/top",
             ("pelanggan terbaik", "top customer", "belanja tertinggi"),
             "customer_top_spenders", False),
    ViewSpec("churn", "Risiko Churn",
             f"{DASHBOARD_ROOT}/churn", "/api/v1/predict/churn",
             ("churn", "risiko pelanggan", "retensi"),
             None, False),
    ViewSpec("forecast", "Proyeksi Revenue",
             f"{DASHBOARD_ROOT}/forecast", "/api/v1/revenue",
             ("forecast", "proyeksi", "prediksi revenue", "ramalan"),
             None),
    ViewSpec("campaigns", "Kinerja Kampanye",
             f"{DASHBOARD_ROOT}/campaigns", "/api/v1/campaigns/performance",
             ("kampanye", "campaign", "cpa", "iklan"),
             "campaign_performance"),
    ViewSpec("destinations", "Destinasi",
             f"{DASHBOARD_ROOT}/destinations",
             "/api/v1/destinations/popularity-vs-bookings",
             ("destinasi", "destination", "wilayah"),
             "destination_popularity"),
    ViewSpec("packages", "Paket Perjalanan",
             f"{DASHBOARD_ROOT}/packages", "/api/v1/packages/top",
             ("paket", "package", "paket terlaris"),
             "package_top_packages", False),
    ViewSpec("partners", "Mitra Hotel & Penerbangan",
             f"{DASHBOARD_ROOT}/partners", "/api/v1/partners/hotels/top",
             ("hotel", "penerbangan", "mitra", "partner"),
             "partner_top_hotels", False),
    ViewSpec("reviews", "Ulasan Pelanggan",
             f"{DASHBOARD_ROOT}/reviews", "/api/v1/reviews/sentiment-summary",
             ("ulasan", "review", "sentimen", "rating"),
             "review_sentiment_summary"),
)

VIEWS_BY_ID: Dict[str, ViewSpec] = {spec.view_id: spec for spec in VIEW_SPECS}
VALID_VIEW_IDS: frozenset[str] = frozenset(VIEWS_BY_ID)


# ---------- Deskripsi untuk prompt ----------
def describe_views() -> str:
    return "\n".join(f"- {spec.view_id}: {spec.label}"
                     for spec in VIEW_SPECS)


# ---------- Pencocokan cadangan tanpa LLM ----------
def match_view_by_keyword(question: str) -> Optional[ViewSpec]:
    question_lower = question.lower()
    best_spec: Optional[ViewSpec] = None
    best_length = 0
    for spec in VIEW_SPECS:
        for keyword in spec.keywords:
            if keyword in question_lower and len(keyword) > best_length:
                best_spec = spec
                best_length = len(keyword)
    return best_spec


def nearby_views(spec: ViewSpec, limit: int = 3) -> List[ViewSpec]:
    prefix = spec.view_id.split("_")[0]
    return [candidate for candidate in VIEW_SPECS
            if candidate.view_id != spec.view_id
            and candidate.view_id.startswith(prefix)][:limit]
