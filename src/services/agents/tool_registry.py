from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from enum import Enum
from typing import Any, Awaitable, Callable, Dict, FrozenSet, List, Optional

from src.services.booking.booking import (
    getBookingStatusSummary, getBookingVolume, getCancellationRate,
    getLeadTimeStats, getPaymentGatewayPerformance)
from src.services.campaign.campaign import (
    getCampaignFunnel, getCampaignPerformance, getCampaignTypeSummary)
from src.services.customer.customer import (
    getAcquisitionChannelDistribution, getAgeGroupDistribution,
    getCityDistribution, getNewCustomersByDay, getRepeatCustomerRatio,
    getSegmentDistribution, getTopCustomers, searchCustomers, getChurnRiskList)
from src.services.destination.destination import (
    getDestinationPopularityVsBookings, getRegionSummary)
from src.services.package.package import getPackageTypeSummary, getTopPackages
from src.services.partner.partner import getTopFlightRoutes, getTopHotels
from src.services.review.review import (
    getDestinationRatingSummary, getRatingTrend, getSentimentSummary)
from src.services.revenue.revenue import (
    getAgentPerformance, getCurrentRevenue, getCustomerTypeRevenue,
    getDailyRevenue, getDiscountImpact, getPaymentStatusSummary,
    getPeriodRevenue, getRevenueByChannel, getRevenueByDestination,
    getRevenueByPackageType, getRevenueByPaymentMethod, getRevenueBySegment)
from src.utils.log import logger

DEFAULT_LIMIT = 10
MAX_TOOLS_PER_TURN = 3


class ToolArgument(str, Enum):
    DATE_RANGE = "date_range"
    LIMIT = "limit"
    GRANULARITY = "granularity"


@dataclass(frozen=True)
class ToolSpec:
    tool_id: str
    description: str
    handler: Callable[..., Awaitable[Any]]
    arguments: FrozenSet[ToolArgument] = frozenset()
    keywords: tuple[str, ...] = ()


# ---------- Katalog tool ----------
TOOL_SPECS: tuple[ToolSpec, ...] = (
    ToolSpec("revenue_current",
             "Revenue hari ini, 7 hari terakhir, bulan ini, dan tahun ini",
             getCurrentRevenue,
             keywords=("revenue hari ini", "revenue sekarang", "ringkasan revenue")),
    ToolSpec("revenue_by_period",
             "Revenue aktual dikelompokkan per periode (harian/mingguan/bulanan)",
             getPeriodRevenue,
             frozenset({ToolArgument.DATE_RANGE, ToolArgument.GRANULARITY}),
             ("tren revenue", "revenue per bulan", "revenue per minggu")),
    ToolSpec("revenue_daily",
             "Revenue aktual per hari dalam satu rentang tanggal",
             getDailyRevenue, frozenset({ToolArgument.DATE_RANGE}),
             ("revenue harian", "revenue per hari")),
    ToolSpec("revenue_by_destination",
             "Revenue dipecah per destinasi wisata",
             getRevenueByDestination, frozenset({ToolArgument.DATE_RANGE}),
             ("revenue destinasi", "destinasi terlaris")),
    ToolSpec("revenue_by_package_type",
             "Revenue dipecah per tipe paket perjalanan",
             getRevenueByPackageType, frozenset({ToolArgument.DATE_RANGE}),
             ("tipe paket", "revenue paket")),
    ToolSpec("revenue_by_channel",
             "Revenue dipecah per kanal booking",
             getRevenueByChannel, frozenset({ToolArgument.DATE_RANGE}),
             ("kanal", "channel", "revenue kanal")),
    ToolSpec("revenue_by_payment_method",
             "Revenue dipecah per metode pembayaran",
             getRevenueByPaymentMethod, frozenset({ToolArgument.DATE_RANGE}),
             ("metode pembayaran", "payment method")),
    ToolSpec("revenue_by_segment",
             "Revenue dipecah per segmen pelanggan",
             getRevenueBySegment, frozenset({ToolArgument.DATE_RANGE}),
             ("revenue segmen", "segmen pelanggan")),
    ToolSpec("agent_performance",
             "Pencapaian target dan kontribusi revenue per agen penjualan",
             getAgentPerformance, frozenset({ToolArgument.DATE_RANGE}),
             ("agen", "agent", "target agen")),
    ToolSpec("customer_type_revenue",
             "Revenue dari pelanggan baru dibanding pelanggan berulang",
             getCustomerTypeRevenue, frozenset({ToolArgument.DATE_RANGE}),
             ("pelanggan baru", "repeat", "customer type")),
    ToolSpec("discount_impact",
             "Dampak diskon terhadap revenue dan volume booking",
             getDiscountImpact, frozenset({ToolArgument.DATE_RANGE}),
             ("diskon", "discount", "potongan harga")),
    ToolSpec("payment_status_summary",
             "Ringkasan status pembayaran booking",
             getPaymentStatusSummary, frozenset({ToolArgument.DATE_RANGE}),
             ("status pembayaran", "belum bayar", "refund")),

    ToolSpec("customer_segment_distribution",
             "Jumlah pelanggan per segmen",
             getSegmentDistribution,
             keywords=("distribusi segmen", "komposisi segmen")),
    ToolSpec("customer_age_distribution",
             "Jumlah pelanggan per kelompok usia",
             getAgeGroupDistribution,
             keywords=("usia", "umur", "age group")),
    ToolSpec("customer_city_distribution",
             "Kota asal pelanggan terbanyak",
             getCityDistribution, frozenset({ToolArgument.LIMIT}),
             ("kota", "city", "asal pelanggan")),
    ToolSpec("customer_acquisition_channel",
             "Jumlah pelanggan per kanal akuisisi",
             getAcquisitionChannelDistribution,
             keywords=("akuisisi", "acquisition channel")),
    ToolSpec("customer_top_spenders",
             "Pelanggan dengan total belanja tertinggi",
             getTopCustomers, frozenset({ToolArgument.LIMIT}),
             ("pelanggan terbaik", "top customer", "belanja tertinggi")),
    ToolSpec("customer_new_by_day",
             "Jumlah pelanggan baru per hari",
             getNewCustomersByDay, frozenset({ToolArgument.DATE_RANGE}),
             ("pelanggan baru per hari", "pertumbuhan pelanggan")),
    ToolSpec("customer_repeat_ratio",
             "Rasio pelanggan berulang terhadap seluruh pelanggan",
             getRepeatCustomerRatio,
             keywords=("rasio repeat", "loyalitas")),

    ToolSpec("booking_status_summary",
             "Jumlah booking per status",
             getBookingStatusSummary, frozenset({ToolArgument.DATE_RANGE}),
             ("status booking", "booking selesai", "booking batal")),
    ToolSpec("booking_volume",
             "Volume booking per hari",
             getBookingVolume, frozenset({ToolArgument.DATE_RANGE}),
             ("volume booking", "jumlah booking")),
    ToolSpec("booking_lead_time",
             "Jarak waktu antara pemesanan dan tanggal perjalanan",
             getLeadTimeStats, frozenset({ToolArgument.DATE_RANGE}),
             ("lead time", "jarak pesan")),
    ToolSpec("booking_cancellation_rate",
             "Tingkat pembatalan booking",
             getCancellationRate, frozenset({ToolArgument.DATE_RANGE}),
             ("pembatalan", "cancel", "batal")),
    ToolSpec("booking_payment_gateway",
             "Kinerja tiap payment gateway",
             getPaymentGatewayPerformance, frozenset({ToolArgument.DATE_RANGE}),
             ("payment gateway", "gerbang pembayaran")),

    ToolSpec("campaign_performance",
             "Kinerja tiap kampanye: biaya, konversi, dan CPA",
             getCampaignPerformance, frozenset({ToolArgument.DATE_RANGE}),
             ("kampanye", "campaign", "cpa")),
    ToolSpec("campaign_type_summary",
             "Ringkasan kinerja per tipe kampanye",
             getCampaignTypeSummary, frozenset({ToolArgument.DATE_RANGE}),
             ("tipe kampanye", "jenis kampanye")),
    ToolSpec("campaign_funnel",
             "Funnel kampanye dari impresi sampai konversi",
             getCampaignFunnel, frozenset({ToolArgument.DATE_RANGE}),
             ("funnel", "konversi kampanye")),

    ToolSpec("destination_popularity",
             "Perbandingan skor popularitas destinasi dengan booking nyata",
             getDestinationPopularityVsBookings,
             frozenset({ToolArgument.DATE_RANGE}),
             ("popularitas destinasi", "destinasi populer")),
    ToolSpec("destination_region_summary",
             "Ringkasan destinasi per wilayah",
             getRegionSummary,
             keywords=("wilayah", "region", "provinsi")),

    ToolSpec("review_sentiment_summary",
             "Ringkasan sentimen ulasan pelanggan",
             getSentimentSummary, frozenset({ToolArgument.DATE_RANGE}),
             ("sentimen", "ulasan", "review")),
    ToolSpec("review_rating_trend",
             "Tren rating ulasan sepanjang waktu",
             getRatingTrend,
             frozenset({ToolArgument.DATE_RANGE, ToolArgument.GRANULARITY}),
             ("tren rating", "rating per bulan")),
    ToolSpec("review_destination_rating",
             "Rating rata-rata per destinasi",
             getDestinationRatingSummary,
             frozenset({ToolArgument.DATE_RANGE, ToolArgument.LIMIT}),
             ("rating destinasi", "destinasi terbaik")),

    ToolSpec("partner_top_hotels",
             "Hotel mitra dengan kontribusi tertinggi",
             getTopHotels, frozenset({ToolArgument.LIMIT}),
             ("hotel", "penginapan")),
    ToolSpec("partner_top_flight_routes",
             "Rute penerbangan dengan volume tertinggi",
             getTopFlightRoutes, frozenset({ToolArgument.LIMIT}),
             ("penerbangan", "rute", "flight")),

    ToolSpec("package_top_packages",
             "Paket perjalanan terlaris",
             getTopPackages, frozenset({ToolArgument.LIMIT}),
             ("paket terlaris", "top package")),
    ToolSpec("package_type_summary",
             "Ringkasan paket per tipe",
             getPackageTypeSummary,
             keywords=("ringkasan paket", "jenis paket")),
    ToolSpec("customer_search",
             "Cari pelanggan berdasarkan nama atau kode",
             searchCustomers, frozenset({ToolArgument.LIMIT}),
             ("cari pelanggan", "siapa pelanggan")),
    ToolSpec("customer_churn_risk",
             "Daftar pelanggan dengan risiko churn tertinggi beserta namanya",
             getChurnRiskList, frozenset({ToolArgument.LIMIT}),
             ("pelanggan berisiko", "siapa yang akan churn", "risiko tinggi")),
)

TOOLS_BY_ID: Dict[str, ToolSpec] = {spec.tool_id: spec for spec in TOOL_SPECS}
VALID_TOOL_IDS: frozenset[str] = frozenset(TOOLS_BY_ID)


# ---------- Deskripsi untuk prompt ----------
def describe_tools() -> str:
    return "\n".join(f"- {spec.tool_id}: {spec.description}"
                     for spec in TOOL_SPECS)


# ---------- Pemilihan cadangan tanpa LLM ----------
def match_tools_by_keyword(question: str,
                           max_tools: int = MAX_TOOLS_PER_TURN) -> List[str]:
    question_lower = question.lower()
    matched: List[str] = []
    for spec in TOOL_SPECS:
        if any(keyword in question_lower for keyword in spec.keywords):
            matched.append(spec.tool_id)
        if len(matched) >= max_tools:
            break
    return matched


# ---------- Eksekusi ----------
async def run_tool(tool_id: str, database: Any, start_date: date,
                   end_date: date, limit: int = DEFAULT_LIMIT,
                   granularity: str = "monthly") -> Any:
    spec = TOOLS_BY_ID.get(tool_id)
    if spec is None:
        raise KeyError(f"Tool tidak dikenal: {tool_id}")

    keyword_arguments: Dict[str, Any] = {}
    if ToolArgument.DATE_RANGE in spec.arguments:
        keyword_arguments["start_date"] = start_date
        keyword_arguments["end_date"] = end_date
    if ToolArgument.LIMIT in spec.arguments:
        keyword_arguments["limit"] = limit
    if ToolArgument.GRANULARITY in spec.arguments:
        keyword_arguments["granularity"] = granularity

    result = await spec.handler(database, **keyword_arguments)
    return _to_plain(result)


def _to_plain(value: Any) -> Any:
    if isinstance(value, list):
        return [_to_plain(item) for item in value]
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        return dump(mode="json")
    return value


async def run_selected_tools(tool_ids: List[str], database: Any,
                             start_date: date, end_date: date,
                             limit: int = DEFAULT_LIMIT,
                             granularity: str = "monthly"
                             ) -> Dict[str, Any]:
    collected: Dict[str, Any] = {}
    for tool_id in tool_ids[:MAX_TOOLS_PER_TURN]:
        if tool_id not in VALID_TOOL_IDS:
            logger.warning("Tool diabaikan karena tidak dikenal: %s", tool_id)
            continue
        try:
            collected[tool_id] = await run_tool(
                tool_id, database, start_date, end_date, limit, granularity)
        except Exception as error:
            logger.exception("Tool %s gagal: %s", tool_id, error)
            collected[tool_id] = {"failed": str(error)}
    return collected
