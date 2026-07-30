from __future__ import annotations

from datetime import datetime, time, timedelta
from typing import Any, Dict, List

from langchain_core.runnables import RunnableConfig

from src.services.agents.state import AgentState, get_date_param
from src.utils.clock import business_today
from src.utils.log import logger

VALID_BOOKING_STATUSES = ["Completed", "Confirmed"]
DEFAULT_EDA_LOOKBACK_DAYS = 365
MONTHLY_TREND_LIMIT = 24


async def _monthly_revenue_trend(database, start_date, end_date) -> List[dict]:
    rows = await database["bookings"].aggregate([
        {"$match": {
            "status": {"$in": VALID_BOOKING_STATUSES},
            "payment_status": "Paid",
            "booking_date": {
                "$gte": datetime.combine(start_date, time.min),
                "$lt": datetime.combine(end_date + timedelta(days=1),
                                        time.min)}}},
        {"$group": {
            "_id": {"year": {"$year": "$booking_date"},
                    "month": {"$month": "$booking_date"}},
            "revenue_idr": {"$sum": "$total_price_idr"},
            "booking_count": {"$sum": 1}}},
        {"$sort": {"_id.year": 1, "_id.month": 1}},
    ]).to_list(None)
    return [{
        "period": f"{row['_id']['year']}-{row['_id']['month']:02d}",
        "revenue_idr": row["revenue_idr"],
        "booking_count": row["booking_count"],
    } for row in rows][-MONTHLY_TREND_LIMIT:]


async def _booking_value_distribution(database, start_date,
                                      end_date) -> Dict[str, Any]:
    rows = await database["bookings"].aggregate([
        {"$match": {
            "status": {"$in": VALID_BOOKING_STATUSES},
            "booking_date": {
                "$gte": datetime.combine(start_date, time.min),
                "$lt": datetime.combine(end_date + timedelta(days=1),
                                        time.min)}}},
        {"$group": {
            "_id": None,
            "median_value": {"$median": {
                "input": "$total_price_idr", "method": "approximate"}},
            "percentiles": {"$percentile": {
                "input": "$total_price_idr", "p": [0.25, 0.75, 0.95],
                "method": "approximate"}},
            "min_value": {"$min": "$total_price_idr"},
            "max_value": {"$max": "$total_price_idr"},
            "total_bookings": {"$sum": 1}}},
    ]).to_list(1)
    if not rows:
        return {}
    row = rows[0]
    percentiles = row.get("percentiles") or [0, 0, 0]
    return {
        "total_bookings": row["total_bookings"],
        "median_idr": row.get("median_value"),
        "p25_idr": percentiles[0],
        "p75_idr": percentiles[1],
        "p95_idr": percentiles[2],
        "min_idr": row["min_value"],
        "max_idr": row["max_value"],
    }


async def _segment_composition(database) -> List[dict]:
    rows = await database["customers"].aggregate([
        {"$group": {"_id": "$segment",
                    "customers": {"$sum": 1},
                    "median_spend_idr": {"$median": {
                        "input": "$total_spent_idr",
                        "method": "approximate"}}}},
        {"$sort": {"customers": -1}},
    ]).to_list(None)
    return [{"segment": row["_id"], "customers": row["customers"],
             "median_spend_idr": row["median_spend_idr"]} for row in rows]


async def _data_quality_snapshot(database) -> Dict[str, Any]:
    total_bookings = await database["bookings"].count_documents({})
    missing_campaign = await database["bookings"].count_documents(
        {"campaign_id": None})
    total_customers = await database["customers"].count_documents({})
    return {
        "total_bookings": total_bookings,
        "total_customers": total_customers,
        "organic_bookings_no_campaign": missing_campaign,
        "organic_share": (round(missing_campaign / total_bookings, 3)
                          if total_bookings else 0.0),
    }


async def eda_node(state: AgentState, config: RunnableConfig) -> dict:
    database = config.get("configurable", {}).get("database")
    params = state.get("params", {})
    question_lower = state.get("standalone_question", "").lower()
    tool_results = dict(state.get("tool_results", {}))
    tools_used = list(state.get("tools_used", []))
    next_hop_count = state.get("hop_count", 0) + 1

    if database is None:
        return {"error_message": "Koneksi database tidak tersedia",
                "hop_count": next_hop_count}

    today = business_today()
    start_date = get_date_param(
        params, "start_date",
        today - timedelta(days=DEFAULT_EDA_LOOKBACK_DAYS))
    end_date = get_date_param(params, "end_date", today)
    if end_date > today:          # EDA hanya atas data yang sudah ada
        end_date = today

    analysis: Dict[str, Any] = {
        "period": {"start": str(start_date), "end": str(end_date)},
        "method_note": ("Statistik memakai median dan kuartil karena "
                        "sebaran nilai transaksi condong ke kanan; "
                        "rata-rata akan menyesatkan sebagai nilai tipikal."),
    }

    if any(word in question_lower for word in
           ("tren", "trend", "pertumbuhan", "growth", "bulanan")):
        analysis["monthly_revenue_trend"] = await _monthly_revenue_trend(
            database, start_date, end_date)

    if any(word in question_lower for word in
           ("sebaran", "distribusi", "distribution", "nilai transaksi",
            "rata-rata", "median")):
        analysis["booking_value_distribution"] = \
            await _booking_value_distribution(database, start_date, end_date)

    if any(word in question_lower for word in
           ("segmen", "segment", "komposisi", "pelanggan")):
        analysis["segment_composition"] = await _segment_composition(database)

    if any(word in question_lower for word in
           ("kualitas data", "data quality", "bersih", "lengkap")):
        analysis["data_quality"] = await _data_quality_snapshot(database)

    if len(analysis) == 2:
        analysis["monthly_revenue_trend"] = await _monthly_revenue_trend(
            database, start_date, end_date)
        analysis["booking_value_distribution"] = \
            await _booking_value_distribution(database, start_date, end_date)

    logger.info("EDA menjalankan %d analisis", len(analysis) - 2)
    tool_results["eda"] = analysis
    tools_used.append("explore_data")

    return {"tool_results": tool_results, "tools_used": tools_used,
            "hop_count": next_hop_count}
