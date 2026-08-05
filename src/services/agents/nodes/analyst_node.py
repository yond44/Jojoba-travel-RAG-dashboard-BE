from __future__ import annotations

import re



from datetime import date, datetime, time, timedelta
from typing import Any, Dict

from langchain_core.runnables import RunnableConfig

from src.services.agents.state import AgentState, get_date_param
from src.utils.clock import business_today
from src.utils.log import logger

VALID_BOOKING_STATUSES = ["Completed", "Confirmed"]
DEFAULT_LOOKBACK_DAYS = 30

COMPARISON_KEYWORDS = ("bandingkan", "dibanding", "dibandingkan", "perbandingan",
                       "versus", " vs ", "sebelumnya", "periode sebelum",
                       "compare", "comparison", "selisih", "pertumbuhan",
                       "growth", "naik atau turun", "lebih tinggi",
                       "lebih rendah")

TOP_N_PATTERN = re.compile(r"\btop\s*(\d{1,2})\b|\b(\d{1,2})\s*(?:teratas|terbaik|terpopuler|terlaris)\b")
DEFAULT_TOP_N = 5
MAX_TOP_N = 25


def _requested_top_n(question_lower: str) -> int:
    match = TOP_N_PATTERN.search(question_lower)
    if not match:
        return DEFAULT_TOP_N
    value = int(match.group(1) or match.group(2))
    return min(max(value, 1), MAX_TOP_N)

# ---------- Agregasi ----------
async def _revenue_total(database, start_date, end_date) -> float:
    rows = await database["bookings"].aggregate([
        {"$match": {
            "status": {"$in": VALID_BOOKING_STATUSES},
            "payment_status": "Paid",
            "booking_date": {
                "$gte": datetime.combine(start_date, time.min),
                "$lt": datetime.combine(end_date + timedelta(days=1),
                                        time.min)}}},
        {"$group": {"_id": None, "total": {"$sum": "$total_price_idr"}}},
    ]).to_list(1)
    return float(rows[0]["total"]) if rows else 0.0


async def _channel_performance(database, start_date, end_date) -> list:
    rows = await database["bookings"].aggregate([
        {"$match": {"booking_date": {
            "$gte": datetime.combine(start_date, time.min),
            "$lt": datetime.combine(end_date + timedelta(days=1),
                                    time.min)}}},
        {"$group": {
            "_id": "$channel",
            "total_bookings": {"$sum": 1},
            "completed_bookings": {"$sum": {
                "$cond": [{"$in": ["$status", VALID_BOOKING_STATUSES]}, 1, 0]}},
            "revenue_idr": {"$sum": {
                "$cond": [{"$eq": ["$payment_status", "Paid"]},
                          "$total_price_idr", 0]}}}},
        {"$sort": {"revenue_idr": -1}},
    ]).to_list(None)
    return [{
        "channel": row["_id"],
        "total_bookings": row["total_bookings"],
        "completion_rate": round(
            row["completed_bookings"] / row["total_bookings"], 3)
        if row["total_bookings"] else 0.0,
        "revenue_idr": row["revenue_idr"],
    } for row in rows]


async def _count_bookings(database, start_date, end_date) -> int:
    return await database["bookings"].count_documents({
        "status": {"$in": VALID_BOOKING_STATUSES},
        "booking_date": {
            "$gte": datetime.combine(start_date, time.min),
            "$lt": datetime.combine(end_date + timedelta(days=1), time.min)},
    })


async def _churn_bucket_distribution(database) -> dict:
    rows = await database["ml_churn_scores"].aggregate([
        {"$group": {"_id": "$risk_bucket", "total": {"$sum": 1}}},
    ]).to_list(None)
    return {row["_id"]: row["total"] for row in rows}


async def _top_destinations(database, start_date, end_date, limit=5) -> list:
    rows = await database["bookings"].aggregate([
        {"$match": {
            "status": {"$in": VALID_BOOKING_STATUSES},
            "booking_date": {
                "$gte": datetime.combine(start_date, time.min),
                "$lt": datetime.combine(end_date + timedelta(days=1),
                                        time.min)}}},
        {"$group": {"_id": "$destination_name",
                    "bookings": {"$sum": 1},
                    "revenue": {"$sum": "$total_price_idr"}}},
        {"$sort": {"revenue": -1}},
        {"$limit": limit},
    ]).to_list(None)
    return [{"destination": row["_id"], "bookings": row["bookings"],
             "revenue_idr": row["revenue"]} for row in rows]


# ---------- Pemilihan agregasi ----------
async def _collect_facts(database, question_lower, start_date, 
                         end_date, top_n=DEFAULT_TOP_N) -> Dict[str, Any]:
    facts: Dict[str, Any] = {
        "period": {"start": str(start_date), "end": str(end_date)}}

    
    if any(word in question_lower for word in
           ("customer", "pelanggan", "pembeli", "top spender")):
        facts["top_customers"] = await _top_customers(
            database, start_date, end_date, top_n)
        
    if any(word in question_lower for word in
           ("revenue", "pendapatan", "omzet", "penjualan", "income", "sales")):
        facts["revenue_idr"] = await _revenue_total(
            database, start_date, end_date)

    if any(word in question_lower for word in
           ("channel", "kanal", "konversi", "conversion")):
        facts["channel_performance"] = await _channel_performance(
            database, start_date, end_date)

    if any(word in question_lower for word in
           ("booking", "pesanan", "transaksi", "berapa banyak")):
        facts["booking_count"] = await _count_bookings(
            database, start_date, end_date)

    if any(word in question_lower for word in
           ("churn", "risiko", "pelanggan pergi")):
        facts["churn_distribution"] = await _churn_bucket_distribution(database)

    if any(word in question_lower for word in
           ("destinasi", "destination", "tujuan", "paket")):
        facts["top_destinations"] = await _top_destinations(database, start_date, end_date, top_n)

    if len(facts) == 1:
        facts["booking_count"] = await _count_bookings(
            database, start_date, end_date)
        facts["revenue_idr"] = await _revenue_total(
            database, start_date, end_date)

    return facts


async def _top_customers(database, start_date, end_date, limit) -> list:
    rows = await database["bookings"].aggregate([
        {"$match": {
            "status": {"$in": VALID_BOOKING_STATUSES},
            "payment_status": "Paid",
            "booking_date": {
                "$gte": datetime.combine(start_date, time.min),
                "$lt": datetime.combine(end_date + timedelta(days=1),
                                        time.min)}}},
        {"$group": {"_id": "$customer_id",
                    "customer_name": {"$first": "$customer_name"},
                    "segment": {"$first": "$customer_segment"},
                    "total_spent_idr": {"$sum": "$total_price_idr"},
                    "booking_count": {"$sum": 1}}},
        {"$sort": {"total_spent_idr": -1}},
        {"$limit": limit},
    ]).to_list(None)
    return [{"customer_name": row["customer_name"],
             "segment": row["segment"],
             "total_spent_idr": row["total_spent_idr"],
             "booking_count": row["booking_count"]} for row in rows]

# ---------- Periode pembanding ----------
def _previous_period(start_date: date, end_date: date) -> tuple[date, date]:
    span_days = (end_date - start_date).days
    previous_end = start_date - timedelta(days=1)
    previous_start = previous_end - timedelta(days=span_days)
    return previous_start, previous_end


def _growth_percent(current: float, previous: float) -> float | None:
    if not previous:
        return None
    return round((current - previous) / previous * 100, 2)


def _build_comparison(current: Dict[str, Any],
                      previous: Dict[str, Any]) -> Dict[str, Any]:
    comparison: Dict[str, Any] = {}
    for key, current_value in current.items():
        if key == "period" or not isinstance(current_value, (int, float)):
            continue
        previous_value = previous.get(key)
        if not isinstance(previous_value, (int, float)):
            continue
        comparison[key] = {
            "current": current_value,
            "previous": previous_value,
            "change": current_value - previous_value,
            "change_pct": _growth_percent(current_value, previous_value),
        }
    return comparison


# ---------- Node ----------
async def analyst_node(state: AgentState, config: RunnableConfig) -> dict:
    database = config.get("configurable", {}).get("database")
    params = state.get("params", {})
    original_question = state.get("original_question", "")
    standalone_question = state.get("standalone_question", "")
    question_lower = f"{original_question} {standalone_question}".lower()
    tool_results = dict(state.get("tool_results", {}))
    tools_used = list(state.get("tools_used", []))
    next_hop_count = state.get("hop_count", 0) + 1
    top_n = _requested_top_n(question_lower)

    if database is None:
        return {"error_message": "Koneksi database tidak tersedia",
                "hop_count": next_hop_count}

    today = business_today()
    start_date = get_date_param(
        params, "start_date", today - timedelta(days=DEFAULT_LOOKBACK_DAYS))
    end_date = get_date_param(params, "end_date", today)

    if start_date > today:
        span_days = (end_date - start_date).days
        end_date = today
        start_date = today - timedelta(days=max(span_days, 1))
        logger.info("Rentang masa depan dipotong ke historis: %s s/d %s",
                    start_date, end_date)
    elif end_date > today:
        end_date = today

    facts = await _collect_facts(database, question_lower, start_date, end_date, top_n)
    tool_results["facts"] = facts
    tools_used.append("query_database")

    wants_comparison = any(word in question_lower
                           for word in COMPARISON_KEYWORDS)
    if wants_comparison:
        previous_start, previous_end = _previous_period(start_date, end_date)
        previous_facts = await _collect_facts(
            database, question_lower, previous_start, previous_end, top_n)
        tool_results["facts_previous_period"] = previous_facts
        tool_results["comparison"] = _build_comparison(facts, previous_facts)
        tools_used.append("query_database_previous_period")
        logger.info("Perbandingan dijalankan: %s s/d %s vs %s s/d %s",
                    start_date, end_date, previous_start, previous_end)

    logger.info("Analyst mengumpulkan %d fakta%s", len(facts) - 1,
                " + pembanding" if wants_comparison else "")

    return {"tool_results": tool_results, "tools_used": tools_used,
            "hop_count": next_hop_count}