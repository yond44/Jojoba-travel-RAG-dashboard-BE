from __future__ import annotations

from datetime import datetime, time, timedelta

from langchain_core.runnables import RunnableConfig

from src.services.agents.state import AgentState, get_date_param
from src.utils.clock import business_today
from src.utils.log import logger
from src.services.agents.tool_registry import (
    match_tools_by_keyword, run_selected_tools)

VALID_BOOKING_STATUSES = ["Completed", "Confirmed"]
DEFAULT_LOOKBACK_DAYS = 30

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


async def analyst_node(state: AgentState,
                       config: RunnableConfig) -> dict:
    database = config.get("configurable", {}).get("database")
    params = state.get("params", {})
    question_lower = state.get("standalone_question", "").lower()
    tool_results = dict(state.get("tool_results", {}))
    tools_used = list(state.get("tools_used", []))

    if database is None:
        return {"error_message": "Koneksi database tidak tersedia",
                "hop_count": state.get("hop_count", 0) + 1}

    today = business_today()
    start_date = get_date_param(
        params, "start_date",
        today - timedelta(days=DEFAULT_LOOKBACK_DAYS))
    end_date = get_date_param(params, "end_date", today)

    facts = {"period": {"start": str(start_date), "end": str(end_date)}}
    
    selected_tool_ids = params.get("tools") or match_tools_by_keyword(
        f"{state.get('original_question', '')} {question_lower}")
    if selected_tool_ids:
        registry_results = await run_selected_tools(
            selected_tool_ids, database, start_date, end_date)
        tool_results.update(registry_results)
        tools_used.extend(registry_results.keys())
    
    if any(word in question_lower for word in
           ("revenue", "pendapatan", "omzet", "penjualan")):
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
        facts["top_destinations"] = await _top_destinations(
            database, start_date, end_date)

    if len(facts) == 1:  # hanya period -> tidak ada agregasi yang cocok
        facts["booking_count"] = await _count_bookings(
            database, start_date, end_date)

    logger.info("Analyst mengumpulkan %d fakta", len(facts) - 1)
    tool_results["facts"] = facts
    tools_used.append("query_database")

    return {"tool_results": tool_results, "tools_used": tools_used,
            "hop_count": state.get("hop_count", 0) + 1}


