from __future__ import annotations

import re
from datetime import timedelta

from langchain_core.runnables import RunnableConfig

from src.services.agents.state import AgentState, get_date_param
from src.services.ml.ml_services import (
    CustomerHistoryNotFoundError, predict_churn)
from src.services.ml.revenue_resolver import resolve_revenue
from src.utils.clock import business_today
from src.utils.log import logger


DEFAULT_LOOKBACK_DAYS = 30

CHURN_KEYWORDS = ("churn", "risiko", "berisiko", "retensi", "akan pergi")
NAME_PATTERN = re.compile(
    r"(?:bernama|nama(?:nya)?|customer|pelanggan)\s+([A-Z][a-zA-Z']+(?:\s+[A-Z][a-zA-Z']+)?)")
MAX_NAME_MATCHES = 10


async def _resolve_customer_ids_by_name(database, question: str,
                                        limit: int) -> list[str]:
    match = NAME_PATTERN.search(question)
    if not match:
        return []
    rows = await database["customers"].find(
        {"name": {"$regex": re.escape(match.group(1)), "$options": "i"}},
        {"_id": 1}).limit(limit).to_list(None)
    return [str(row["_id"]) for row in rows]

async def ml_inference_node(state: AgentState,
                            config: RunnableConfig) -> dict:
    database = config.get("configurable", {}).get("database")
    params = state.get("params", {})
    tool_results = dict(state.get("tool_results", {}))
    tools_used = list(state.get("tools_used", []))
    contains_forecast = state.get("contains_forecast", False)

    if database is None:
        return {"error_message": "Koneksi database tidak tersedia",
                "hop_count": state.get("hop_count", 0) + 1}

    customer_ids = params.get("customer_ids") or []
    question = f"{state.get('original_question', '')} " \
               f"{state.get('standalone_question', '')}"
    asks_about_churn = any(word in question.lower() for word in CHURN_KEYWORDS)
    
    if not customer_ids and asks_about_churn:
        customer_ids = await _resolve_customer_ids_by_name(
            database, question, MAX_NAME_MATCHES)

    if customer_ids:
        try:
            churn_response = await predict_churn(database,
                                                 customer_ids=customer_ids)
            tool_results["churn"] = churn_response.model_dump(mode="json")
            tools_used.append("predict_churn")
        except CustomerHistoryNotFoundError as error:
            logger.info("Churn tidak bisa dihitung: %s", error)
            tool_results["churn"] = {"not_found": str(error)}
            tools_used.append("predict_churn")
    elif asks_about_churn:
        tool_results["churn"] = {
            "unavailable": "Pertanyaan menyangkut churn tetapi tidak ada "
                           "pelanggan yang bisa diidentifikasi dari "
                           "pertanyaan. Minta pengguna menyebut nama atau "
                           "kode pelanggan, atau arahkan ke halaman Risiko "
                           "Churn untuk daftar pelanggan berisiko."}
        tools_used.append("predict_churn")
    else:
        today = business_today()
        start_date = get_date_param(
            params, "start_date",
            today - timedelta(days=DEFAULT_LOOKBACK_DAYS))
        end_date = get_date_param(params, "end_date", today)
        try:
            revenue_result = await resolve_revenue(database,
                                                   start_date, end_date,)
            tool_results["revenue"] = revenue_result
            tools_used.append("resolve_revenue")
            contains_forecast = revenue_result.get("contains_forecast", False)
        except (ValueError, RuntimeError) as error:
            logger.warning("Revenue tidak tersedia: %s", error)
            tool_results["revenue"] = {"unavailable": str(error)}
            tools_used.append("resolve_revenue")

    return {
        "tool_results": tool_results,
        "tools_used": tools_used,
        "contains_forecast": contains_forecast,
        "hop_count": state.get("hop_count", 0) + 1,
    }
