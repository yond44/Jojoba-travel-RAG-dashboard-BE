from __future__ import annotations

from datetime import timedelta

from langchain_core.runnables import RunnableConfig

from src.services.agents.state import AgentState, get_date_param
from src.services.ml.ml_services import (
    CustomerHistoryNotFoundError, predict_churn)
from src.services.ml.revenue_resolver import resolve_revenue
from src.utils.clock import business_today
from src.utils.log import logger


DEFAULT_LOOKBACK_DAYS = 30


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
