from __future__ import annotations

from datetime import timedelta

from langchain_core.runnables import RunnableConfig

from src.model.navigation_schemas import (
    NavigationOption, NavigationResult, NavigationSpec)
from src.services.agents.state import AgentState, get_date_param
from src.services.agents.tool_registry import run_tool
from src.services.agents.view_registry import (
    VIEWS_BY_ID, match_view_by_keyword, nearby_views)
from src.utils.clock import business_today
from src.utils.log import logger

DEFAULT_LOOKBACK_DAYS = 30


async def navigation_node(state: AgentState,
                          config: RunnableConfig) -> dict:
    database = config.get("configurable", {}).get("database")
    params = state.get("params", {})
    question = state.get("standalone_question", "")
    original_question = state.get("original_question", "")
    tool_results = dict(state.get("tool_results", {}))
    tools_used = list(state.get("tools_used", []))
    next_hop_count = state.get("hop_count", 0) + 1

    # ---------- Penentuan halaman tujuan ----------
    requested_view_id = params.get("target_view")
    view_spec = VIEWS_BY_ID.get(requested_view_id) if requested_view_id else None
    if view_spec is None:
        view_spec = match_view_by_keyword(f"{original_question} {question}")

    if view_spec is None:
        logger.info("Tidak ada halaman yang cocok untuk permintaan navigasi")
        tool_results["navigation"] = NavigationResult(
            alternatives=[
                NavigationOption(view_id=spec.view_id, label=spec.label,
                                 dashboard_path=spec.dashboard_path)
                for spec in VIEWS_BY_ID.values()][:5]
        ).model_dump(mode="json")
        tools_used.append("navigate_dashboard")
        return {"tool_results": tool_results, "tools_used": tools_used,
                "hop_count": next_hop_count}

    # ---------- Rentang tanggal untuk parameter halaman ----------
    today = business_today()
    start_date = get_date_param(
        params, "start_date", today - timedelta(days=DEFAULT_LOOKBACK_DAYS))
    end_date = get_date_param(params, "end_date", today)
    if end_date > today:
        end_date = today

    query_params = {}
    if view_spec.accepts_date_range:
        query_params = {"start_date": str(start_date),
                        "end_date": str(end_date)}

    navigation_spec = NavigationSpec(
        view_id=view_spec.view_id,
        label=view_spec.label,
        dashboard_path=view_spec.dashboard_path,
        api_path=view_spec.api_path,
        query_params=query_params,
        reason=f"Permintaan cocok dengan halaman {view_spec.label}")

    navigation_result = NavigationResult(
        target=navigation_spec,
        alternatives=[
            NavigationOption(view_id=spec.view_id, label=spec.label,
                             dashboard_path=spec.dashboard_path)
            for spec in nearby_views(view_spec)])

    tool_results["navigation"] = navigation_result.model_dump(mode="json")
    tools_used.append("navigate_dashboard")

    # ---------- Data pendamping agar jawaban tidak hanya berupa tautan ----------
    if view_spec.tool_id and database is not None:
        try:
            tool_results[view_spec.tool_id] = await run_tool(
                view_spec.tool_id, database, start_date, end_date)
            tools_used.append(view_spec.tool_id)
        except Exception as error:
            logger.exception("Data pendamping navigasi gagal: %s", error)

    logger.info("Navigasi diarahkan ke %s", view_spec.view_id)
    return {"tool_results": tool_results, "tools_used": tools_used,
            "navigation": navigation_result.model_dump(mode="json"),
            "hop_count": next_hop_count}
