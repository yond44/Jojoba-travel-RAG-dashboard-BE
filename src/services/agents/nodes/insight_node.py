from __future__ import annotations

from src.services.agents.state import AgentState
from src.services.rag.engine import RagUnavailableError, ask
from src.utils.log import logger


async def insight_rag_node(state: AgentState) -> dict:
    question = state.get("standalone_question", "")
    language = state.get("language", "id")
    tool_results = dict(state.get("tool_results", {}))
    tools_used = list(state.get("tools_used", []))
    sources = list(state.get("sources", []))

    try:
        rag_result = ask(question, language=language)
    except RagUnavailableError as error:
        logger.error("RAG tidak tersedia: %s", error)
        return {"error_message": f"Layanan insight sedang tidak tersedia: "
                                 f"{error}",
                "hop_count": state.get("hop_count", 0) + 1}

    tool_results["insight"] = {
        "answer": rag_result["answer"],
        "response_type": rag_result["response_type"],
        "top_score": rag_result["top_score"],
    }
    tools_used.append("search_insights")
    sources.extend(rag_result.get("sources", []))

    if rag_result["response_type"] == "not_found":
        tool_results["insight"]["needs_backlog"] = True

    return {"tool_results": tool_results, "tools_used": tools_used,
            "sources": sources,
            "degraded": state.get("degraded", False) or rag_result["degraded"],
            "hop_count": state.get("hop_count", 0) + 1}
