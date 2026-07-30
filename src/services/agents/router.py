from __future__ import annotations

from langgraph.graph import END

from src.services.agents.state import MAX_GRAPH_HOPS, AgentState
from src.utils.log import logger


INTENT_TO_NODE = {
    "raw_fact": "analyst",
    "entity_prediction": "ml_inference",
    "insight": "insight_rag",
    "exploratory": "eda",
    "navigation": "navigation",
}

VISUALIZATION_NODE = "visualization"
DIRECT_ANSWER_INTENTS = {"greeting", "gratitude", "out_of_scope"}


def route_after_supervisor(state: AgentState) -> str:
    hop_count = state.get("hop_count", 0)

    if hop_count > MAX_GRAPH_HOPS:
        logger.error("Graf melewati %d lompatan — dihentikan paksa",
                     MAX_GRAPH_HOPS)
        return "synthesizer"

    if state.get("is_answered"):
        return END

    intent = state.get("intent", "")

    if intent in DIRECT_ANSWER_INTENTS:
        return END

    if state.get("error_message"):
        return "synthesizer"

    if state.get("tool_results"):
        if not state.get("visualization_attempted"):
            return VISUALIZATION_NODE
        return "synthesizer"

    next_node = INTENT_TO_NODE.get(intent)
    if next_node is None:
        logger.warning("Intent '%s' tidak dikenali router — "
                       "diserahkan ke synthesizer", intent)
        return "synthesizer"

    return next_node
