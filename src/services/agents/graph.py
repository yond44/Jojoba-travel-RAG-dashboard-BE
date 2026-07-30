from __future__ import annotations

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from src.services.agents.nodes import (
    analyst_node, eda_node, insight_rag_node, ml_inference_node,
    navigation_node, supervisor_node, synthesizer_node, visualization_node)
from src.services.agents.router import route_after_supervisor
from src.services.agents.state import AgentState
from src.utils.log import logger

SUPERVISOR = "supervisor"
SPECIALIST_NODES = {
    "ml_inference": ml_inference_node,
    "analyst": analyst_node,
    "insight_rag": insight_rag_node,
    "eda": eda_node,
    "navigation": navigation_node,
}
VISUALIZATION = "visualization"
SYNTHESIZER = "synthesizer"


def build_agent_graph(checkpointer=None):
    workflow = StateGraph(AgentState)

    workflow.add_node(SUPERVISOR, supervisor_node)
    for node_name, node_function in SPECIALIST_NODES.items():
        workflow.add_node(node_name, node_function)
    workflow.add_node(VISUALIZATION, visualization_node)
    workflow.add_node(SYNTHESIZER, synthesizer_node)

    workflow.set_entry_point(SUPERVISOR)

    workflow.add_conditional_edges(
        SUPERVISOR,
        route_after_supervisor,
        {**{name: name for name in SPECIALIST_NODES},
         VISUALIZATION: VISUALIZATION,
         SYNTHESIZER: SYNTHESIZER,
         END: END},
    )

    for node_name in SPECIALIST_NODES:
        workflow.add_edge(node_name, SUPERVISOR)

    workflow.add_edge(VISUALIZATION, SUPERVISOR)

    workflow.add_edge(SYNTHESIZER, END)

    compiled_graph = workflow.compile(
        checkpointer=checkpointer or MemorySaver())
    logger.info("Graf agent ter-compile: %d node",
                len(SPECIALIST_NODES) + 3)
    return compiled_graph
