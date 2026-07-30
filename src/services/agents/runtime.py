"""
src/services/agents/runtime.py
Resepsionis gedung: satu pintu masuk untuk seluruh sistem agent.

Tanggung jawabnya: menyalakan graf sekali (thread-safe), menerima
pertanyaan, menyiapkan state awal, menjalankan graf dengan thread_id
yang benar, lalu merapikan hasilnya menjadi respons yang stabil
bentuknya.

Yang TIDAK dilakukan di sini: berpikir. Semua keputusan ada di node
dan router — runtime hanya mengurus siklus hidup dan pembungkusan.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from motor.motor_asyncio import AsyncIOMotorDatabase

from src.model.agent_schemas import ChatResponse
from src.services.agents.graph import build_agent_graph
from src.services.agents.prompts import detect_language, get_prompt_metadata
from src.services.agents.state import build_initial_state
from src.utils.log import logger

_compiled_graph = None
_graph_lock = asyncio.Lock()

BACKLOG_COLLECTION = "unanswered_questions"


async def get_agent_graph():
    global _compiled_graph
    if _compiled_graph is None:
        async with _graph_lock:
            if _compiled_graph is None:
                _compiled_graph = build_agent_graph()
    return _compiled_graph


async def _record_unanswered_question(database, question: str,
                                      top_score: float) -> None:
    try:
        await database[BACKLOG_COLLECTION].insert_one({
            "question": question,
            "top_score": top_score,
            "recorded_at": datetime.now(timezone.utc).isoformat(),
        })
    except Exception as error:
        logger.warning("Gagal mencatat backlog: %s", error) 
                                                            


async def ask_agent(question: str,
                    database: Optional[AsyncIOMotorDatabase] = None,
                    thread_id: Optional[str] = None,
                    language: Optional[str] = None) -> Dict[str, Any]:
    started_at = time.time()
    effective_thread_id = thread_id or str(uuid.uuid4())
    effective_language = language or detect_language(question)

    graph = await get_agent_graph()
    initial_state = build_initial_state(
        question=question, language=effective_language,
        thread_id=effective_thread_id)


    graph_config = {"configurable": {"thread_id": effective_thread_id,
                                     "database": database}}

    try:
        final_state = await graph.ainvoke(initial_state, config=graph_config)
    except Exception as error: 
        logger.exception("Eksekusi graf gagal: %s", error)
        return ChatResponse(
            question=question,
            answer="Maaf, terjadi kendala saat memproses permintaan.",
            thread_id=effective_thread_id, intent="error",
            language=effective_language, degraded=True,
            elapsed_seconds=round(time.time() - started_at, 3),
            prompt_version=get_prompt_metadata()["prompt_version"],
            answered_at=datetime.now(timezone.utc),
        ).model_dump(mode="json")

    insight_result = final_state.get("tool_results", {}).get("insight", {})
    if database is not None and insight_result.get("needs_backlog"):
        await _record_unanswered_question(
            database, final_state.get("standalone_question", question),
            insight_result.get("top_score", 0.0))

    response = ChatResponse(
        question=question,
        answer=final_state.get("final_answer", ""),
        thread_id=effective_thread_id,
        intent=final_state.get("intent", ""),
        language=final_state.get("language", effective_language),
        sources=final_state.get("sources", []),
        tools_used=final_state.get("tools_used", []),
        chart_spec=final_state.get("chart_spec"),
        contains_forecast=final_state.get("contains_forecast", False),
        degraded=final_state.get("degraded", False),
        elapsed_seconds=round(time.time() - started_at, 3),
        prompt_version=get_prompt_metadata()["prompt_version"],
        navigation=final_state.get("navigation"),
        answered_at=datetime.now(timezone.utc),
    )
    logger.info("Agent menjawab intent=%s tools=%s dalam %.2fs",
                response.intent, response.tools_used, response.elapsed_seconds)
    return response.model_dump(mode="json")
