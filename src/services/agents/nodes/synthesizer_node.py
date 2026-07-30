from __future__ import annotations

import json

from langchain_core.messages import AIMessage

from src.services.agents.prompts import build_synthesizer_prompt, get_canned
from src.services.agents.state import AgentState
from src.services.rag.embeddings import setup_llm
from src.utils.log import logger


def _format_fallback_answer(tool_results: dict, language: str) -> str:
    if "insight" in tool_results:
        return tool_results["insight"]["answer"]

    lines = []
    revenue = tool_results.get("revenue")
    if isinstance(revenue, dict) and "segments" in revenue:
        for segment in revenue["segments"]:
            label = "aktual" if segment["kind"] == "actual" else "proyeksi"
            lines.append(f"{segment['start']} s/d {segment['end']} "
                         f"({label}): Rp {segment['total_idr']:,.0f}")

    churn = tool_results.get("churn")
    if isinstance(churn, dict) and churn.get("predictions"):
        for prediction in churn["predictions"]:
            lines.append(f"{prediction['customer_id']}: risiko "
                         f"{prediction['risk_bucket']} "
                         f"({prediction['churn_proba']:.0%})")

    facts = tool_results.get("facts")
    if isinstance(facts, dict):
        for key, value in facts.items():
            if key != "period":
                lines.append(f"{key}: {value}")

    return "\n".join(lines) if lines else get_canned("no_data", language)


def _final_payload(*, answer: str, is_degraded: bool,
                   hop_count: int) -> dict:
    return {
        "final_answer": answer,
        "is_answered": True,
        "degraded": is_degraded,
        "messages": [AIMessage(content=answer)],
        "hop_count": hop_count,
    }


async def synthesizer_node(state: AgentState) -> dict:
    language = state.get("language", "id")
    tool_results = state.get("tool_results", {})
    question = state.get("standalone_question", "")
    next_hop_count = state.get("hop_count", 0) + 1

    if state.get("error_message"):
        return _final_payload(answer=get_canned("error", language),
                              is_degraded=True, hop_count=next_hop_count)

    if not tool_results:
        return _final_payload(answer=get_canned("no_data", language),
                              is_degraded=state.get("degraded", False),
                              hop_count=next_hop_count)

    system_prompt = build_synthesizer_prompt(language)
    language_name = "Bahasa Indonesia" if language == "id" else "English"
    full_prompt = (
        f"{system_prompt}\n\n"
        f"<tool_results>\n"
        f"{json.dumps(tool_results, ensure_ascii=False, indent=2)}\n"
        f"</tool_results>\n\n"
        f"<user_question>\n{question}\n</user_question>"
        f"Answer entirely in {language_name}.")

    try:
        answer_text = str(await setup_llm().acomplete(full_prompt)).strip()
        if not answer_text:
            raise ValueError("LLM mengembalikan jawaban kosong")
        is_degraded = state.get("degraded", False)
    except Exception as error:  
        logger.error("Synthesizer gagal, memakai perakit cadangan: %s", error)
        answer_text = _format_fallback_answer(tool_results, language)
        is_degraded = True

    return _final_payload(answer=answer_text, is_degraded=is_degraded,
                          hop_count=next_hop_count)
