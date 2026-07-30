from __future__ import annotations

import json
import re

from langchain_core.messages import AIMessage, HumanMessage
from pydantic import ValidationError

from src.model.agent_schemas import SupervisorDecision
from src.services.agents.prompts import (
    build_supervisor_prompt, detect_greeting, detect_gratitude,
    detect_language, get_canned)
from src.services.agents.state import AgentState
from src.services.rag.embeddings import setup_llm
from src.utils.clock import business_today
from src.utils.log import logger

_JSON_BLOCK_PATTERN = re.compile(r"\{.*\}", re.DOTALL)

# Berapa banyak pesan terakhir yang disodorkan sebagai riwayat.
HISTORY_MESSAGE_LIMIT = 6

# Kata kunci jalur cadangan — dipakai HANYA bila LLM gagal.
PREDICTION_KEYWORDS = ("prediksi", "proyeksi", "forecast", "revenue",
                       "churn", "risiko", "bulan depan", "minggu depan")
INSIGHT_KEYWORDS = ("kenapa", "mengapa", "strategi", "sebaiknya",
                    "rekomendasi", "artinya", "why", "should")
FACT_KEYWORDS = ("berapa", "jumlah", "total", "how many", "daftar")
FORECAST_KEYWORDS = ("proyeksi", "prediksi", "forecast", "ramalan",
                     "ke depan", "kedepan", "mendatang", "next week",
                     "next month", "upcoming")
REVENUE_KEYWORDS = ("revenue", "pendapatan", "omzet", "penjualan", "income")

def _extract_json_object(raw_text: str) -> dict | None:
    """LLM sering membungkus JSON dengan basa-basi atau pagar markdown.
    Ambil blok kurung kurawal terluar, lalu parse."""
    match = _JSON_BLOCK_PATTERN.search(raw_text)
    if match is None:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def _fallback_decision(question: str) -> SupervisorDecision:
    question_lower = question.lower()
    if any(word in question_lower for word in INSIGHT_KEYWORDS):
        intent = "insight"
    elif any(word in question_lower for word in PREDICTION_KEYWORDS):
        intent = "entity_prediction"
    elif any(word in question_lower for word in FACT_KEYWORDS):
        intent = "raw_fact"
    else:
        intent = "insight"  

    logger.info("Supervisor memakai jalur cadangan -> intent=%s", intent)
    return SupervisorDecision(standalone_question=question, intent=intent)


def _answered_payload(*, intent: str, language: str, answer: str,
                      question: str, hop_count: int,
                      standalone_question: str | None = None) -> dict:
    payload = {
        "intent": intent,
        "language": language,
        "final_answer": answer,
        "messages": [HumanMessage(content=question),
                     AIMessage(content=answer)],
        "is_answered": True,
        "hop_count": hop_count,
    }
    if standalone_question is not None:
        payload["standalone_question"] = standalone_question
    return payload


async def supervisor_node(state: AgentState) -> dict:

    
    question = state.get("original_question", "")
    language = state.get("language") or detect_language(question)
    hop_count = state.get("hop_count", 0) + 1
    
    if state.get("intent") and state.get("tool_results"):
        return {"hop_count": hop_count}

    # --- Jalur nol-LLM: sapaan & terima kasih dijawab langsung ---
    if detect_greeting(question):
        return _answered_payload(
            intent="greeting", language=language,
            answer=get_canned("greeting", language),
            question=question, hop_count=hop_count)

    if detect_gratitude(question):
        return _answered_payload(
            intent="gratitude", language=language,
            answer=get_canned("gratitude", language),
            question=question, hop_count=hop_count)

    # --- Klasifikasi via LLM, dengan jaring pengaman berlapis ---
    system_prompt = build_supervisor_prompt(language)
    conversation_history = "\n".join(
        str(message.content)
        for message in state.get("messages", [])[-HISTORY_MESSAGE_LIMIT:])
    full_prompt = (
        f"{system_prompt}\n\n"
        f"<business_today>{business_today()}</business_today>\n"
        f"<conversation_history>\n{conversation_history}\n"
        f"</conversation_history>\n"
        f"<latest_user_message>\n{question}\n</latest_user_message>")

    try:
        raw_response = str(await setup_llm().acomplete(full_prompt))
        parsed_json = _extract_json_object(raw_response)
        decision = (SupervisorDecision.model_validate(parsed_json)
                    if parsed_json else _fallback_decision(question))
    except ValidationError as error:
        logger.warning("JSON Supervisor tidak sesuai kontrak: %s", error)
        decision = _fallback_decision(question)
    except Exception as error:  
        logger.error("Supervisor gagal memanggil LLM: %s", error)
        decision = _fallback_decision(question)

    
    NON_FORECASTABLE_KEYWORDS = ("conversion", "konversi", "channel", "kanal",
                             "kampanye", "campaign", "cpa", "agen", "agent",
                             "segmentasi", "segment")

    question_lower = f"{question} {decision.standalone_question}".lower()
    
    if (any(word in question_lower for word in FORECAST_KEYWORDS)
            and any(word in question_lower for word in REVENUE_KEYWORDS)):
        logger.info("Permintaan proyeksi revenue — intent dikoreksi ke "
                    "entity_prediction")
        decision.intent = "entity_prediction"
        
    if (decision.intent == "entity_prediction"
            and any(word in question_lower
                    for word in NON_FORECASTABLE_KEYWORDS)):
        logger.info("Metrik non-forecastable terdeteksi — intent "
                    "dikoreksi ke raw_fact")
        decision.intent = "raw_fact"
        
    EXPLORATORY_KEYWORDS = ("tren", "trend", "sebaran", "distribusi",
                        "distribution", "komposisi", "pola", "eksplorasi",
                        "kualitas data", "gambaran umum", "overview")

    if (decision.intent in ("raw_fact", "insight")
            and any(word in question_lower
                    for word in EXPLORATORY_KEYWORDS)):
        logger.info("Pertanyaan bersifat eksploratif — intent dikoreksi "
                    "ke exploratory")
        decision.intent = "exploratory"
    
    if decision.intent == "out_of_scope":
        return _answered_payload(
            intent="out_of_scope", language=language,
            answer=get_canned("off_topic", language),
            question=question, hop_count=hop_count,
            standalone_question=decision.standalone_question)

    logger.info("Supervisor: intent=%s | '%s'", decision.intent,
                decision.standalone_question[:60])
    return {
        "intent": decision.intent,
        "language": language,
        "standalone_question": decision.standalone_question,
        "params": decision.params.model_dump(mode="json"),
        "messages": [HumanMessage(content=question)],
        "hop_count": hop_count,
    }
