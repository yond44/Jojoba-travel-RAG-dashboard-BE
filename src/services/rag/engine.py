"""
src/services/rag/engine.py
Orkestrator RAG — satu pintu untuk menjawab pertanyaan dari korpus
insight (ml_insights) dan kebijakan bisnis (business_knowledge).

Alur satu pertanyaan:
    1. Validasi input (kosong / terlalu panjang)
    2. Jalur nol-LLM: sapaan & ucapan terima kasih dijawab canned
    3. Cache tingkat 1 (exact hash) -> tingkat 2 (semantic)
    4. Retrieval (dense; hybrid bila flag menyala)
    5. GERBANG RELEVANSI: top_score di bawah ambang -> NOT_FOUND jujur,
       LLM tidak pernah dipanggil (hemat kuota, nol ruang mengarang)
    6. Panggil LLM dengan prompt RAG + retry backoff
    7. Token NOT_FOUND dari LLM dihormati apa adanya
    8. Verifikasi groundedness pasca-jawaban
    9. Simpan ke kedua tingkat cache, kembalikan hasil terstruktur

Pembagian tanggung jawab yang dijaga ketat:
  - retrieval melapor (top_score), engine memutuskan (NOT_FOUND)
  - engine menjawab, PEMANGGIL yang mencatat backlog pertanyaan tak
    terjawab (engine tidak menyentuh MongoDB sama sekali)
  - engine tidak tahu HTTP; routes/agent yang memetakan ke status code
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from src.services.agents.prompts import (
    build_rag_agent_prompt,
    detect_gratitude,
    detect_greeting,
    detect_language,
    get_canned,
    get_prompt_metadata,
)
from src.services.rag.cache import (
    clear_query_cache,
    embed_query_safe,
    get_cache_stats,
    hash_query,
    query_cache,
    semantic_cache,
)
from src.services.rag.config import (
    GROUNDEDNESS_ENABLED,
    GROUNDEDNESS_THRESHOLD,
    MAX_QUERY_LENGTH,
    MIN_RELEVANCE_SCORE,
    RETRY_BASE_DELAY,
)
from src.services.rag.embeddings import setup_llm
from src.services.rag.groundedness import check_groundedness
from src.services.rag.retrieval import retrieve_advanced
from src.services.rag.vector_store import collection_count
from src.utils.log import logger

NOT_FOUND_TOKEN = "NOT_FOUND"

MAX_LLM_ATTEMPTS = 3


class RagUnavailableError(Exception):
    """Index kosong atau LLM tidak dapat dihubungi setelah semua percobaan.
    Nama berdomain, bukan HTTP — routes yang memetakannya ke 503."""


def _build_answer_payload(*, question: str, answer: str, response_type: str,
                          started_at: float,
                          language: str,
                          sources: Optional[List[Dict[str, Any]]] = None,
                          top_score: float = 0.0,
                          from_cache: bool = False,
                          degraded: bool = False,
                          groundedness: Optional[Dict[str, Any]] = None,
                          attempts: int = 0) -> Dict[str, Any]:
    return {
        "question": question,
        "answer": answer,
        "response_type": response_type,   
                                          
        "success": response_type in ("answer", "greeting", "gratitude"),
        "language": language,
        "sources": sources or [],
        "top_score": top_score,
        "from_cache": from_cache,
        "degraded": degraded,             
                                          
        "groundedness": groundedness,
        "llm_attempts": attempts,
        "elapsed_seconds": round(time.time() - started_at, 3),
        **get_prompt_metadata(),         
    }


def _call_llm_with_retry(prompt: str) -> tuple[str, int]:
    language_model = setup_llm()
    last_error: Optional[Exception] = None

    for attempt_number in range(1, MAX_LLM_ATTEMPTS + 1):
        try:
            completion = language_model.complete(prompt)
            answer_text = str(completion).strip()
            if answer_text:
                return answer_text, attempt_number
            last_error = ValueError("LLM mengembalikan jawaban kosong")
        except Exception as error:     
            last_error = error       
        wait_seconds = RETRY_BASE_DELAY * (2 ** (attempt_number - 1))
        logger.warning("LLM gagal (percobaan %d/%d): %s — tunggu %.1fs",
                       attempt_number, MAX_LLM_ATTEMPTS, last_error,
                       wait_seconds)
        if attempt_number < MAX_LLM_ATTEMPTS:
            time.sleep(wait_seconds)

    raise RagUnavailableError(
        f"LLM tidak dapat dihubungi setelah {MAX_LLM_ATTEMPTS} percobaan: "
        f"{last_error}")


def ask(question: str,
        language: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
        use_cache: bool = True) -> Dict[str, Any]:
    started_at = time.time()
    question = (question or "").strip()
    detected_language = language or detect_language(question)

    # --- 1. Validasi input ------------------------------------------------
    if not question:
        return _build_answer_payload(
            question=question, answer=get_canned("error", detected_language),
            response_type="invalid", started_at=started_at,
            language=detected_language)

    if len(question) > MAX_QUERY_LENGTH:
        logger.info("Pertanyaan ditolak: %d karakter (maks %d)",
                    len(question), MAX_QUERY_LENGTH)
        return _build_answer_payload(
            question=question, answer=get_canned("error", detected_language),
            response_type="invalid", started_at=started_at,
            language=detected_language)

    # --- 2. Jalur nol-LLM -------------------------------------------------
    if detect_greeting(question):
        return _build_answer_payload(
            question=question,
            answer=get_canned("greeting", detected_language),
            response_type="greeting", started_at=started_at,
            language=detected_language)

    if detect_gratitude(question):
        return _build_answer_payload(
            question=question,
            answer=get_canned("gratitude", detected_language),
            response_type="gratitude", started_at=started_at,
            language=detected_language)

    # --- 3. Cache dua tingkat --------------------------------------------
    exact_cache_key = hash_query(f"{question}|{filters}|{detected_language}")
    question_embedding = None

    if use_cache:
        cached_payload = query_cache.get(exact_cache_key)
        if cached_payload is not None:
            logger.info("Cache HIT (exact)")
            return {**cached_payload, "from_cache": True,
                    "elapsed_seconds": round(time.time() - started_at, 3)}

        question_embedding = embed_query_safe(question)
        if question_embedding is not None:
            semantic_hit = semantic_cache.get(question_embedding)
            if semantic_hit is not None:
                logger.info("Cache HIT (semantic, similarity=%.3f)",
                            semantic_hit.similarity)
                return {**semantic_hit.result, "from_cache": True,
                        "elapsed_seconds": round(time.time() - started_at, 3)}

    # --- 4. Retrieval -----------------------------------------------------
    retrieval_result = retrieve_advanced(question, filters=filters)
    context = retrieval_result["context"]
    sources = retrieval_result["sources"]
    top_score = retrieval_result["top_score"]

    # --- 5. Gerbang relevansi --------------------------------------------
    if not context or top_score < MIN_RELEVANCE_SCORE:
        logger.info("NOT_FOUND: top_score=%.3f < ambang %.3f",
                    top_score, MIN_RELEVANCE_SCORE)
        return _build_answer_payload(
            question=question,
            answer=get_canned("no_data", detected_language),
            response_type="not_found", started_at=started_at,
            language=detected_language, sources=sources, top_score=top_score)

    # --- 6. Panggil LLM ---------------------------------------------------
    system_prompt = build_rag_agent_prompt(detected_language)
    language_name = "Bahasa Indonesia" if detected_language == "id" else "English"
    full_prompt = (f"{system_prompt}\n\n"
                   f"<retrieved_context>\n{context}\n</retrieved_context>\n\n"
                   f"<user_question>\n{question}\n</user_question>"
                   f"Answer entirely in {language_name}.")

    answer_text, attempts = _call_llm_with_retry(full_prompt)


    if NOT_FOUND_TOKEN in answer_text.upper():
        logger.info("Model melaporkan NOT_FOUND meski skor lolos ambang")
        return _build_answer_payload(
            question=question,
            answer=get_canned("no_data", detected_language),
            response_type="not_found", started_at=started_at,
            language=detected_language, sources=sources,
            top_score=top_score, attempts=attempts)

    # --- 8. Verifikasi groundedness --------------------------------------
    groundedness_report = None
    is_degraded = False
    if GROUNDEDNESS_ENABLED:
        report = check_groundedness(answer_text, context,
                                    overall_threshold=GROUNDEDNESS_THRESHOLD)
        groundedness_report = report.as_dict()
        is_degraded = not report.checked
        if report.checked and not report.is_grounded:
            logger.warning("Jawaban gagal verifikasi groundedness "
                           "(skor %.2f) — dikembalikan sebagai NOT_FOUND",
                           report.score)
            return _build_answer_payload(
                question=question,
                answer=get_canned("no_data", detected_language),
                response_type="not_found", started_at=started_at,
                language=detected_language, sources=sources,
                top_score=top_score, groundedness=groundedness_report,
                attempts=attempts)

    # --- 9. Simpan & kembalikan ------------------------------------------
    payload = _build_answer_payload(
        question=question, answer=answer_text, response_type="answer",
        started_at=started_at, language=detected_language, sources=sources,
        top_score=top_score, degraded=is_degraded,
        groundedness=groundedness_report, attempts=attempts)

    if use_cache:
        query_cache.set(exact_cache_key, payload)
        if question_embedding is not None:
            semantic_cache.set(exact_cache_key, question_embedding,
                               question, payload)

    return payload


def get_rag_status() -> Dict[str, Any]:
    """Ringkasan kesehatan RAG untuk endpoint status/monitoring."""
    vector_count = collection_count()
    return {
        "ready": vector_count > 0,
        "indexed_vectors": vector_count,
        "min_relevance_score": MIN_RELEVANCE_SCORE,
        "groundedness_enabled": GROUNDEDNESS_ENABLED,
        "cache": get_cache_stats(),
        **get_prompt_metadata(),
    }


def reset_caches() -> None:
    """Dipanggil setelah reindex manual dari luar jalur indexer."""
    clear_query_cache()
    logger.info("Cache RAG dikosongkan atas permintaan eksplisit")
