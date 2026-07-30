"""
test/ManualCodeTest/test_rag_manual.py

Uji RAG ujung-ke-ujung terhadap ChromaDB + MongoDB + Groq SUNGGUHAN.
Dipakai setelah `python -m jobs.reindex_rag` berhasil.

    python test/ManualCodeTest/test_rag_manual.py

Enam level, dari yang tidak butuh LLM sampai yang butuh:
    1. Index terisi dan metadatanya benar
    2. Embedding hidup dan berdimensi sesuai kontrak cache
    3. Retrieval mengembalikan chunk relevan (skor masuk akal)
    4. Filter metadata memisahkan insight vs playbook
    5. Jalur nol-LLM (sapaan) dan gerbang NOT_FOUND
    6. Jawaban sungguhan + groundedness + cache
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.services.rag.config import (  # noqa: E402
    MIN_RELEVANCE_SCORE, SEMANTIC_CACHE_DIM,
)
from src.services.rag.embeddings import setup_embeddings  # noqa: E402
from src.services.rag.engine import ask, get_rag_status  # noqa: E402
from src.services.rag.retrieval import retrieve, retrieve_advanced  # noqa: E402
from src.services.rag.vector_store import all_documents, collection_count  # noqa: E402

# Pertanyaan yang HARUS terjawab dari korpus (sesuaikan bila chunk berubah)
ANSWERABLE_QUESTIONS = [
    "Berapa proyeksi revenue 12 bulan ke depan?",
    "Seberapa akurat model churn untuk repeat customer?",
    "Kampanye mana yang biaya konversinya paling murah?",
]

# Pertanyaan di luar korpus — WAJIB berakhir not_found, bukan karangan
UNANSWERABLE_QUESTIONS = [
    "Bagaimana resep nasi goreng spesial?",
    "Siapa pemenang piala dunia 2022?",
    "Berapa harga saham Tesla hari ini?",
]


def level_1_index_terisi() -> None:
    vector_count = collection_count()
    print(f"[1] Vektor ter-index: {vector_count}")
    assert vector_count > 0, ("Index kosong — jalankan "
                              "`python -m jobs.reindex_rag` dulu.")

    documents = all_documents()
    source_types = {doc["metadata"].get("source_type") for doc in documents}
    print(f"    source_type ditemukan: {sorted(source_types)}")
    assert "insight" in source_types, "Chunk ml_insights belum ter-index"
    assert "playbook" in source_types, "Chunk business_knowledge belum ter-index"

    missing_id = [doc for doc in documents if not doc["metadata"].get("id")]
    assert not missing_id, f"{len(missing_id)} chunk tanpa metadata id"
    print("    metadata lengkap (id, topic, source_type) OK")


def level_2_embedding_hidup() -> None:
    embedding = setup_embeddings().get_query_embedding("uji dimensi embedding")
    print(f"[2] Dimensi embedding: {len(embedding)}")
    assert len(embedding) == SEMANTIC_CACHE_DIM, (
        f"Dimensi {len(embedding)} != SEMANTIC_CACHE_DIM "
        f"{SEMANTIC_CACHE_DIM} — semantic cache akan menolak entri.")


def level_3_retrieval_relevan() -> None:
    for question in ANSWERABLE_QUESTIONS:
        result = retrieve_advanced(question)
        top_score = result["top_score"]
        print(f"[3] '{question[:45]}...' -> top_score {top_score:.3f}, "
              f"{len(result['sources'])} sumber")
        assert result["sources"], "Tidak ada chunk terambil sama sekali"
        assert top_score >= MIN_RELEVANCE_SCORE, (
            f"Skor {top_score:.3f} di bawah ambang {MIN_RELEVANCE_SCORE} — "
            f"pertanyaan ini akan ditolak sebagai NOT_FOUND.")

    for question in UNANSWERABLE_QUESTIONS:
        top_score = retrieve_advanced(question)["top_score"]
        print(f"[3] (di luar korpus) '{question[:35]}...' -> {top_score:.3f}")
        assert top_score < MIN_RELEVANCE_SCORE + 0.15, (
            f"Pertanyaan di luar korpus mendapat skor {top_score:.3f} — "
            f"ambang terlalu rendah, akan meloloskan jawaban ngawur.")


def level_4_filter_metadata() -> None:
    playbook_chunks = retrieve("kebijakan diskon segmen premium",
                               filters={"source_type": "playbook"})
    insight_chunks = retrieve("akurasi model churn",
                              filters={"source_type": "insight"})
    print(f"[4] filter playbook -> {len(playbook_chunks)} chunk | "
          f"filter insight -> {len(insight_chunks)} chunk")
    assert all(chunk.metadata["source_type"] == "playbook"
               for chunk in playbook_chunks), "Filter playbook bocor"
    assert all(chunk.metadata["source_type"] == "insight"
               for chunk in insight_chunks), "Filter insight bocor"


def level_5_jalur_tanpa_llm() -> None:
    greeting_result = ask("halo")
    print(f"[5] 'halo' -> {greeting_result['response_type']} "
          f"({greeting_result['elapsed_seconds']}s)")
    assert greeting_result["response_type"] == "greeting"
    assert greeting_result["llm_attempts"] == 0, "Sapaan memanggil LLM"

    for question in UNANSWERABLE_QUESTIONS:
        result = ask(question, use_cache=False)
        print(f"[5] '{question[:35]}...' -> {result['response_type']}")
        assert result["response_type"] == "not_found", (
            f"Pertanyaan di luar korpus dijawab '{result['answer'][:80]}' — "
            f"ini halusinasi yang lolos gerbang.")


def level_6_jawaban_dan_cache() -> None:
    question = ANSWERABLE_QUESTIONS[0]

    first_call_started = time.time()
    first_result = ask(question, use_cache=True)
    first_duration = time.time() - first_call_started

    print(f"[6] Jawaban: {first_result['answer'][:160]}...")
    print(f"    type={first_result['response_type']} "
          f"attempts={first_result['llm_attempts']} "
          f"top_score={first_result['top_score']} "
          f"prompt={first_result['prompt_version']}")
    assert first_result["response_type"] == "answer", (
        f"Pertanyaan yang seharusnya terjawab malah "
        f"'{first_result['response_type']}'")
    assert first_result["sources"], "Jawaban tanpa sumber"

    groundedness = first_result.get("groundedness")
    if groundedness:
        print(f"    groundedness={groundedness['score']} "
              f"checked={groundedness['checked']} "
              f"unverified_numbers={groundedness['unverified_numbers'][:3]}")
        assert groundedness["checked"], ("Verifikasi groundedness tidak "
                                         "berjalan — cek embedding.")

    second_call_started = time.time()
    second_result = ask(question, use_cache=True)
    second_duration = time.time() - second_call_started
    print(f"[6] Panggilan kedua from_cache={second_result['from_cache']} "
          f"({first_duration:.2f}s -> {second_duration:.2f}s)")
    assert second_result["from_cache"], "Cache tidak bekerja"
    assert second_duration < first_duration, "Cache tidak mempercepat"


def main() -> None:
    print("=== STATUS RAG ===")
    for key, value in get_rag_status().items():
        if key != "cache":
            print(f"  {key}: {value}")
    print()

    level_1_index_terisi()
    level_2_embedding_hidup()
    level_3_retrieval_relevan()
    level_4_filter_metadata()
    level_5_jalur_tanpa_llm()
    level_6_jawaban_dan_cache()

    print("\nSEMUA LEVEL LOLOS — RAG siap dipakai agent.")


if __name__ == "__main__":
    main()
