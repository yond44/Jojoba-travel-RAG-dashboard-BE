"""
test/UnitTest/test_rag_unit.py

Unit test komponen RAG — TANPA MongoDB, ChromaDB, maupun Groq.
Cepat (< 2 detik), deterministik, aman dijalankan di CI setiap commit.

    pytest test/UnitTest/test_rag_unit.py -v

Pembagian dengan test manual: file ini menguji LOGIKA (rumus BM25,
normalisasi, gerbang keputusan, perilaku cache); test manual menguji
INTEGRASI (index sungguhan, LLM sungguhan). Bug logika ketahuan di sini
dalam hitungan detik, bukan setelah memanggil API berbayar.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.services.rag import engine as engine_module  # noqa: E402
from src.services.rag.cache import TTLLRUCache, hash_query  # noqa: E402
from src.services.rag.groundedness import (  # noqa: E402
    _normalize_number_token, _split_into_sentences, check_groundedness,
)
from src.services.rag.indexer import compute_chunk_id  # noqa: E402
from src.services.rag.retrieval import (  # noqa: E402
    BM25, _normalize_scores, _weighted_fuse, adaptive_top_k,
)
from src.services.rag.types import RetrievedChunk  # noqa: E402


# ---------------------------------------------------------------------------
# BM25
# ---------------------------------------------------------------------------
@pytest.fixture
def sample_corpus() -> list[dict]:
    return [
        {"text": "Proyeksi revenue 12 bulan ke depan mencapai 2.84 triliun",
         "metadata": {"id": "forecast", "source_type": "insight"}},
        {"text": "Model churn memiliki AUC 0.93 untuk repeat customer",
         "metadata": {"id": "churn", "source_type": "insight"}},
        {"text": "Segmen premium tidak diberi diskon massal",
         "metadata": {"id": "premium", "source_type": "playbook"}},
    ]


def test_bm25_menemukan_dokumen_dengan_kata_kunci(sample_corpus):
    index = BM25(sample_corpus)
    results = index.search("churn AUC", top_k=3)
    assert results, "BM25 tidak menemukan apa pun"
    best_doc_index = results[0][0]
    assert "churn" in sample_corpus[best_doc_index]["text"].lower()


def test_bm25_mengabaikan_dokumen_tanpa_kata_kunci(sample_corpus):
    index = BM25(sample_corpus)
    results = index.search("kata yang tidak ada di korpus", top_k=3)
    assert results == [], "Dokumen tanpa term sama sekali tidak boleh berskor"


def test_bm25_menghormati_allowed_indices(sample_corpus):
    index = BM25(sample_corpus)
    results = index.search("revenue churn premium", top_k=5,
                           allowed_indices={2})
    assert all(doc_index == 2 for doc_index, _ in results)


def test_bm25_korpus_kosong_tidak_meledak():
    index = BM25([])
    assert index.search("apa saja", top_k=5) == []


# ---------------------------------------------------------------------------
# Fusion & normalisasi
# ---------------------------------------------------------------------------
def test_normalize_scores_memetakan_ke_nol_satu():
    assert _normalize_scores([2.0, 4.0, 6.0]) == [0.0, 0.5, 1.0]


def test_normalize_scores_menangani_nilai_identik():
    assert _normalize_scores([3.0, 3.0]) == [1.0, 1.0]


def test_normalize_scores_daftar_kosong():
    assert _normalize_scores([]) == []


def test_weighted_fuse_menggabungkan_chunk_yang_sama():
    shared_text = "Revenue tumbuh 14 persen dibanding periode sebelumnya"
    dense = [RetrievedChunk(shared_text, 0.9, {"id": "a"})]
    sparse = [RetrievedChunk(shared_text, 5.0, {"id": "a"})]
    fused = _weighted_fuse(dense, sparse, dense_weight=0.7)
    assert len(fused) == 1, "Chunk identik harus digabung, bukan diduplikasi"
    assert fused[0].score == pytest.approx(1.0)


def test_weighted_fuse_mengurutkan_menurun():
    dense = [RetrievedChunk("chunk pertama yang panjangnya cukup", 0.9, {}),
             RetrievedChunk("chunk kedua yang panjangnya cukup", 0.1, {})]
    fused = _weighted_fuse(dense, [], dense_weight=1.0)
    assert fused[0].score >= fused[1].score


# ---------------------------------------------------------------------------
# adaptive_top_k
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("query,base_k,expected", [
    ("revenue naik?", 5, 10),                        # pendek & umum -> lebar
    ("berapa revenue tahun 2026", 5, 3),             # spesifik -> sempit
    ("apa yang menyebabkan pelanggan pergi dari layanan kami", 5, 5),
])
def test_adaptive_top_k(query, base_k, expected):
    assert adaptive_top_k(query, base_k) == expected


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------
def test_hash_query_stabil_dan_tidak_sensitif_kapital():
    assert hash_query("Berapa Revenue?") == hash_query("berapa revenue?")
    assert hash_query("a") != hash_query("b")


def test_ttl_cache_menyimpan_dan_mengambil():
    cache = TTLLRUCache(max_size=2, ttl=60)
    cache.set("kunci", {"jawaban": "nilai"})
    assert cache.get("kunci") == {"jawaban": "nilai"}
    assert cache.stats()["hits"] == 1


def test_ttl_cache_kedaluwarsa():
    cache = TTLLRUCache(max_size=2, ttl=0.05)
    cache.set("kunci", "nilai")
    time.sleep(0.1)
    assert cache.get("kunci") is None
    assert cache.stats()["evictions_ttl"] == 1


def test_ttl_cache_mengusir_entri_terlama():
    cache = TTLLRUCache(max_size=2, ttl=60)
    cache.set("pertama", 1)
    cache.set("kedua", 2)
    cache.set("ketiga", 3)          # melebihi kapasitas
    assert cache.get("pertama") is None
    assert cache.get("ketiga") == 3


# ---------------------------------------------------------------------------
# Indexer: identitas chunk
# ---------------------------------------------------------------------------
def test_chunk_id_berubah_saat_teks_berubah():
    id_lama = compute_chunk_id("ml_insights:forecast", "revenue 2.8 triliun")
    id_baru = compute_chunk_id("ml_insights:forecast", "revenue 2.9 triliun")
    assert id_lama != id_baru, ("Teks berubah harus menghasilkan id baru — "
                                "ini mesin incremental indexing.")


def test_chunk_id_stabil_untuk_teks_sama():
    first = compute_chunk_id("ml_insights:churn", "AUC 0.93")
    second = compute_chunk_id("ml_insights:churn", "AUC 0.93")
    assert first == second, "Teks sama harus TIDAK di-embed ulang"


# ---------------------------------------------------------------------------
# Groundedness (bagian yang tidak butuh embedding)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("token_a,token_b", [
    ("1.500.000", "1500000"),
    ("12.5%", "12,5%"),
    ("2,84", "2.84"),
])
def test_normalisasi_angka_menyamakan_format(token_a, token_b):
    assert _normalize_number_token(token_a) == _normalize_number_token(token_b)


def test_split_kalimat_membuang_potongan_terlalu_pendek():
    sentences = _split_into_sentences(
        "Ya. Revenue bulan depan diproyeksikan naik empat belas persen.")
    assert len(sentences) == 1, "Potongan 'Ya.' terlalu pendek untuk dinilai"


def test_groundedness_konteks_kosong_langsung_gagal():
    report = check_groundedness(
        "Revenue bulan depan diproyeksikan mencapai tiga puluh miliar.", "")
    assert report.is_grounded is False
    assert report.score == 0.0


def test_groundedness_jawaban_kosong_dianggap_lolos():
    report = check_groundedness("", "konteks apa pun yang cukup panjang")
    assert report.is_grounded is True
    assert report.total == 0


# ---------------------------------------------------------------------------
# Engine: gerbang keputusan (retrieval & LLM di-stub)
# ---------------------------------------------------------------------------
def test_engine_menolak_pertanyaan_kosong():
    result = engine_module.ask("   ")
    assert result["response_type"] == "invalid"
    assert result["llm_attempts"] == 0


def test_engine_menolak_pertanyaan_terlalu_panjang():
    result = engine_module.ask("revenue " * 5000)
    assert result["response_type"] == "invalid"


def test_engine_menjawab_sapaan_tanpa_llm(monkeypatch):
    def gagal_bila_dipanggil(*args, **kwargs):
        raise AssertionError("Sapaan tidak boleh memanggil retrieval")

    monkeypatch.setattr(engine_module, "retrieve_advanced",
                        gagal_bila_dipanggil)
    result = engine_module.ask("halo")
    assert result["response_type"] == "greeting"


def test_engine_not_found_saat_skor_di_bawah_ambang(monkeypatch):
    monkeypatch.setattr(
        engine_module, "retrieve_advanced",
        lambda question, filters=None: {
            "context": "konteks yang tidak relevan sama sekali",
            "sources": [], "top_score": 0.05, "from_cache": False})

    def gagal_bila_dipanggil(prompt):
        raise AssertionError("LLM dipanggil padahal skor di bawah ambang")

    monkeypatch.setattr(engine_module, "_call_llm_with_retry",
                        gagal_bila_dipanggil)

    result = engine_module.ask("pertanyaan di luar korpus", use_cache=False)
    assert result["response_type"] == "not_found"


def test_engine_menghormati_token_not_found_dari_model(monkeypatch):
    monkeypatch.setattr(
        engine_module, "retrieve_advanced",
        lambda question, filters=None: {
            "context": "Revenue bulan depan diproyeksikan tiga puluh miliar.",
            "sources": [{"id": "forecast"}], "top_score": 0.9,
            "from_cache": False})
    monkeypatch.setattr(engine_module, "_call_llm_with_retry",
                        lambda prompt: ("NOT_FOUND", 1))

    result = engine_module.ask("pertanyaan apa pun", use_cache=False)
    assert result["response_type"] == "not_found"
    assert "NOT_FOUND" not in result["answer"], (
        "Token internal bocor ke jawaban pengguna")


def test_engine_mengembalikan_jawaban_lengkap(monkeypatch):
    monkeypatch.setattr(
        engine_module, "retrieve_advanced",
        lambda question, filters=None: {
            "context": "Proyeksi revenue 12 bulan ke depan 2.84 triliun.",
            "sources": [{"id": "forecast", "source_type": "insight"}],
            "top_score": 0.88, "from_cache": False})
    monkeypatch.setattr(
        engine_module, "_call_llm_with_retry",
        lambda prompt: ("Proyeksi revenue 12 bulan ke depan "
                        "adalah 2.84 triliun rupiah.", 1))
    monkeypatch.setattr(engine_module, "GROUNDEDNESS_ENABLED", False)

    result = engine_module.ask("berapa proyeksi revenue?", use_cache=False)
    assert result["response_type"] == "answer"
    assert result["success"] is True
    assert result["sources"]
    assert result["prompt_version"]
