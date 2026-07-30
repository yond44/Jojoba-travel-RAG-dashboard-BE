from __future__ import annotations

import math
import re
from collections import Counter
from functools import lru_cache
from typing import Any, Dict, List, Optional, Set, Tuple

from src.services.rag.cache import TTLLRUCache, hash_query
from src.services.rag.config import (
    ADAPTIVE_TOP_K, CACHE_TTL, HYBRID_ALPHA, HYBRID_ENABLED,
    SIMILARITY_TOP_K,
)
from src.services.rag.embeddings import setup_embeddings
from src.services.rag.types import RetrievedChunk
from src.services.rag.vector_store import all_documents, dense_search
from src.utils.log import logger

_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")

_retrieval_cache = TTLLRUCache(max_size=256, ttl=CACHE_TTL)


def _tokenize(text: str) -> List[str]:
    return _TOKEN_PATTERN.findall(text.lower())


class BM25:
    def __init__(self, corpus: List[Dict[str, Any]],
                 term_saturation: float = 1.5,
                 length_penalty: float = 0.75):
        self.term_saturation = term_saturation
        self.length_penalty = length_penalty
        self.documents = corpus

        self.tokenized_docs = [_tokenize(doc["text"]) for doc in corpus]
        self.doc_lengths = [len(tokens) for tokens in self.tokenized_docs]
        self.average_doc_length = (
            sum(self.doc_lengths) / len(self.doc_lengths)
            if self.doc_lengths else 0.0)

        self.term_frequencies = [Counter(tokens)
                                 for tokens in self.tokenized_docs]

        doc_frequency: Counter = Counter()
        for tokens in self.tokenized_docs:
            for term in set(tokens):
                doc_frequency[term] += 1

        total_docs = len(corpus)
        self.inverse_doc_frequency = {
            term: math.log(1 + (total_docs - freq + 0.5) / (freq + 0.5))
            for term, freq in doc_frequency.items()}

    def search(self, query: str, top_k: int,
               allowed_indices: Optional[Set[int]] = None
               ) -> List[Tuple[int, float]]:
        query_terms = _tokenize(query)
        scored_docs: List[Tuple[int, float]] = []

        for doc_index, doc_term_freq in enumerate(self.term_frequencies):
            if (allowed_indices is not None
                    and doc_index not in allowed_indices):
                continue

            score = 0.0
            doc_length = self.doc_lengths[doc_index] or 1
            for term in query_terms:
                if term not in doc_term_freq:
                    continue
                term_freq = doc_term_freq[term]
                length_normalizer = (
                    1 - self.length_penalty
                    + self.length_penalty * doc_length
                    / (self.average_doc_length or 1))
                score += (self.inverse_doc_frequency.get(term, 0.0)
                          * (term_freq * (self.term_saturation + 1))
                          / (term_freq
                             + self.term_saturation * length_normalizer))
            if score > 0:
                scored_docs.append((doc_index, score))

        scored_docs.sort(key=lambda pair: pair[1], reverse=True)
        return scored_docs[:top_k]


@lru_cache(maxsize=1)
def _build_bm25_index() -> BM25:
    corpus = all_documents()
    logger.info("BM25 index built over %d chunks", len(corpus))
    return BM25(corpus)


def invalidate_bm25() -> None:
    _build_bm25_index.cache_clear()
    _retrieval_cache.clear()


def adaptive_top_k(query: str, base_top_k: int) -> int:
    word_count = len(query.split())
    is_specific = bool(re.search(r"\b\d{4}\b|\d+%", query))

    if word_count <= 4 and not is_specific:
        return min(base_top_k * 2, 10)
    if word_count >= 14 or is_specific:
        return max(3, base_top_k - 2)
    return base_top_k


def _normalize_scores(scores: List[float]) -> List[float]:
    if not scores:
        return []
    lowest, highest = min(scores), max(scores)
    if highest - lowest < 1e-9:
        return [1.0] * len(scores)
    return [(score - lowest) / (highest - lowest) for score in scores]


def _weighted_fuse(dense_results: List[RetrievedChunk],
                   sparse_results: List[RetrievedChunk],
                   dense_weight: float) -> List[RetrievedChunk]:
    dense_norms = _normalize_scores(
        [chunk.score for chunk in dense_results])
    sparse_norms = _normalize_scores(
        [chunk.score for chunk in sparse_results])

    fused_bucket: Dict[str, Tuple[float, RetrievedChunk]] = {}

    for chunk, normalized_score in zip(dense_results, dense_norms):
        dedup_key = chunk.text[:120]
        fused_bucket[dedup_key] = (dense_weight * normalized_score, chunk)

    for chunk, normalized_score in zip(sparse_results, sparse_norms):
        dedup_key = chunk.text[:120]
        previous_score, kept_chunk = fused_bucket.get(dedup_key, (0.0, chunk))
        fused_bucket[dedup_key] = (
            previous_score + (1 - dense_weight) * normalized_score,
            kept_chunk)

    fused_chunks = [
        RetrievedChunk(kept_chunk.text, combined_score, kept_chunk.metadata)
        for combined_score, kept_chunk in fused_bucket.values()]
    fused_chunks.sort(key=lambda chunk: chunk.score, reverse=True)
    return fused_chunks


def retrieve(query: str,
             filters: Optional[Dict[str, Any]] = None,
             top_k: Optional[int] = None,
             use_hybrid: Optional[bool] = None) -> List[RetrievedChunk]:

    use_hybrid = HYBRID_ENABLED if use_hybrid is None else use_hybrid
    base_top_k = top_k or SIMILARITY_TOP_K
    effective_top_k = (adaptive_top_k(query, base_top_k)
                       if ADAPTIVE_TOP_K else base_top_k)

    query_embedding = setup_embeddings().get_query_embedding(query)
    dense_results = [
        RetrievedChunk(hit["text"], hit["score"], hit["metadata"])
        for hit in dense_search(query_embedding, top_k=effective_top_k,
                                where=filters)]
    if not use_hybrid:
        return dense_results[:effective_top_k]

    bm25_index = _build_bm25_index()
    allowed_indices: Optional[Set[int]] = None
    if filters:
        allowed_indices = {
            doc_index for doc_index, doc in enumerate(bm25_index.documents)
            if all(doc["metadata"].get(field) == value
                   for field, value in filters.items())}

    sparse_results = [
        RetrievedChunk(bm25_index.documents[doc_index]["text"],
                       bm25_score,
                       bm25_index.documents[doc_index]["metadata"])
        for doc_index, bm25_score in bm25_index.search(
            query, top_k=effective_top_k, allowed_indices=allowed_indices)]

    fused_chunks = _weighted_fuse(dense_results, sparse_results, HYBRID_ALPHA)
    return fused_chunks[:effective_top_k]


def retrieve_advanced(question: str,
                      filters: Optional[Dict[str, Any]] = None
                      ) -> Dict[str, Any]:
    cache_key = hash_query(f"{question}|{filters}")
    cached_result = _retrieval_cache.get(cache_key)
    if cached_result is not None:
        return {**cached_result, "from_cache": True}

    retrieved_chunks = retrieve(question, filters=filters)

    context = "\n\n".join(
        f"[{chunk.metadata.get('source_type', 'insight')}] {chunk.text}"
        for chunk in retrieved_chunks)

    sources = [{
        "id": chunk.metadata.get("id", ""),
        "topic": chunk.metadata.get("topic", ""),
        "source_type": chunk.metadata.get("source_type", "insight"),
        "score": round(chunk.score, 4),
        "text": chunk.text[:300],
    } for chunk in retrieved_chunks]

    result = {
        "context": context,
        "sources": sources,
        "top_score": (round(retrieved_chunks[0].score, 4)
                      if retrieved_chunks else 0.0),
    }
    _retrieval_cache.set(cache_key, result)
    return {**result, "from_cache": False}
