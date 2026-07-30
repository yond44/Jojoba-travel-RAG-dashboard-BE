from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from src.services.rag.embeddings import setup_embeddings

logger = logging.getLogger(__name__)

_SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!?])\s+|\n+")

_NUMBER_TOKEN_PATTERN = re.compile(r"\b\d+(?:[.,]\d+)*%?\b")

_MIN_SENTENCE_LENGTH = 15


@dataclass
class GroundednessReport:
    score: float
    is_grounded: bool
    supported: int
    total: int
    unsupported_sentences: List[str] = field(default_factory=list)
    unverified_numbers: List[str] = field(default_factory=list)
    checked: bool = True

    def as_dict(self) -> dict:
        return {"score": round(self.score, 3), "is_grounded": self.is_grounded,
                "supported": self.supported, "total": self.total,
                "unsupported_sentences": self.unsupported_sentences[:5],
                "unverified_numbers": self.unverified_numbers[:10],
                "checked": self.checked}


def _split_into_sentences(text: str) -> List[str]:
    return [sentence.strip() for sentence in _SENTENCE_SPLIT_PATTERN.split(text)
            if len(sentence.strip()) > _MIN_SENTENCE_LENGTH]


def _cosine_similarity(vector_a: np.ndarray, vector_b: np.ndarray) -> float:
    norm_a, norm_b = np.linalg.norm(vector_a), np.linalg.norm(vector_b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(vector_a, vector_b) / (norm_a * norm_b))


def _normalize_number_token(token: str) -> str:
    has_percent_sign = token.endswith("%")
    digits_part = token[:-1] if has_percent_sign else token

    separator_count = digits_part.count(".") + digits_part.count(",")
    if separator_count >= 2:
        digits_part = digits_part.replace(".", "").replace(",", "")
    else:
        digits_part = digits_part.replace(",", ".")

    try:
        numeric_value = float(digits_part)
    except ValueError:
        return token  # bukan angka yang bisa di-parse, bandingkan apa adanya

    canonical_text = ("%.4f" % numeric_value).rstrip("0").rstrip(".")
    return canonical_text + ("%" if has_percent_sign else "")


def _embed_sentences_or_none(embed_model, sentences: List[str]) -> Optional[List[np.ndarray]]:
    if not sentences:
        return []
    try:
        batch_embed_fn = getattr(embed_model, "get_text_embedding_batch", None)
        if batch_embed_fn is not None:
            raw_vectors = batch_embed_fn(sentences)
        else:
            raw_vectors = [embed_model.get_text_embedding(sentence) for sentence in sentences]
        return [np.asarray(vector, dtype=np.float32) for vector in raw_vectors]
    except Exception as e:
        logger.warning("⚠️ Groundedness: embedding gagal, skip pengecekan: %s", e)
        return None


def check_groundedness(answer: str, context: str, sim_threshold: float = 0.55,
                       overall_threshold: float = 0.35) -> GroundednessReport:
    answer_sentences = _split_into_sentences(answer)
    context_sentences = _split_into_sentences(context)

    if not answer_sentences:
        return GroundednessReport(1.0, True, 0, 0)
    if not context_sentences:
        return GroundednessReport(0.0, False, 0, len(answer_sentences),
                                  unsupported_sentences=answer_sentences)

    embed_model = setup_embeddings()

    combined_sentences = context_sentences + answer_sentences
    combined_vectors = _embed_sentences_or_none(embed_model, combined_sentences)

    if combined_vectors is None:
        return GroundednessReport(score=1.0, is_grounded=True, supported=0,
                                  total=len(answer_sentences), checked=False)

    context_vectors = combined_vectors[:len(context_sentences)]
    answer_vectors = combined_vectors[len(context_sentences):]

    supported_count = 0
    unsupported_sentences: List[str] = []
    for sentence, sentence_vector in zip(answer_sentences, answer_vectors):
        best_similarity = max(
            (_cosine_similarity(sentence_vector, context_vector)
             for context_vector in context_vectors), default=0.0)
        if best_similarity >= sim_threshold:
            supported_count += 1
        else:
            unsupported_sentences.append(sentence)

    context_numbers = {_normalize_number_token(token)
                       for token in _NUMBER_TOKEN_PATTERN.findall(context)}
    unverified_numbers = [token for token in _NUMBER_TOKEN_PATTERN.findall(answer)
                          if _normalize_number_token(token) not in context_numbers]

    groundedness_score = supported_count / len(answer_sentences)
    is_grounded = (groundedness_score >= overall_threshold
                  and not (len(unverified_numbers) > 2 and groundedness_score < 0.6))

    if not is_grounded:
        logger.warning("⚠️ Low groundedness: score=%.2f unverified=%s",
                       groundedness_score, unverified_numbers[:5])

    return GroundednessReport(groundedness_score, is_grounded, supported_count,
                              len(answer_sentences), unsupported_sentences, unverified_numbers)