from __future__ import annotations

from pathlib import Path

from src.config.settings import get_settings


_s = get_settings()

GROQ_API_KEY = _s.groq_api_key
GROQ_MODEL = _s.groq_model

CACHE_TTL = _s.cache_ttl
CACHE_MAX_SIZE = _s.cache_max_size
SEMANTIC_CACHE_ENABLED = _s.semantic_cache_enabled
SEMANTIC_CACHE_THRESHOLD = _s.semantic_cache_threshold
SEMANTIC_CACHE_MAX_SIZE = _s.cache_max_size
SEMANTIC_CACHE_DIM = _s.embedding_dim

SIMILARITY_TOP_K = _s.similarity_top_k
MAX_QUERY_LENGTH = 2000
RETRY_BASE_DELAY = 2.0

MIN_RELEVANCE_SCORE = _s.min_relevance_score

HYBRID_ENABLED = _s.hybrid_enabled
HYBRID_ALPHA = _s.hybrid_alpha
RERANK_ENABLED = _s.rerank_enabled
RERANK_MODEL = _s.rerank_model
RERANK_TOP_N = _s.rerank_top_n
ADAPTIVE_TOP_K = _s.adaptive_top_k
QUERY_REWRITE_ENABLED = _s.query_rewrite_enabled
COMPRESSION_ENABLED = _s.compression_enabled
GROUNDEDNESS_ENABLED = _s.groundedness_enabled
GROUNDEDNESS_THRESHOLD = _s.groundedness_threshold

EMBEDDING_MODEL = _s.embedding_model
GROQ_MODEL = _s.groq_model

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CHROMA_DB_DIR = PROJECT_ROOT / "data" / "chroma_db"
DATA_HASH_FILE = CHROMA_DB_DIR / ".data_hash"
COLLECTION_NAME = _s.chroma_collection
