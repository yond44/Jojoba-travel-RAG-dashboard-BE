from __future__ import annotations

import hashlib
import time
from typing import Any, Dict, Iterable, List, Optional

from pymongo import MongoClient

from src.config.settings import get_settings
from src.services.rag.cache import clear_query_cache
from src.services.rag.embeddings import setup_embeddings
from src.services.rag.retrieval import invalidate_bm25
from src.services.rag.vector_store import get_collection
from src.utils.log import logger


SOURCE_COLLECTIONS: Dict[str, str] = {
    "ml_insights": "insight",
    "business_knowledge": "playbook",
}

EMBEDDING_BATCH_SIZE = 64
MAX_DELETE_RATIO = 0.5      
MIN_CHUNK_LENGTH = 20      


def compute_chunk_id(chunk_key: str, text: str) -> str:
    return hashlib.sha1(f"{chunk_key}::{text}".encode("utf-8")).hexdigest()


def _load_source_chunks(database) -> Dict[str, Dict[str, Any]]:
    collected: Dict[str, Dict[str, Any]] = {}

    for collection_name, source_type in SOURCE_COLLECTIONS.items():
        documents = list(database[collection_name].find({}, {"_id": 0}))
        skipped = 0

        for document in documents:
            text = (document.get("text") or "").strip()
            logical_key = document.get("id") or document.get("chunk_id")

            if not logical_key or len(text) < MIN_CHUNK_LENGTH:
                skipped += 1
                continue

            chunk_id = compute_chunk_id(f"{collection_name}:{logical_key}", text)
            collected[chunk_id] = {
                "text": text,
                "metadata": {
                    "id": str(logical_key),
                    "topic": str(document.get("topic", "")),
                    "source_type": source_type,
                    "source_collection": collection_name,
                    "updated_at": str(document.get("updated_at", "")),
                },
            }

        logger.info("Sumber %s: %d dokumen, %d layak index, %d dilewati",
                    collection_name, len(documents),
                    len(documents) - skipped, skipped)

    return collected


def _batched(items: List[str], batch_size: int) -> Iterable[List[str]]:
    for start in range(0, len(items), batch_size):
        yield items[start:start + batch_size]


def reindex_corpus(force_delete: bool = False,
                   mongo_client: Optional[MongoClient] = None
                   ) -> Dict[str, Any]:
    started_at = time.time()
    settings = get_settings()

    owns_client = mongo_client is None
    client = mongo_client or MongoClient(settings.mongo_url,
                                         serverSelectionTimeoutMS=8000)
    try:
        client.admin.command("ping")
        database = client[settings.database_name]

        desired_chunks = _load_source_chunks(database)
        collection = get_collection()
        existing_ids = set(collection.get(include=[]).get("ids", []))

        # --- PENGAMAN 1: sumber kosong tidak boleh mengosongkan index ---
        if not desired_chunks:
            if existing_ids:
                raise RuntimeError(
                    "Sumber korpus mengembalikan NOL chunk sementara index "
                    f"berisi {len(existing_ids)} vektor. Penghapusan "
                    "dibatalkan — periksa koneksi MongoDB, nama database, "
                    "dan isi collection sumber terlebih dahulu.")
            logger.warning("Sumber kosong dan index juga kosong — "
                           "tidak ada yang dikerjakan.")
            return {"added": 0, "deleted": 0, "unchanged": 0,
                    "total_indexed": 0, "duration_seconds": 0.0}

        ids_to_add = [chunk_id for chunk_id in desired_chunks
                      if chunk_id not in existing_ids]
        ids_to_delete = [chunk_id for chunk_id in existing_ids
                         if chunk_id not in desired_chunks]

        # --- PENGAMAN 2: penghapusan masif butuh persetujuan eksplisit ---
        if existing_ids and not force_delete:
            delete_ratio = len(ids_to_delete) / len(existing_ids)
            if delete_ratio > MAX_DELETE_RATIO:
                raise RuntimeError(
                    f"Rencana penghapusan {len(ids_to_delete)} dari "
                    f"{len(existing_ids)} vektor ({delete_ratio:.0%}) "
                    f"melewati ambang {MAX_DELETE_RATIO:.0%}. Bila memang "
                    f"disengaja, jalankan ulang dengan force_delete=True.")

        # --- Tambah chunk baru/berubah, per batch ---
        if ids_to_add:
            embedder = setup_embeddings()
            for batch_ids in _batched(ids_to_add, EMBEDDING_BATCH_SIZE):
                batch_texts = [desired_chunks[cid]["text"] for cid in batch_ids]
                batch_metadatas = [desired_chunks[cid]["metadata"]
                                   for cid in batch_ids]
                batch_embeddings = [embedder.get_text_embedding(text)
                                    for text in batch_texts]
                collection.upsert(ids=batch_ids, documents=batch_texts,
                                  metadatas=batch_metadatas,
                                  embeddings=batch_embeddings)
                logger.info("Embed & upsert %d chunk", len(batch_ids))

        if ids_to_delete:
            collection.delete(ids=ids_to_delete)
            logger.info("Hapus %d chunk usang", len(ids_to_delete))

     
        if ids_to_add or ids_to_delete:
            invalidate_bm25()
            clear_query_cache()
            logger.info("BM25 index dan query cache di-invalidasi")

        summary = {
            "added": len(ids_to_add),
            "deleted": len(ids_to_delete),
            "unchanged": len(desired_chunks) - len(ids_to_add),
            "total_indexed": collection.count(),
            "duration_seconds": round(time.time() - started_at, 2),
        }
        logger.info("Reindex selesai: %s", summary)
        return summary

    finally:
        if owns_client:
            client.close()
