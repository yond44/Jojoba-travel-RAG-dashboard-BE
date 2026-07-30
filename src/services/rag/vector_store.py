from __future__ import annotations

import threading
from typing import Any, Dict, List, Optional

import chromadb
from chromadb.api.models.Collection import Collection

from src.services.rag.config import CHROMA_DB_DIR, COLLECTION_NAME
from src.utils.log import logger


_collection: Collection | None = None
_lock = threading.Lock()


def get_collection() -> Collection:
    global _collection
    
    if _collection is None:
        with _lock:
            if _collection is None:
                client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))
                _collection = client.get_or_create_collection(
                    COLLECTION_NAME,
                    metadata={"hnsw:space": "cosine"} 
                )
                logger.info("ChromaDB ready: %d vectors in '%s'",
                            _collection.count(), COLLECTION_NAME)
    return _collection


def dense_search(query_embedding: List[float], top_k: int,  
                 where: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    result = get_collection().query(
        query_embeddings=[query_embedding], n_results=top_k,
        where=where or None,
        include=["documents", "metadatas", "distances"]
    )
    
    docs = (result.get("documents") or [[]])[0]
    metas = (result.get("metadatas") or [[]])[0]
    dists = (result.get("distances") or [[]])[0]
    
    return [{"text": text or "", "score": 1.0 - float(distance), "metadata": metadata or {}}
            for text, metadata, distance in zip(docs, metas, dists)]
    
    
def all_documents() -> List[Dict[str, Any]]:
    result = get_collection().get(include=["documents", "metadatas"])
    return [{"text": doc or "", "metadata": metadata or {}}
            for doc, metadata in zip(result.get("documents") or [],
                                     result.get("metadatas") or [])]  
    
    
def collection_count() -> int:
    return get_collection().count() 