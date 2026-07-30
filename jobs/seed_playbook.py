"""
jobs/seed_playbook.py

Memasukkan chunk business playbook (hasil review manusia) ke collection
`business_knowledge`. Dijalankan manual setiap kali playbook diedit:

    python -m jobs.seed_playbook

Pola upsert per-id: chunk yang diedit tertimpa, yang dihapus dari file
ikut dihapus dari collection — file JSON adalah sumber kebenaran konten,
MongoDB adalah sumber kebenaran runtime. Indexer RAG membaca collection
ini bersama ml_insights; field source_type membedakan FAKTA-DATA
(insight) dari KEBIJAKAN (playbook) di prompt Synthesizer.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from pymongo import MongoClient, ReplaceOne

from src.config.settings import get_settings

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("seed_playbook")

PLAYBOOK_FILE = Path("business_playbook.json")
COLLECTION = "business_knowledge"


def main() -> None:
    settings = get_settings()
    client = MongoClient(settings.mongo_url, serverSelectionTimeoutMS=8000)
    client.admin.command("ping")
    db = client[settings.database_name]

    with open(PLAYBOOK_FILE, encoding="utf-8") as f:
        chunks = json.load(f)

    now = datetime.now(timezone.utc).isoformat()
    ops = []
    for c in chunks:
        assert {"id", "topic", "source_type", "text"} <= set(c), \
            f"Chunk tidak lengkap: {c.get('id', '?')}"
        c["updated_at"] = now
        ops.append(ReplaceOne({"id": c["id"]}, c, upsert=True))

    result = db[COLLECTION].bulk_write(ops)

    # Chunk yang dihapus dari file ikut dihapus dari collection
    file_ids = [c["id"] for c in chunks]
    deleted = db[COLLECTION].delete_many({"id": {"$nin": file_ids}})

    logger.info("Playbook tersinkron: %d chunk (upsert %d, baru %d, "
                "dihapus %d).", len(chunks), result.modified_count,
                result.upserted_count, deleted.deleted_count)
    logger.info("Jalankan indexer RAG setelah ini agar ChromaDB ikut segar.")


if __name__ == "__main__":
    main()
