from __future__ import annotations


import logging

from pymongo import ASCENDING, DESCENDING, MongoClient

from src.config.settings import get_settings



logging.basicConfig(level=logging.INFO,
                   format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("ensure_indexes")


