from __future__ import annotations

import argparse
import sys

from src.services.rag.indexer import reindex_corpus
import logging


logger = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--force-delete", action="store_true",
        help="Lewati ambang keamanan penghapusan (>50% index). "
             "Pakai hanya bila pemangkasan korpus memang disengaja.")
    args = parser.parse_args()

    try:
        summary = reindex_corpus(force_delete=args.force_delete)
    except RuntimeError as error:
        # Pengaman indexer sengaja melempar: job gagal DENGAN SUARA,
        # bukan diam-diam merusak index.
        logger.error("Reindex dibatalkan: %s", error)
        sys.exit(1)

    logger.info("Index siap dipakai: %d vektor total.",
                summary["total_indexed"])


if __name__ == "__main__":
    main()
