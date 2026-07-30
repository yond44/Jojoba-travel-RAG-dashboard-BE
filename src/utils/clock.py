"""
src/utils/clock.py

Satu-satunya sumber "hari ini" untuk LOGIKA BISNIS (garis pemisah
aktual vs forecast, snapshot fitur). Timestamp metadata (scored_at,
generated_at) TIDAK lewat sini — itu jam mesin, biarkan asli.

Mode demo/porto : set VIRTUAL_TODAY=2026-07-16 di .env
                  -> sistem membeku di tanggal terakhir data, resolver
                     selalu punya masa lalu DAN masa depan untuk demo.
Mode produksi   : kosongkan VIRTUAL_TODAY -> tanggal sungguhan.
"""

import os
from datetime import date, datetime, timezone


def business_today() -> date:
    pinned = os.getenv("VIRTUAL_TODAY")
    return date.fromisoformat(pinned) if pinned else \
        datetime.now(timezone.utc).date()
