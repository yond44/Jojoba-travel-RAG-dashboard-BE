from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Dict

from src.config.settings import get_settings
from src.utils.log import logger


class DailyBudgetExceededError(Exception):
    pass


class DailyRequestBudget:
    def __init__(self, max_requests_per_day: int):
        self.max_requests_per_day = max_requests_per_day
        self.current_date = datetime.now(timezone.utc).date()
        self.requests_today = 0
        self.lock = asyncio.Lock()

    # ---------- Konsumsi kuota ----------
    async def consume(self, amount: int = 1) -> None:
        async with self.lock:
            today = datetime.now(timezone.utc).date()
            if today != self.current_date:
                logger.info("Kuota harian direset: %s -> %s (%d terpakai)",
                            self.current_date, today, self.requests_today)
                self.current_date = today
                self.requests_today = 0

            if self.requests_today + amount > self.max_requests_per_day:
                raise DailyBudgetExceededError(
                    f"Kuota demo harian ({self.max_requests_per_day} "
                    f"permintaan) sudah habis. Coba lagi besok.")

            self.requests_today += amount

    # ---------- Pelaporan ----------
    async def snapshot(self) -> Dict[str, object]:
        async with self.lock:
            return {
                "date": str(self.current_date),
                "requests_today": self.requests_today,
                "max_requests_per_day": self.max_requests_per_day,
                "remaining": max(
                    self.max_requests_per_day - self.requests_today, 0),
            }


chat_daily_budget = DailyRequestBudget(get_settings().chat_daily_budget)
