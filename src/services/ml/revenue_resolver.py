from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone

from motor.motor_asyncio import AsyncIOMotorDatabase

from src.utils.clock import business_today
from src.utils.log import logger

VALID_STATUSES = ["Completed", "Confirmed"]

DAILY_HORIZON_DAYS = 30
WEEKLY_HORIZON_DAYS = 12 * 7


def _to_dt(d: date) -> datetime:
    return datetime.combine(d, time.min)


# ---------------------------------------------------------------------------
# AKTUAL
# ---------------------------------------------------------------------------
async def _actual_total(db: AsyncIOMotorDatabase,
                        start: date, end: date) -> float:
    pipeline = [
        {"$match": {
            "status": {"$in": VALID_STATUSES},
            "payment_status": "Paid",
            "booking_date": {"$gte": _to_dt(start),
                             "$lt": _to_dt(end + timedelta(days=1))},
        }},
        {"$group": {"_id": None, "total": {"$sum": "$total_price_idr"}}},
    ]
    rows = await db["bookings"].aggregate(pipeline).to_list(1)
    return float(rows[0]["total"]) if rows else 0.0


async def _actual_segment(db: AsyncIOMotorDatabase,
                          start: date, end: date) -> dict:
    """Fakta + insight terhitung: vs periode sebelumnya yang sama panjang
    (momentum) dan vs periode sama tahun lalu (musiman)."""
    n_days = (end - start).days + 1
    total = await _actual_total(db, start, end)
    prev_total = await _actual_total(
        db, start - timedelta(days=n_days), start - timedelta(days=1))
    yoy_total = await _actual_total(
        db, start - timedelta(days=365), end - timedelta(days=365))

    def growth(now: float, base: float) -> float | None:
        return round((now - base) / base * 100, 1) if base > 0 else None

    return {
        "kind": "actual",
        "start": str(start), "end": str(end), "days": n_days,
        "total_idr": total,
        "avg_daily_idr": round(total / n_days, 2),
        "vs_previous_period_pct": growth(total, prev_total),
        "vs_same_period_last_year_pct": growth(total, yoy_total),
    }


# ---------------------------------------------------------------------------
# FORECAST
# ---------------------------------------------------------------------------
def _overlap_fraction(p_start: date, p_end: date,
                      q_start: date, q_end: date) -> float:
    lo, hi = max(p_start, q_start), min(p_end, q_end)
    if lo > hi:
        return 0.0
    return ((hi - lo).days + 1) / ((p_end - p_start).days + 1)


def _period_end(period_start: date, horizon: str) -> date:
    if horizon == "daily":
        return period_start
    if horizon == "weekly":
        return period_start + timedelta(days=6)
    nxt = (period_start.replace(day=28) + timedelta(days=4)).replace(day=1)
    return nxt - timedelta(days=1)


def _preferred_granularities(span_days: int) -> list[str]:
    """Urutan preferensi granularitas untuk rentang ini. Berbentuk daftar
    supaya sistem tetap hidup bila model daily BELUM dilatih (keputusan
    yang sengaja dibuka): preferensi pertama kosong -> jatuh ke berikutnya,
    dengan pro-rata menjaga jawabannya tetap masuk akal."""
    if span_days <= DAILY_HORIZON_DAYS:
        return ["daily", "weekly", "monthly"]
    if span_days <= WEEKLY_HORIZON_DAYS:
        return ["weekly", "monthly"]
    return ["monthly"]


async def _forecast_segment(db: AsyncIOMotorDatabase,
                            start: date, end: date) -> dict:
    span = (end - start).days + 1

    docs, horizon = [], None
    for candidate in _preferred_granularities(span):
        docs = await (db["ml_forecast_results"]
                      .find({"horizon": candidate})
                      .sort("period", 1).to_list(None))
        if docs:
            horizon = candidate
            break
    if not docs:
        raise RuntimeError("ml_forecast_results kosong — jalankan "
                           "jobs/retrain_forecast dulu (lihat RUNBOOK_ML.md).")

    total, covered_until, prorated = 0.0, None, False
    period_breakdown = []          

    for document in docs:
        period_start = date.fromisoformat(document["period"])
        period_end = _period_end(period_start, horizon)
        overlap = _overlap_fraction(period_start, period_end, start, end)
        if overlap > 0:
            contribution = document["forecast_idr"] * overlap
            total += contribution
            period_breakdown.append({
                "period_start": str(period_start),
                "period_end": str(period_end),
                "forecast_idr": round(contribution, 2),
                "partial": overlap < 1.0,
            })
            covered_until = max(covered_until or period_end,
                                min(period_end, end))
            if overlap < 1.0:
                prorated = True

    seg = {
        "kind": "forecast",
        "start": str(start), "end": str(end), "days": span,
        "granularity_used": horizon,
        "total_idr": round(total, 2),
        "model_mape_pct": float(docs[0]["model_mape_pct"]),
        "generated_at": docs[0]["generated_at"],
        "prorated_edges": prorated,
        "periods": period_breakdown,
    }
    if covered_until is None or covered_until < end:
        seg["warning"] = (
            f"Forecast hanya tersedia s.d. {covered_until}. Sisanya di luar "
            f"horizon model — perpanjang horizon di jobs/retrain_forecast "
            f"bila periode ini sering ditanyakan.")
    return seg


# ---------------------------------------------------------------------------
# RESOLVER UTAMA
# ---------------------------------------------------------------------------
async def resolve_revenue(db: AsyncIOMotorDatabase,
                          start: date, end: date) -> dict:
    """Hari INI masuk sisi forecast: harinya belum selesai, angka aktualnya
    masih berubah — menyebutnya 'aktual' menyesatkan pemakai."""
    if start > end:
        raise ValueError(f"start ({start}) melewati end ({end}).")

    today = business_today()
    segments = []

    if start < today:
        segments.append(await _actual_segment(
            db, start, min(end, today - timedelta(days=1))))
    if end >= today:
        segments.append(await _forecast_segment(db, max(start, today), end))

    result = {
        "query": {"start": str(start), "end": str(end)},
        "resolved_at": datetime.now(timezone.utc).isoformat(),
        "segments": segments,
        "total_idr": round(sum(s["total_idr"] for s in segments), 2),
        "contains_forecast": any(s["kind"] == "forecast" for s in segments),
    }
    logger.info("Revenue resolved %s..%s: %s", start, end,
                "+".join(s["kind"] for s in segments))
    return result
