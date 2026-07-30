from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# 1) CHURN
# ---------------------------------------------------------------------------
class ChurnPredictionRequest(BaseModel):
    customer_ids: list[str] = Field(
        min_length=1,
        max_length=100,
        description="Daftar customer_id (string hex ObjectId), 1-100 id.",
        examples=[["665f1c2b8a3d4e5f6a7b8c9d"]],
    )


class ChurnPrediction(BaseModel):
    customer_id: str
    churn_proba: float = Field(ge=0, le=1, description="Probabilitas churn 0-1")
    risk_bucket: Literal["Low", "Medium", "High", "Very High"]
    track: Literal["single", "repeat"] = Field(
        description="Jalur populasi: single booker atau repeat customer. "
                    "Penting bagi pemakai: akurasi model berbeda per jalur.")
    scored_at: datetime


class ChurnPredictionResponse(BaseModel):
    predictions: list[ChurnPrediction]
    not_found_customer_ids: list[str] = Field(
        default_factory=list,
        description="Id yang diminta tetapi tidak punya riwayat booking "
                    "valid — tidak bisa di-skor.")
    model_name: str = Field(description="Model yang dipakai (jejak audit).")


# ---------------------------------------------------------------------------
# 2) FORECAST (legacy, horizon tetap) — dipertahankan untuk kompatibilitas;
#    pintu utama pertanyaan revenue sekarang adalah /revenue (resolver).
# ---------------------------------------------------------------------------
class ForecastPeriod(BaseModel):
    period: str = Field(description="Tanggal awal periode, format YYYY-MM-DD")
    forecast_idr: float


class ForecastResponse(BaseModel):
    horizon: Literal["daily", "weekly", "monthly", "yearly"]
    periods: list[ForecastPeriod]
    total_idr: float
    model_mape_pct: float
    generated_at: datetime


# ---------------------------------------------------------------------------
# 3) REVENUE RESOLVER — jawaban rentang tanggal apa pun
# ---------------------------------------------------------------------------
class RevenueSegment(BaseModel):
    """Satu potongan jawaban. kind menentukan field mana yang terisi:
    'actual'  -> avg_daily_idr + kedua field pertumbuhan
    'forecast'-> granularity_used, model_mape_pct, generated_at, dst.
    Field di luar jenisnya bernilai None — frontend/agent cukup cek kind."""
    kind: Literal["actual", "forecast"]
    start: str
    end: str
    days: int
    total_idr: float
    # --- khusus actual ---
    avg_daily_idr: float | None = None
    vs_previous_period_pct: float | None = Field(
        default=None, description="Pertumbuhan vs periode sebelumnya yang "
                                  "sama panjang. None bila pembanding nol.")
    vs_same_period_last_year_pct: float | None = None
    # --- khusus forecast ---
    granularity_used: Literal["daily", "weekly", "monthly"] | None = None
    model_mape_pct: float | None = None
    generated_at: str | None = None
    prorated_edges: bool | None = Field(
        default=None, description="True bila dokumen tepi dihitung pro-rata "
                                  "(rentang tidak tepat di batas periode).")
    warning: str | None = Field(
        default=None, description="Terisi bila rentang melewati horizon model.")
    
    periods: list[dict] | None = Field(
        default=None,
        description="Rincian per periode (harian/mingguan/bulanan sesuai "
                    "granularity_used). None untuk segmen aktual.")


class RevenueResponse(BaseModel):
    query: dict
    resolved_at: str
    segments: list[RevenueSegment]
    total_idr: float
    contains_forecast: bool = Field(
        description="True bila ada porsi prediksi di jawaban — WAJIB "
                    "ditampilkan sebagai label oleh frontend/LLM supaya "
                    "angka model tidak menyamar jadi fakta.")


# ---------------------------------------------------------------------------
# 4) SEGMENTASI
# ---------------------------------------------------------------------------
class SegmentResponse(BaseModel):
    customer_id: str
    cluster: int
    recency_days: int
    frequency: int
    monetary_total: float
    scored_at: datetime
