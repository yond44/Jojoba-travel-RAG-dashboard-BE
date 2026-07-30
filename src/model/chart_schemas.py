
from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field

ChartType = Literal["line", "bar", "area", "pie", "table"]


class ChartPoint(BaseModel):
    x_value: str = Field(description="Label sumbu X, mis. '2026-08-03'")
    y_value: float


class ChartSeries(BaseModel):
    name: str = Field(description="Nama deret, mis. 'Aktual' / 'Proyeksi'")
    points: List[ChartPoint]
    is_forecast: bool = Field(
        default=False,
        description="True -> klien menggambarnya putus-putus dan memberi "
                    "label proyeksi. Fakta dan prediksi tidak boleh "
                    "terlihat sama.")


class ChartSpec(BaseModel):
    chart_type: ChartType
    title: str
    x_label: str = ""
    y_label: str = ""
    series: List[ChartSeries] = Field(default_factory=list)
    note: Optional[str] = Field(
        default=None,
        description="Catatan yang WAJIB ditampilkan klien, mis. tingkat "
                    "kesalahan model atau keterangan periode sebagian.")
