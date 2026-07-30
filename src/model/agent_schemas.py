from __future__ import annotations

from datetime import date, datetime
from typing import Any, Dict, List, Literal, Optional
import uuid

from pydantic import BaseModel, Field, field_validator

IntentType = Literal[
    "greeting",
    "raw_fact",
    "entity_prediction",
    "insight",
    "exploratory",
    "navigation",
    "out_of_scope",
]

class SupervisorParams(BaseModel):
    tools: List[str] = Field(default_factory=list)
    target_view: Optional[str] = None
    start_date: Optional[date] = Field(
        default=None, description="Awal rentang waktu bila disebut")
    end_date: Optional[date] = Field(
        default=None, description="Akhir rentang waktu bila disebut")
    customer_ids: List[str] = Field(
        default_factory=list, description="ID customer bila disebut")
    metric: Optional[str] = Field(
        default=None, description="Metrik yang diminta, mis. 'revenue'")
    notes: str = Field(default="", description="Catatan bebas Supervisor")


class SupervisorDecision(BaseModel):
    standalone_question: str = Field(
        description="Pertanyaan utuh setelah kata ganti diselesaikan "
                    "dari riwayat percakapan")
    intent: IntentType
    params: SupervisorParams = Field(default_factory=SupervisorParams)


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    thread_id: Optional[str] = Field(
        default=None,
        description="ID percakapan. Kosong = mulai percakapan baru.")
    language: Optional[Literal["id", "en"]] = None


class ChatResponse(BaseModel):
    question: str
    answer: str
    thread_id: str
    intent: str
    language: str
    sources: List[Dict[str, Any]] = Field(default_factory=list)
    tools_used: List[str] = Field(default_factory=list)
    chart_spec: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Spesifikasi grafik siap render. None bila jawaban "
                    "ini tidak punya data yang layak digambar.")
    contains_forecast: bool = Field(
        default=False,
        description="True bila jawaban memuat angka proyeksi — frontend "
                    "wajib menampilkannya sebagai label, bukan fakta.")
    degraded: bool = False
    elapsed_seconds: float
    prompt_version: str
    navigation: Optional[Dict[str, Any]] = None
    answered_at: datetime


    @field_validator("question")
    @classmethod
    def clean_question(cls, value: str) -> str:
        cleaned = "".join(character for character in value
                          if character.isprintable() or character in "\n\t").strip()
        if not cleaned:
            raise ValueError("question cannot be empty")
        return cleaned

    @field_validator("thread_id")
    @classmethod
    def validate_thread_id(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        try:
            return str(uuid.UUID(value))
        except ValueError:
            raise ValueError("thread_id must be a valid UUID")