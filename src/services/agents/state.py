"""
src/services/agents/state.py
"Map kerja" yang berpindah antar node di dalam graf.

Cara membacanya: AgentState hanyalah dictionary dengan bentuk yang
disepakati. Setiap node menerima state, mengerjakan satu hal, lalu
MENGEMBALIKAN dictionary berisi field yang ingin dia ubah — LangGraph
yang menggabungkannya ke state utama. Node tidak pernah memodifikasi
state langsung; itu yang membuat alurnya bisa ditelusuri.

Satu field istimewa: `messages`. Anotasi add_messages memberitahu
LangGraph "kalau node mengembalikan messages, TAMBAHKAN ke daftar,
jangan timpa" — berbeda dari field biasa yang selalu ditimpa.
"""

from __future__ import annotations

from datetime import date
from typing import Annotated, Any, Dict, List, Optional, Sequence

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from src.utils.log import logger

# Batas lompatan antar node dalam satu pertanyaan. Pola hub-and-spoke
# (semua node kembali ke supervisor) memungkinkan multi-langkah, tapi
# juga memungkinkan berputar selamanya bila ada bug. Angka ini adalah
# rem daruratnya.
MAX_GRAPH_HOPS = 4


class AgentState(TypedDict, total=False):
    """total=False: tidak semua field wajib ada sejak awal — node
    mengisinya bertahap sepanjang perjalanan."""

    # --- Input & konteks percakapan ---
    messages: Annotated[Sequence[BaseMessage], add_messages]
    original_question: str
    language: str
    thread_id: str

    # --- Hasil Supervisor ---
    standalone_question: str
    intent: str
    params: Dict[str, Any]

    # --- Hasil kerja node spesialis ---
    tool_results: Dict[str, Any]
    sources: List[Dict[str, Any]]
    tools_used: List[str]

    chart_spec: Optional[Dict[str, Any]]
    visualization_attempted: bool

    # --- Jawaban & status ---
    final_answer: str
    contains_forecast: bool
    degraded: bool
    error_message: Optional[str]

    # --- Kendali alur ---
    hop_count: int
    is_answered: bool
    
    navigation: Optional[Dict[str, Any]]


def build_initial_state(*, question: str, language: str,
                        thread_id: str) -> AgentState:
    return AgentState(
        messages=[],
        original_question=question,
        language=language,
        thread_id=thread_id,
        standalone_question=question,
        intent="",
        params={},
        tool_results={},
        sources=[],
        tools_used=[],
        chart_spec=None,
        visualization_attempted=False,
        final_answer="",
        contains_forecast=False,
        degraded=False,
        error_message=None,
        hop_count=0,
        navigation=None,
        is_answered=False,
    )


def get_date_param(params: Dict[str, Any], key: str, default: date) -> date:
    raw_value = params.get(key)
    if isinstance(raw_value, date):
        return raw_value
    if isinstance(raw_value, str) and raw_value.strip():
        try:
            return date.fromisoformat(raw_value.strip()[:10])
        except ValueError:
            logger.warning("Tanggal '%s' pada '%s' tidak terbaca — "
                           "memakai default %s", raw_value, key, default)
    return default
