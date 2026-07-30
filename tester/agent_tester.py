"""
tester/agent_tester.py
Streamlit untuk menguji SELURUH sistem lewat API — chat, grafik,
sumber, rute, dan waktu tanggap dalam satu layar.

Menguji lewat HTTP, bukan memanggil fungsi Python langsung, adalah
keputusan sengaja: yang diuji harus persis yang dipakai pengguna
sungguhan, termasuk serialisasi respons dan penanganan error di route.

Jalankan (API harus sudah hidup di terminal lain):
    streamlit run tester/agent_tester.py
"""

from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional

import plotly.graph_objects as go
import requests
import streamlit as st

API_BASE_URL = "http://localhost:8001/api/v1"
REQUEST_TIMEOUT_SECONDS = 120

FORECAST_COLOR = "#e76f51"
ACTUAL_COLOR = "#264653"
CATEGORY_COLOR = "#2a9d8f"

SAMPLE_QUESTIONS = [
    "halo",
    "Tolong prediksi revenue 3 minggu ke depan beserta rincian mingguannya",
    "Berapa jumlah booking bulan lalu?",
    "Destinasi mana yang revenue-nya tertinggi tahun ini?",
    "Bagaimana tren revenue bulanan kita?",
    "Bagaimana sebaran nilai transaksi pelanggan?",
    "Kenapa banyak pelanggan kami churn?",
    "Sebaiknya bagaimana strategi retensi untuk pelanggan risiko tinggi?",
    "Bagaimana performa per kanal booking?",
    "Bagaimana resep rendang yang enak?",
]

st.set_page_config(page_title="Jojoba Agent Tester", page_icon="🧪",
                   layout="wide")


def call_chat_api(question: str, thread_id: Optional[str],
                  language: str) -> Dict[str, Any]:
    response = requests.post(
        f"{API_BASE_URL}/chat",
        json={"question": question, "thread_id": thread_id,
              "language": language},
        timeout=REQUEST_TIMEOUT_SECONDS)
    response.raise_for_status()
    return response.json()


def fetch_chat_status() -> Dict[str, Any]:
    response = requests.get(f"{API_BASE_URL}/chat/status", timeout=15)
    return {"status_code": response.status_code, "body": response.json()}


def render_chart(chart_spec: Dict[str, Any]) -> None:
    """Render spesifikasi grafik dari agent. Deret proyeksi digambar
    putus-putus supaya tidak pernah tertukar dengan data aktual."""
    chart_type = chart_spec["chart_type"]
    figure = go.Figure()

    if chart_type == "pie":
        first_series = chart_spec["series"][0]
        figure.add_trace(go.Pie(
            labels=[point["x_value"] for point in first_series["points"]],
            values=[point["y_value"] for point in first_series["points"]],
            hole=0.5))
    else:
        for series in chart_spec["series"]:
            x_values = [point["x_value"] for point in series["points"]]
            y_values = [point["y_value"] for point in series["points"]]
            is_forecast = series.get("is_forecast", False)

            if chart_type == "bar":
                figure.add_trace(go.Bar(x=x_values, y=y_values,
                                        name=series["name"],
                                        marker_color=CATEGORY_COLOR))
            else:
                figure.add_trace(go.Scatter(
                    x=x_values, y=y_values, name=series["name"],
                    mode="lines+markers",
                    fill="tozeroy" if chart_type == "area" else None,
                    line=dict(
                        color=FORECAST_COLOR if is_forecast else ACTUAL_COLOR,
                        dash="dash" if is_forecast else "solid",
                        width=2.5)))

    figure.update_layout(
        title=chart_spec.get("title", ""),
        xaxis_title=chart_spec.get("x_label", ""),
        yaxis_title=chart_spec.get("y_label", ""),
        height=380, margin=dict(t=50, b=20),
        legend=dict(orientation="h", y=1.12))
    st.plotly_chart(figure, use_container_width=True)

    if chart_spec.get("note"):
        st.caption(chart_spec["note"])

def render_navigation(navigation: dict) -> None:
    target = navigation.get("target")
    if target:
        query = "&".join(f"{key}={value}"
                         for key, value in target["query_params"].items())
        full_path = f"{target['dashboard_path']}?{query}" if query else target["dashboard_path"]
        st.info(f"📍 Buka **{target['label']}** → `{full_path}`")
    alternatives = navigation.get("alternatives") or []
    if alternatives:
        st.caption("Halaman terkait: " + " • ".join(
            option["label"] for option in alternatives))


def render_diagnostics(result: Dict[str, Any]) -> None:
    """Panel yang membuat perilaku agent bisa diperiksa, bukan ditebak."""
    first, second, third, fourth = st.columns(4)
    first.metric("Intent", result.get("intent", "-"))
    second.metric("Waktu", f"{result.get('elapsed_seconds', 0):.2f}s")
    third.metric("Tools", ", ".join(result.get("tools_used") or []) or "-")
    fourth.metric("Prompt", result.get("prompt_version", "-"))
    

    badges: List[str] = []
    if result.get("contains_forecast"):
        badges.append("⚠️ memuat proyeksi")
    if result.get("degraded"):
        badges.append("🔻 mode terdegradasi")
    if badges:
        st.warning(" • ".join(badges))

    sources = result.get("sources") or []
    if sources:
        with st.expander(f"📚 {len(sources)} sumber dipakai"):
            for source in sources:
                st.markdown(
                    f"**[{source.get('source_type', '?')}]** "
                    f"`{source.get('id', '?')}` — skor "
                    f"{source.get('score', 0):.3f}")
                st.caption(source.get("text", "")[:300])


# --- Sidebar -------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🧪 Agent Tester")

    if st.button("Cek kesiapan sistem"):
        try:
            status = fetch_chat_status()
            if status["status_code"] == 200:
                st.success("Sistem siap")
                st.json(status["body"])
            else:
                st.error(f"HTTP {status['status_code']}")
                st.json(status["body"])
        except requests.RequestException as error:
            st.error(f"API tidak terjangkau: {error}")

    language = st.radio("Bahasa", ["id", "en"], horizontal=True)

    st.divider()
    st.caption("Percakapan")
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = str(uuid.uuid4())
    st.code(st.session_state.thread_id, language=None)
    if st.button("Mulai percakapan baru"):
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.chat_history = []
        st.rerun()

    st.divider()
    st.caption("Contoh pertanyaan")
    for sample in SAMPLE_QUESTIONS:
        if st.button(sample, key=f"sample_{sample[:24]}"):
            st.session_state.pending_question = sample
            st.rerun()

# --- Riwayat percakapan --------------------------------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.title("Jojoba Travel AI Advisor — Agent Tester")

for turn in st.session_state.chat_history:
    with st.chat_message("user"):
        st.write(turn["question"])
    with st.chat_message("assistant"):
        st.write(turn["result"].get("answer", ""))
        if turn["result"].get("chart_spec"):
            render_chart(turn["result"]["chart_spec"])
        render_diagnostics(turn["result"])

pending_question = st.session_state.pop("pending_question", None)
typed_question = st.chat_input("Tanya apa saja tentang bisnis Jojoba...")
question = pending_question or typed_question

if question:
    with st.chat_message("user"):
        st.write(question)
    with st.chat_message("assistant"):
        with st.spinner("Agent sedang bekerja..."):
            try:
                result = call_chat_api(question,
                                       st.session_state.thread_id, language)
            except requests.RequestException as error:
                st.error(f"Permintaan gagal: {error}")
                result = None

        if result is not None:
            st.write(result.get("answer", ""))
            if result.get("navigation"):
                render_navigation(result["navigation"])
            if result.get("chart_spec"):
                render_chart(result["chart_spec"])
            render_diagnostics(result)
            st.session_state.chat_history.append(
                {"question": question, "result": result})
