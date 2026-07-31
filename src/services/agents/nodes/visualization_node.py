from __future__ import annotations

from typing import Any, Dict, List, Optional

from langchain_core.runnables import RunnableConfig

from src.model.chart_schemas import ChartPoint, ChartSeries, ChartSpec
from src.services.agents.state import AgentState
from src.utils.log import logger

# Maksimal titik per deret. Grafik dengan ratusan titik tidak terbaca,
# dan payload-nya membebani klien tanpa menambah pemahaman.
MAX_POINTS_PER_SERIES = 60



def _normalize_period_label(raw_label: Any, granularity: str) -> str:
    text = str(raw_label)
    if granularity == "monthly" and len(text) >= 7:
        return text[:7]
    return text

def _period_point(item: Dict[str, Any], granularity: str) -> Optional[ChartPoint]:
    label = item.get("period") or item.get("period_start")
    value = item.get("revenue_idr")
    if value is None:
        value = item.get("forecast_idr")
    if label is None or value is None:
        return None
    return ChartPoint(x_value=_normalize_period_label(label, granularity),
                      y_value=float(value))


def _spec_from_revenue(revenue_result: Dict[str, Any]) -> Optional[ChartSpec]:
    segments = revenue_result.get("segments") or []
    if not segments:
        return None

    series_list: List[ChartSeries] = []
    mape_note = None

    for segment in segments:
        granularity = segment.get("granularity_used", "monthly")
        breakdown = segment.get("periods") or []
        points = [point for point in
                  (_period_point(item, granularity)
                   for item in breakdown[:MAX_POINTS_PER_SERIES])
                  if point is not None]

        if not points:
            points = [ChartPoint(
                x_value=f"{segment['start']} s/d {segment['end']}",
                y_value=float(segment["total_idr"]))]

        is_forecast = segment["kind"] == "forecast"
        series_list.append(ChartSeries(
            name="Proyeksi" if is_forecast else "Aktual",
            points=points, is_forecast=is_forecast))

        if is_forecast and segment.get("model_mape_pct") is not None:
            mape_note = (f"Deret proyeksi berasal dari model dengan "
                         f"rata-rata kesalahan {segment['model_mape_pct']}%.")

    return ChartSpec(chart_type="line", title="Revenue: aktual vs proyeksi",
                     x_label="Periode", y_label="Revenue (IDR)",
                     series=series_list, note=mape_note)
    
def _spec_from_facts(facts: Dict[str, Any]) -> Optional[ChartSpec]:
    channel_rows = facts.get("channel_performance")
    if channel_rows:
        return ChartSpec(
            chart_type="bar", title="Revenue per kanal",
            x_label="Kanal", y_label="Revenue (IDR)",
            series=[ChartSeries(name="Revenue", points=[
                ChartPoint(x_value=str(row["channel"]),
                           y_value=float(row["revenue_idr"]))
                for row in channel_rows])],
            note=f"Periode: {facts['period']['start']} s/d "
                 f"{facts['period']['end']}")

    destination_rows = facts.get("top_destinations")
    if destination_rows:
        return ChartSpec(
            chart_type="bar", title="Destinasi dengan revenue tertinggi",
            x_label="Destinasi", y_label="Revenue (IDR)",
            series=[ChartSeries(name="Revenue", points=[
                ChartPoint(x_value=str(row["destination"]),
                           y_value=float(row["revenue_idr"]))
                for row in destination_rows])])

    churn_distribution = facts.get("churn_distribution")
    if churn_distribution:
        bucket_order = ["Low", "Medium", "High", "Very High"]
        return ChartSpec(
            chart_type="pie", title="Distribusi risiko churn",
            series=[ChartSeries(name="Customer", points=[
                ChartPoint(x_value=bucket,
                           y_value=float(churn_distribution[bucket]))
                for bucket in bucket_order if bucket in churn_distribution])])

    return None


def _spec_from_eda(eda_result: Dict[str, Any]) -> Optional[ChartSpec]:
    monthly_trend = eda_result.get("monthly_revenue_trend") or []
    if not monthly_trend:
        return None
    return ChartSpec(
        chart_type="area", title="Tren revenue bulanan (historis)",
        x_label="Bulan", y_label="Revenue (IDR)",
        series=[ChartSeries(name="Revenue aktual", points=[
            ChartPoint(x_value=row["period"], y_value=float(row["revenue_idr"]))
            for row in monthly_trend[-MAX_POINTS_PER_SERIES:]])])


async def visualization_node(state: AgentState,
                             config: RunnableConfig) -> dict:
    tool_results = state.get("tool_results", {})
    next_hop_count = state.get("hop_count", 0) + 1

    chart_spec = None
    if isinstance(tool_results.get("eda"), dict):
        chart_spec = _spec_from_eda(tool_results["eda"])
    if chart_spec is None and isinstance(tool_results.get("revenue"), dict):
        chart_spec = _spec_from_revenue(tool_results["revenue"])
    if chart_spec is None and isinstance(tool_results.get("facts"), dict):
        chart_spec = _spec_from_facts(tool_results["facts"])

    if chart_spec is None:
        logger.info("Tidak ada data yang layak digambar — grafik dilewati")
        return {"chart_spec": None, "visualization_attempted": True,
                "hop_count": next_hop_count}

    logger.info("Grafik dibuat: %s (%d deret)", chart_spec.chart_type,
                len(chart_spec.series))
    return {"chart_spec": chart_spec.model_dump(mode="json"),
            "visualization_attempted": True,
            "hop_count": next_hop_count}
