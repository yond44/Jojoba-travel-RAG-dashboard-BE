"""
jobs/retrain_forecast.py

Trainer forecast revenue TERJADWAL — dijalankan cron mingguan.
Melatih TIGA model granularitas (daily, weekly, monthly) dari data
bookings terbaru, lalu:
  1. Menyimpan model .pkl ke ARTIFACTS_DIR/models
  2. Meng-upsert hasil prediksi ke collection ml_forecast_results
  3. Memperbarui chunk insight forecast di ml_insights
  4. Mencatat metadata run ke ml_training_runs (riwayat, untuk audit)

Yearly TIDAK punya model sendiri: dia agregat 12 dokumen monthly —
keputusan desain "latih di granularitas halus, agregasi ke atas".

Jalankan manual : python -m jobs.retrain_forecast
Cron mingguan   : 0 2 * * 1 cd /path/proyek && /path/.venv/bin/python -m jobs.retrain_forecast
Churn TIDAK diretrain di sini (kesepakatan: model churn stabil;
yang berkala adalah re-scoring, di jobs terpisah).
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("retrain_forecast")

# --- Konfigurasi lokasi: SATU sumber dengan aplikasi (env yang sama) ------
ARTIFACTS_DIR = os.getenv("ARTIFACTS_DIR", "src/artifacts")
MODELS_DIR = os.path.join(ARTIFACTS_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

VALID_STATUSES = ["Completed", "Confirmed"]

# --- Spesifikasi per granularitas -----------------------------------------
# Satu tabel konfigurasi, satu fungsi training — bukan tiga salinan kode.
# season   : lag musiman (daily 364 = hari yang sama tahun lalu, menjaga
#            keselarasan hari-dalam-minggu; weekly 52; monthly 12)
# period_fn: fitur kalender yang di-dummy-kan (daily: hari-dalam-minggu,
#            karena pola Jumat vs Selasa jauh lebih kuat daripada tanggal)
# test_n   : panjang uji out-of-time; horizon: berapa periode diprediksi maju
GRANULARITY_SPECS = {
    "daily": dict(freq="D", season=364, test_n=28, horizon=30,
                  period_fn=lambda idx: idx.dayofweek),
    "weekly": dict(freq="W-MON", season=52, test_n=12, horizon=12,
                   period_fn=lambda idx: idx.isocalendar().week.values),
    "monthly": dict(freq="MS", season=12, test_n=6, horizon=12,
                    period_fn=lambda idx: idx.month),
}


# ---------------------------------------------------------------------------
# 1) LOAD DATA — MongoDB utama, JSON fallback untuk pengembangan offline
# ---------------------------------------------------------------------------
def _mongo_flat(x):
    import datetime as _dt
    if type(x).__name__ == "ObjectId":
        return str(x)
    if isinstance(x, _dt.datetime):
        return x.isoformat()
    if isinstance(x, dict):
        if "$oid" in x:
            return x["$oid"]
        if "$date" in x:
            return x["$date"]
        return {k: _mongo_flat(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_mongo_flat(v) for v in x]
    return x


def load_bookings():
    """Kembalikan (DataFrame bookings, handle db|None)."""
    uri = os.getenv("MONGO_URI")
    if uri:
        from pymongo import MongoClient
        client = MongoClient(uri, serverSelectionTimeoutMS=8000)
        client.admin.command("ping")
        db = client[os.getenv("DATABASE_NAME", "jojoba_travel")]
        rows = list(db["bookings"].find(
            {}, {"booking_date": 1, "total_price_idr": 1,
                 "status": 1, "payment_status": 1}))
        logger.info("Loaded %d bookings dari MongoDB", len(rows))
        return pd.json_normalize([_mongo_flat(r) for r in rows]), db
    # Fallback offline: file export JSON di working dir
    with open("bookings.json") as f:
        rows = json.load(f)
    logger.warning("MONGO_URI kosong — fallback ke bookings.json (dev mode)")
    return pd.json_normalize([_mongo_flat(r) for r in rows]), None


def parse_dates_3tier(s: pd.Series) -> pd.Series:
    """Parser bertingkat yang sama dengan notebook: epoch ms -> mixed ->
    dayfirst. Data tanggal campuran sudah pernah menghanguskan 34% baris;
    trainer terjadwal tidak boleh mengulanginya."""
    raw = s.map(lambda v: v.get("$numberLong") if isinstance(v, dict) else v)
    num = pd.to_numeric(raw, errors="coerce")
    num = num.where(num > 1e11)
    out = pd.to_datetime(num, unit="ms", errors="coerce", utc=True)
    mask = out.isna()
    if mask.any():
        out[mask] = pd.to_datetime(raw[mask], errors="coerce",
                                   utc=True, format="mixed")
    mask = out.isna() & s.notna()
    if mask.any():
        out[mask] = pd.to_datetime(raw[mask], errors="coerce",
                                   utc=True, format="mixed", dayfirst=True)
    out = out.dt.tz_localize(None)
    fail_rate = (out.isna() & s.notna()).mean()
    if fail_rate > 0.05:
        raise ValueError(f"{fail_rate:.0%} tanggal gagal parse — "
                         f"format baru terdeteksi, trainer dihentikan.")
    return out


# ---------------------------------------------------------------------------
# 2) TRAINING SATU GRANULARITAS
# ---------------------------------------------------------------------------
def train_one(ts: pd.Series, name: str, spec: dict) -> dict:
    """Latih Ridge untuk satu granularitas.

    Gerbang kualitas: model HARUS mengalahkan seasonal naive di uji
    out-of-time. Kalau kalah, kita simpan penanda use_naive=True dan
    forecast memakai nilai periode-sama-tahun-lalu — jujur lebih baik
    daripada model yang kalah dari tebakan sederhana.
    """
    d = pd.DataFrame({"y": ts})
    d["lag1"] = d["y"].shift(1)
    d["lag_s"] = d["y"].shift(spec["season"])
    d["t"] = np.arange(len(d))
    d["period"] = spec["period_fn"](d.index)
    d = pd.get_dummies(d, columns=["period"], drop_first=True).dropna()

    if len(d) < spec["test_n"] * 3:
        raise ValueError(f"{name}: data terlalu pendek ({len(d)} baris) "
                         f"setelah lag {spec['season']}.")

    train, test = d.iloc[:-spec["test_n"]], d.iloc[-spec["test_n"]:]
    Xc = [c for c in d.columns if c != "y"]
    model = Ridge(alpha=1.0).fit(train[Xc], train["y"])
    pred = model.predict(test[Xc])

    mape = float(np.mean(np.abs((test["y"] - pred) / test["y"])) * 100)
    mape_naive = float(np.mean(
        np.abs((test["y"] - test["lag_s"]) / test["y"])) * 100)
    use_naive = mape >= mape_naive
    logger.info("%s | MAPE %.1f%% vs naive %.1f%% | %s",
                name, mape, mape_naive,
                "PAKAI NAIVE" if use_naive else "model lolos")

    # Forecast recursive: hasil tiap langkah jadi lag langkah berikutnya
    hist = ts.copy()
    fc = {}
    offset = {"D": pd.Timedelta(days=1), "W-MON": pd.Timedelta(weeks=1),
              "MS": pd.offsets.MonthBegin(1)}[spec["freq"]]
    for _ in range(spec["horizon"]):
        nxt = hist.index[-1] + offset
        if use_naive:
            val = float(hist.iloc[-spec["season"]])
        else:
            row = {"lag1": hist.iloc[-1],
                   "lag_s": hist.iloc[-spec["season"]],
                   "t": len(hist)}
            p = spec["period_fn"](pd.DatetimeIndex([nxt]))[0]
            for c in Xc:
                if c.startswith("period_"):
                    row[c] = 1.0 if c == f"period_{p}" else 0.0
            val = float(model.predict(pd.DataFrame([row])[Xc])[0])
        fc[nxt] = max(val, 0.0)  # revenue tidak boleh negatif
        hist.loc[nxt] = val

    return {"model": model, "features": Xc, "season": spec["season"],
            "freq": spec["freq"], "use_naive": use_naive,
            "mape": round(mape, 2), "mape_naive": round(mape_naive, 2),
            "forecast": pd.Series(fc)}



# ---------------------------------------------------------------------------
# 3) MAIN
# ---------------------------------------------------------------------------
def main():
    now_iso = datetime.now(timezone.utc).isoformat()
    bookings, db = load_bookings()

    bookings["booking_date"] = parse_dates_3tier(bookings["booking_date"])
    bookings["total_price_idr"] = pd.to_numeric(
        bookings["total_price_idr"], errors="coerce")
    rev = bookings[
        bookings["status"].isin(VALID_STATUSES)
        & (bookings["payment_status"] == "Paid")
        & bookings["booking_date"].notna()
        & bookings["total_price_idr"].notna()
    ].set_index("booking_date")["total_price_idr"]

    results, forecast_docs = {}, []
    for name, spec in GRANULARITY_SPECS.items():
        ts = rev.resample(spec["freq"]).sum().iloc[:-1]  # buang periode berjalan
        res = train_one(ts, name, spec)
        results[name] = res

        # Model + metadata dalam satu file per granularitas —
        # inilah "split beberapa model" yang bisa diretrain terpisah.
        joblib.dump(
            {k: res[k] for k in
             ("model", "features", "season", "freq", "use_naive", "mape")},
            os.path.join(MODELS_DIR, f"revenue_forecast_{name}.pkl"))

        forecast_docs += [
            {"period": str(idx.date()), "horizon": name,
             "forecast_idr": float(v), "model_mape_pct": res["mape"],
             "generated_at": now_iso}
            for idx, v in res["forecast"].items()]

    # --- Writeback: hasil prediksi + insight + riwayat run ----------------
    monthly_total = float(results["monthly"]["forecast"].sum())
    insight = {
        "id": "revenue_forecast",
        "topic": "forecast",
        "text": (f"[Update: {datetime.now():%d %B %Y}] Proyeksi revenue "
                 f"12 bulan ke depan Rp {monthly_total/1e12:.2f} triliun. "
                 f"MAPE uji out-of-time: bulanan "
                 f"{results['monthly']['mape']:.1f}%, mingguan "
                 f"{results['weekly']['mape']:.1f}%, harian "
                 f"{results['daily']['mape']:.1f}%. Tersedia prediksi "
                 f"per hari (30 hari), per minggu (12 minggu), dan "
                 f"per bulan (12 bulan) di ml_forecast_results."),
    }
    run_meta = {
        "job": "retrain_forecast", "ran_at": now_iso,
        "rows_used": int(len(rev)),
        "mape": {k: results[k]["mape"] for k in results},
        "used_naive": {k: results[k]["use_naive"] for k in results},
    }

    if db is not None:
        from pymongo import ReplaceOne
        db["ml_forecast_results"].bulk_write([
            ReplaceOne({"period": d["period"], "horizon": d["horizon"]},
                       d, upsert=True) for d in forecast_docs])
        db["ml_insights"].replace_one({"id": insight["id"]},
                                      insight, upsert=True)
        db["ml_training_runs"].insert_one(run_meta)  # append: riwayat audit
        logger.info("Writeback: %d dok forecast, insight, training run.",
                    len(forecast_docs))
    else:
        logger.warning("Dev mode: writeback dilewati. Run meta: %s",
                       json.dumps(run_meta, indent=2))


if __name__ == "__main__":
    main()
