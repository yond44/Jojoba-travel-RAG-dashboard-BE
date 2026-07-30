"""
jobs/retrain_churn.py

Job churn dengan DUA mode dalam satu file — karena keduanya berbagi
90% kode (pembangunan fitur) dan hanya beda otaknya:

  RETRAIN (tahunan):   python -m jobs.retrain_churn
      Latih model baru pada snapshot yang digeser maju, uji, lalu
      deploy HANYA bila tidak lebih buruk dari model lama
      (gerbang champion/challenger). Setelah deploy, langsung rescore.
      Cron: 0 3 1 1 *        (1 Januari, 03:00)

  RESCORE (harian/mingguan): python -m jobs.retrain_churn --rescore-only
      TANPA training. Muat model yang sudah ada, hitung fitur seluruh
      customer per HARI INI, tulis ulang ml_churn_scores. Model lama
      membaca data baru — murah, aman, menjaga dashboard tetap segar.
      Cron: 0 4 * * *        (tiap hari, 04:00)

Kenapa retrain churn setahunan, bukan mingguan seperti forecast:
label churn butuh 365 hari untuk matang, jadi retrain sering hanya
melatih ulang informasi yang hampir sama — sementara skor yang stabil
justru aset bagi tim marketing. Rescore-lah yang harus rajin.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("retrain_churn")

ARTIFACTS_DIR = os.getenv("ARTIFACTS_DIR", "src/artifacts")
MODELS_DIR = os.path.join(ARTIFACTS_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

VALID_STATUSES = ["Completed", "Confirmed"]
LABEL_WINDOW_DAYS = 365
CATEGORICAL = ["review_sentiment", "segment", "acquisition_channel"]
# Gerbang deploy: model baru boleh menggantikan model lama hanya bila
# AUC-nya tidak turun lebih dari toleransi ini. Retrain otomatis tanpa
# gerbang = risiko mengganti model bagus dengan model buruk tanpa sadar.
AUC_TOLERANCE = 0.01


# ---------------------------------------------------------------------------
# 1) LOAD + PARSE (pola yang sama dengan retrain_forecast)
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


def parse_dates_3tier(s: pd.Series) -> pd.Series:
    raw = s.map(lambda v: v.get("$numberLong") if isinstance(v, dict) else v)
    num = pd.to_numeric(raw, errors="coerce").where(lambda n: n > 1e11)
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
    if (out.isna() & s.notna()).mean() > 0.05:
        raise ValueError(f"Tanggal gagal parse melebihi 5% — job dihentikan.")
    return out


def load_data():
    """(bookings, reviews, customers, db|None). Mongo utama, JSON fallback."""
    uri = os.getenv("MONGO_URI")
    if uri:
        from pymongo import MongoClient
        client = MongoClient(uri, serverSelectionTimeoutMS=8000)
        client.admin.command("ping")
        db = client[os.getenv("DATABASE_NAME", "jojoba_travel")]
        def pull(name, proj):
            rows = list(db[name].find({}, proj))
            return pd.json_normalize([_mongo_flat(r) for r in rows])
        bk = pull("bookings", {"customer_id": 1, "booking_date": 1,
                               "total_price_idr": 1, "status": 1,
                               "pax_count": 1, "nights": 1})
        rv = pull("reviews", {"customer_id": 1, "review_date": 1, "rating": 1,
                              "sentiment": 1, "aspects.value_for_money": 1})
        cs = pull("customers", {"age": 1, "segment": 1,
                                "acquisition_channel": 1})
        logger.info("Loaded dari MongoDB: %d bookings, %d reviews, "
                    "%d customers", len(bk), len(rv), len(cs))
    else:
        logger.warning("MONGO_URI kosong — fallback ke file JSON (dev mode)")
        db = None
        def load(name):
            with open(f"{name}.json") as f:
                return pd.json_normalize([_mongo_flat(r) for r in json.load(f)])
        bk, rv, cs = load("bookings"), load("reviews"), load("customers")

    for df in (bk, rv, cs):
        df.columns = [c.replace(".", "_") for c in df.columns]
    bk["booking_date"] = parse_dates_3tier(bk["booking_date"])
    rv["review_date"] = parse_dates_3tier(rv["review_date"])
    bk["total_price_idr"] = pd.to_numeric(bk["total_price_idr"],
                                          errors="coerce")
    return bk, rv, cs, db


# ---------------------------------------------------------------------------
# 2) FITUR — rumus SATU-SATUNYA di job ini, dipakai training DAN rescore.
#    Salinan persis dari kontrak yang sudah lolos test paritas.
# ---------------------------------------------------------------------------
def build_features(bk: pd.DataFrame, rv: pd.DataFrame, cs: pd.DataFrame,
                   snapshot: pd.Timestamp) -> pd.DataFrame:
    valid = bk[bk["status"].isin(VALID_STATUSES)
               & (bk["booking_date"] <= snapshot)]
    g = valid.groupby("customer_id")
    feat = pd.DataFrame({
        "last_booking": g["booking_date"].max(),
        "first_booking": g["booking_date"].min(),
        "frequency": g.size(),
        "monetary_total": g["total_price_idr"].sum(),
        "monetary_avg": g["total_price_idr"].mean(),
        "pax_avg": g["pax_count"].mean(),
        "nights_avg": g["nights"].mean(),
    })
    feat["recency_days"] = (snapshot - feat["last_booking"]).dt.days
    feat["tenure_days"] = (snapshot - feat["first_booking"]).dt.days
    feat["avg_interval"] = np.where(
        feat["frequency"] > 1,
        (feat["last_booking"] - feat["first_booking"]).dt.days
        / (feat["frequency"] - 1), np.nan)
    feat["recency_ratio"] = (
        feat["recency_days"] / pd.Series(feat["avg_interval"],
                                         index=feat.index).replace(0, np.nan)
    ).replace([np.inf, -np.inf], np.nan)

    cancel = (bk[bk["booking_date"] <= snapshot]
              .assign(c=lambda d: d["status"].eq("Cancelled"))
              .groupby("customer_id")["c"].mean().rename("cancel_rate"))
    feat = feat.join(cancel)
    feat["cancel_rate"] = feat["cancel_rate"].fillna(0)

    last_rv = (rv[rv["review_date"] <= snapshot]
               .sort_values(["review_date"])
               .groupby("customer_id")
               .agg(review_rating=("rating", "last"),
                    review_sentiment=("sentiment", "last"),
                    aspect_value=("aspects_value_for_money", "mean")))
    feat = feat.join(last_rv)
    feat["has_review"] = feat["review_rating"].notna().astype(int)
    feat["review_sentiment"] = feat["review_sentiment"].fillna("No Review")

    prof = cs.set_index("_id")[["age", "segment", "acquisition_channel"]]
    feat = feat.join(prof)
    feat["track"] = np.where(feat["frequency"] == 1, "single", "repeat")
    return feat


def encode_align(feat: pd.DataFrame,
                 feature_columns: list[str] | None) -> pd.DataFrame:
    X = pd.get_dummies(feat.drop(columns=["last_booking", "first_booking",
                                          "track"]),
                       columns=CATEGORICAL)
    if feature_columns is None:          # mode training: kontrak baru lahir
        return X
    missing = [c for c in feature_columns if c not in X.columns
               and not any(c.startswith(f"{k}_") for k in CATEGORICAL)]
    if missing:
        raise ValueError(f"Fitur numerik hilang: {missing}")
    return X.reindex(columns=feature_columns, fill_value=0)


def risk_bucket(p: float) -> str:
    return ("Very High" if p >= 0.75 else "High" if p >= 0.5
            else "Medium" if p >= 0.25 else "Low")


# ---------------------------------------------------------------------------
# 3) RESCORE — dipakai kedua mode setelah model tersedia
# ---------------------------------------------------------------------------
def rescore_all(bk, rv, cs, db) -> pd.DataFrame:
    model = joblib.load(os.path.join(MODELS_DIR, "churn_model_sklearn.pkl"))
    with open(os.path.join(MODELS_DIR, "churn_features.json")) as f:
        contract = json.load(f)

    now = pd.Timestamp(datetime.now(timezone.utc)).tz_localize(None)
    feat = build_features(bk, rv, cs, snapshot=now)
    X = encode_align(feat, contract)
    proba = model.predict_proba(X)[:, 1]

    scores = pd.DataFrame({
        "customer_id": feat.index,
        "churn_proba": np.round(proba, 4),
        "risk_bucket": [risk_bucket(p) for p in proba],
        "track": feat["track"].values,
        "monetary_total": feat["monetary_total"].values,
        "tenure_days": feat["tenure_days"].values,
        "scored_at": datetime.now(timezone.utc).isoformat(),
    })
    if db is not None:
        db["ml_churn_scores"].delete_many({})
        db["ml_churn_scores"].insert_many(scores.to_dict("records"))
        logger.info("Rescore: %d customer ditulis ke ml_churn_scores",
                    len(scores))
    else:
        logger.info("Dev mode: rescore %d customer (writeback dilewati)",
                    len(scores))
    return scores


# ---------------------------------------------------------------------------
# 4) RETRAIN — dengan gerbang champion/challenger
# ---------------------------------------------------------------------------
def retrain(bk, rv, cs, db):
    now = pd.Timestamp(datetime.now(timezone.utc)).tz_localize(None)
    snapshot = now - pd.Timedelta(days=LABEL_WINDOW_DAYS)
    label_end = snapshot + pd.Timedelta(days=LABEL_WINDOW_DAYS)
    logger.info("Retrain: snapshot=%s, label window s.d. %s",
                snapshot.date(), label_end.date())

    feat = build_features(bk, rv, cs, snapshot=snapshot)

    # Label dari perilaku nyata + ritme personal (definisi yang sama
    # dengan notebook — kontrak bisnis, jangan diubah diam-diam)
    future = bk[bk["status"].isin(VALID_STATUSES)
                & (bk["booking_date"] > snapshot)
                & (bk["booking_date"] <= label_end)]
    returned = set(future["customer_id"])
    beyond = ((feat["recency_days"] + LABEL_WINDOW_DAYS)
              > 3 * feat["avg_interval"]) | feat["avg_interval"].isna()
    y = ((~feat.index.isin(returned)) & beyond).astype(int)
    logger.info("Populasi %d | churn rate %.1f%%", len(feat), y.mean() * 100)

    X = encode_align(feat, None)
    Xtr, Xte, ytr, yte, tr_track, te_track = train_test_split(
        X, y, feat["track"], test_size=0.2, stratify=y, random_state=42)

    model = HistGradientBoostingClassifier(random_state=42,
                                           class_weight="balanced")
    model.fit(Xtr, ytr)
    proba = model.predict_proba(Xte)[:, 1]
    auc = float(roc_auc_score(yte, proba))
    auc_by_track = {
        t: float(roc_auc_score(yte[te_track == t], proba[te_track == t]))
        for t in ("single", "repeat")}
    logger.info("AUC baru: %.4f (single %.4f, repeat %.4f)",
                auc, auc_by_track["single"], auc_by_track["repeat"])

    # --- Gerbang champion/challenger --------------------------------------
    prev_auc = None
    if db is not None:
        prev = db["ml_training_runs"].find_one(
            {"job": "retrain_churn", "deployed": True},
            sort=[("ran_at", -1)])
        prev_auc = prev["auc"] if prev else None

    deploy = prev_auc is None or auc >= prev_auc - AUC_TOLERANCE
    if deploy:
        joblib.dump(model,
                    os.path.join(MODELS_DIR, "churn_model_sklearn.pkl"))
        with open(os.path.join(MODELS_DIR, "churn_features.json"), "w") as f:
            json.dump(list(X.columns), f, indent=2)
        logger.info("DEPLOY: model baru menggantikan lama "
                    "(AUC %.4f vs prev %s)", auc, prev_auc)
    else:
        logger.warning("DITOLAK: AUC baru %.4f < lama %.4f - %.2f. "
                       "Model lama dipertahankan; selidiki penyebabnya "
                       "(drift data? bug fitur?).",
                       auc, prev_auc, AUC_TOLERANCE)

    run_meta = {"job": "retrain_churn",
                "ran_at": datetime.now(timezone.utc).isoformat(),
                "snapshot": str(snapshot.date()),
                "population": int(len(feat)),
                "churn_rate": round(float(y.mean()), 4),
                "auc": round(auc, 4), "auc_by_track": auc_by_track,
                "prev_auc": prev_auc, "deployed": bool(deploy)}
    if db is not None:
        db["ml_training_runs"].insert_one(run_meta)
        db["ml_insights"].replace_one(
            {"id": "churn_model_overview"},
            {"id": "churn_model_overview", "topic": "churn",
             "text": (f"[Update: {datetime.now():%d %B %Y}] Model churn "
                      f"{'diperbarui' if deploy else 'dipertahankan'} — AUC "
                      f"{auc:.3f} (single {auc_by_track['single']:.3f}, "
                      f"repeat {auc_by_track['repeat']:.3f}), churn rate "
                      f"{y.mean():.1%} dari {len(feat):,} customer.")},
            upsert=True)
    else:
        logger.info("Dev mode: %s", json.dumps(run_meta, indent=2))

    if deploy:
        rescore_all(bk, rv, cs, db)   # model baru langsung dipakai menskor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rescore-only", action="store_true",
                    help="Lewati training; hanya skor ulang populasi "
                         "dengan model yang sudah ada (jadwal harian).")
    args = ap.parse_args()

    bk, rv, cs, db = load_data()
    if args.rescore_only:
        rescore_all(bk, rv, cs, db)
    else:
        retrain(bk, rv, cs, db)


if __name__ == "__main__":
    main()
