"""
test/ManualCodeTest/test_feature_builder_manual.py
Uji feature_builder 5 level, puncaknya uji paritas train vs serve.

Kriteria kelulusan yang TEGAS (tidak ada jalur abu-abu):
  max diff <  0.01  -> LOLOS
  max diff >= 0.01  -> GAGAL, dengan laporan tersangka

Catatan paritas: referensi adalah exports/churn_scores_all_customers.csv
hasil notebook, yang di-skor pada TRAINING_SNAPSHOT. Kalau kamu
meregenerate artefak (retrain), regenerate juga CSV-nya — membandingkan
model baru dengan skor model lama pasti gagal dan itu bukan bug.

Jalankan dari root: python test/ManualCodeTest/test_feature_builder_manual.py
"""

import sys
import asyncio
import logging
from pathlib import Path
from datetime import datetime, timezone

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
logging.basicConfig(level=logging.INFO)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from motor.motor_asyncio import AsyncIOMotorClient  # noqa: E402

from src.config.settings import get_settings  # noqa: E402
from src.services.ml.artifact_loader import load_artifacts  # noqa: E402
from src.services.ml.feature_builder import (  # noqa: E402
    align_features, build_churn_features)

# Snapshot yang dipakai notebook saat menghasilkan CSV referensi.
# Satu konstanta eksplisit — bukan menebak dari kolom CSV yang tidak ada.
TRAINING_SNAPSHOT = datetime(2025, 7, 16, tzinfo=timezone.utc)
PARITY_THRESHOLD = 0.01
REF_CSV = PROJECT_ROOT / "src/artifacts/exports/churn_scores_all_customers.csv"


async def main():
    settings = get_settings()
    db = AsyncIOMotorClient(settings.mongo_url)[settings.database_name]
    art = load_artifacts()

    # --- LEVEL 1: jalan & tidak kosong -----------------------------------
    feat = await build_churn_features(db)
    assert len(feat) > 0, "Kosong — cek koneksi/nama collection"
    print(f"[1] Build OK: {feat.shape}")

    # --- LEVEL 2: aturan domain ------------------------------------------
    assert (feat["recency_days"] >= 0).all(), "Ada recency negatif!"
    assert (feat["frequency"] >= 1).all()
    assert set(feat["track"].unique()) <= {"single", "repeat"}
    num = feat.select_dtypes("number")
    assert np.isfinite(num.fillna(0).to_numpy()).all(), "Masih ada inf!"
    print(f"[2] Sanity OK. Track: {feat['track'].value_counts().to_dict()}")

    # --- LEVEL 3: alignment sesuai kontrak -------------------------------
    X = align_features(feat, art.churn_features)
    assert list(X.columns) == art.churn_features, "Urutan kolom beda!"
    print(f"[3] Alignment OK: {X.shape}")

    # --- LEVEL 4: end-to-end prediksi ------------------------------------
    proba = art.churn_model.predict_proba(X)[:, 1]
    assert ((proba >= 0) & (proba <= 1)).all()
    print(f"[4] Prediksi OK. Mean proba: {proba.mean():.3f}")

    # --- LEVEL 5: PARITAS train vs serve ---------------------------------
    ref = pd.read_csv(REF_CSV, index_col="customer_id")["churn_proba"]
    ref.index = ref.index.astype(str).str.strip()

    feat_t = await build_churn_features(db, snapshot=TRAINING_SNAPSHOT)
    proba_t = art.churn_model.predict_proba(
        align_features(feat_t, art.churn_features))[:, 1]
    serving = pd.Series(proba_t, index=feat_t.index.astype(str).str.strip())

    common = serving.index.intersection(ref.index)
    assert len(common) > 0.9 * len(ref), (
        f"Overlap populasi hanya {len(common):,}/{len(ref):,} — "
        f"kemungkinan snapshot salah atau filter status berbeda.")

    diff = (serving.loc[common] - ref.loc[common]).abs()
    print(f"[5] Paritas {len(common):,} customer | "
          f"max {diff.max():.6f} | mean {diff.mean():.6f} | "
          f">{PARITY_THRESHOLD}: {(diff > PARITY_THRESHOLD).sum():,}")

    if diff.max() >= PARITY_THRESHOLD:
        worst = diff.nlargest(5)
        print("\nTersangka terburuk (bandingkan fitur vs notebook!):")
        cols = ["track", "frequency", "recency_days", "avg_interval",
                "recency_ratio", "has_review", "review_sentiment"]
        print(feat_t.loc[worst.index, [c for c in cols
                                       if c in feat_t.columns]])
        raise AssertionError(
            f"SKEW: max diff {diff.max():.4f} >= {PARITY_THRESHOLD}. "
            f"Bila kamu baru retrain, regenerate dulu CSV referensinya.")

    print("\nSEMUA LEVEL LOLOS — feature_builder identik dengan training.")


if __name__ == "__main__":
    asyncio.run(main())
