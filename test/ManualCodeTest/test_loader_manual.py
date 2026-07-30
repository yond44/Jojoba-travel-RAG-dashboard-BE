"""
test/ManualCodeTest/test_loader_manual.py
Smoke test artifact_loader — selaras dengan loader final:
forecast .pkl TIDAK dimuat serving (hasilnya dibaca dari MongoDB),
jadi yang dicek: churn, kontrak fitur, kmeans, dan keras opsional.

Jalankan dari root proyek: python test/ManualCodeTest/test_loader_manual.py
"""

import sys
import logging
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
logging.basicConfig(level=logging.INFO)

from src.services.ml.artifact_loader import load_artifacts, get_artifacts  # noqa: E402

art = load_artifacts()

print("Fitur churn      :", len(art.churn_features), "kolom")
print("Contoh fitur     :", art.churn_features[:5])
print("Model churn      :", type(art.churn_model).__name__)
print("KMeans features  :", art.kmeans["features"])
print("Keras dimuat?    :", art.churn_keras is not None)

# Singleton: panggilan kedua harus objek yang SAMA
assert get_artifacts() is art, "Singleton gagal — dimuat dua kali!"

# Model hidup: prediksi dummy tidak boleh error
import pandas as pd  # noqa: E402
dummy = pd.DataFrame([[0] * len(art.churn_features)],
                     columns=art.churn_features)
proba = art.churn_model.predict_proba(dummy)[0, 1]
print(f"Prediksi dummy   : {proba:.4f}")

# Rantai kmeans hidup: log1p -> scaler -> predict
import numpy as np  # noqa: E402
rfm = pd.DataFrame([[100, 2, 5_000_000]], columns=art.kmeans["features"])
cluster = art.kmeans["kmeans"].predict(
    art.kmeans["scaler"].transform(np.log1p(rfm)))[0]
print(f"Cluster dummy    : {int(cluster)}")

print("\nSEMUA CEK LOLOS")
