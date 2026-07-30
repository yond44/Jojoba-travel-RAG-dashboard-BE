# RUNBOOK ML — Jojoba Travel

Urutan menghidupkan lapisan ML dari nol sampai terjadwal.
Semua perintah dijalankan dari **root proyek**, dengan virtualenv aktif.

---

## 0. Persiapan `.env`

```env
MONGO_URI=mongodb+srv://...          # kredensial BARU (yang lama dianggap bocor)
DATABASE_NAME=jojoba_travel
ARTIFACTS_DIR=src/artifacts          # satu lokasi untuk notebook, jobs, dan API
VIRTUAL_TODAY=2026-07-16             # demo porto: jam bisnis dipatok ke akhir data.
                                     # Produksi nyata: hapus baris ini.
```

Pastikan `.gitignore` memuat:

```
src/artifacts/models/*
src/artifacts/exports/*
!src/artifacts/models/churn_features.json
```

---

## 1. FIRST TRAIN — dijalankan SEKALI, lewat notebook

Notebook `jojoba_ml_master_fixed.ipynb` adalah initial trainer lengkap:
churn (sklearn + keras), K-Means, forecast weekly/monthly, seluruh
collection MongoDB, dan file exports.

```bash
pip install jupyter ipykernel
# jalankan Jupyter DENGAN env var yang sama dengan aplikasi.
# Linux/Mac:
ARTIFACTS_DIR=src/artifacts jupyter notebook
# Windows PowerShell:
#   $env:ARTIFACTS_DIR="src/artifacts"; jupyter notebook
# lalu buka notebook -> Run All -> isi MongoDB URI saat diminta
```

Hasil yang harus ada setelah selesai:

| Lokasi | Isi |
|---|---|
| `src/artifacts/models/` | churn_model_sklearn.pkl, churn_features.json, churn_model.keras + churn_keras_preprocessor.pkl, kmeans_segmentation.pkl, revenue_forecast_weekly.pkl, revenue_forecast_monthly.pkl |
| `src/artifacts/exports/` | cleaning_audit.json, insights_for_rag.json, semua CSV |
| MongoDB | ml_churn_scores, ml_customer_segments, ml_forecast_results, ml_insights |

## 2. Lengkapi horizon daily + baseline riwayat training

```bash
python -m jobs.retrain_forecast
```

Menambahkan `revenue_forecast_daily.pkl` ke models/, dokumen horizon
`daily` di ml_forecast_results, dan entri pertama `ml_training_runs`.

## 3. Baseline churn di ml_training_runs (sekali)

```bash
python -m jobs.retrain_churn
```

Run pertama otomatis deploy (belum ada champion). Setelah ini, retrain
berikutnya harus MENGALAHKAN run ini (toleransi AUC 0.01) untuk boleh
menggantikan model.

## 4. Verifikasi serving

```bash
python test/ManualCodeTest/test_loader_manual.py
uvicorn src.main:app --reload
```

Cek di Swagger `/docs`:
- `GET /api/v1/revenue?start=2026-06-01&end=2026-08-31` → dua segmen
  (actual + forecast), `contains_forecast: true`
- `POST /api/v1/predict/churn` dengan 2-3 id asli dari ml_churn_scores
- `GET /api/v1/forecast?horizon=daily`

## 4b. (Opsional, direkomendasikan) Hidupkan simulator data harian

Simulator menyuntikkan aktivitas bisnis sintetis per hari dengan hukum
relasi yang dipelajari dari data nyata (lihat docstring
`jobs/simulate_daily_data.py`). Dengan ini cron benar-benar punya data
baru untuk dikerjakan.

```bash
python -m jobs.simulate_daily_data --dry-run   # audit dulu tanpa menulis
python -m jobs.simulate_daily_data             # backfill s.d. kemarin
```

PENTING — interaksi dengan VIRTUAL_TODAY: begitu simulator dipakai,
**hapus `VIRTUAL_TODAY` dari `.env`**. Keduanya saling meniadakan:
jam beku membuat simulator berpikir tidak ada hari yang perlu diisi,
dan simulator membuat jam beku tidak diperlukan (datanya kini mengalir
sungguhan). Pilih SATU: jam virtual (statis, nol perawatan) ATAU
simulator (hidup, cron bermakna).

Semua dokumen sintetis bertanda `"source": "simulator"` — bisa
dibersihkan kapan pun dengan `delete_many({"source": "simulator"})`
di keempat collection.

## 5. Penjadwalan (cron)

```cron
# simulator data — harian 03:00, SEBELUM rescore (mode simulator saja)
0 3 * * *  cd /path/proyek && .venv/bin/python -m jobs.simulate_daily_data
# retrain forecast — mingguan, Senin 02:00
0 2 * * 1  cd /path/proyek && .venv/bin/python -m jobs.retrain_forecast
# rescore churn — harian 04:00 (model lama, data baru; BUKAN retrain)
0 4 * * *  cd /path/proyek && .venv/bin/python -m jobs.retrain_churn --rescore-only
# retrain churn — tahunan 1 Jan 03:00, dengan gerbang champion/challenger
0 3 1 1 *  cd /path/proyek && .venv/bin/python -m jobs.retrain_churn
```

Windows: Task Scheduler dengan perintah yang sama.
Catatan porto: dengan `VIRTUAL_TODAY` terpasang, cukup jalankan tiap job
2-3 kali manual agar `ml_training_runs` punya riwayat sebagai bukti,
lalu dokumentasikan desain penjadwalan ini di README.

---

## Peta siapa-melatih-apa

| Model | First train | Retrain rutin | Alasan irama |
|---|---|---|---|
| Forecast daily/weekly/monthly | notebook (w/m) + jobs (d) | Mingguan (jobs) | Titik data baru langsung mengubah kurva |
| Churn sklearn | notebook | Tahunan + gerbang AUC (jobs) | Label butuh 365 hari matang; hubungan bergeser pelan |
| Churn Keras | notebook | Tidak (pembanding) | Kalah tipis dari sklearn; bukti kompetensi DL |
| K-Means segmentasi | notebook | Manual bila perlu | Definisi segmen stabil lebih bernilai bagi marketing |
| Skor churn (bukan model) | notebook | Harian (`--rescore-only`) | Fitur bergerak tiap hari meski model tetap |
