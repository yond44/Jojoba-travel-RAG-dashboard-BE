# Jojoba Travel — AI Business Advisor

Sistem analitik untuk agen perjalanan fiktif "Jojoba Travel": dashboard
berisi angka bisnis, dan seorang advisor AI yang bisa ditanyai dengan
bahasa sehari-hari — dalam Bahasa Indonesia maupun Inggris.

Tanya *"berapa proyeksi revenue tiga minggu ke depan?"* dan dapat angka
beserta tingkat kesalahan modelnya. Tanya *"kenapa pelanggan kami
churn?"* dan dapat temuan dari analisis yang sudah divalidasi statistik.
Tanya *"buka halaman kinerja kampanye"* dan dashboard benar-benar
berpindah halaman.

---

## Daftar isi

- [Apa yang bisa dilakukan sistem ini](#apa-yang-bisa-dilakukan-sistem-ini)
- [Cara kerjanya, dalam bahasa sederhana](#cara-kerjanya-dalam-bahasa-sederhana)
- [Arsitektur](#arsitektur)
- [Bagian machine learning](#bagian-machine-learning)
- [Bagian RAG](#bagian-rag)
- [Bagian agent](#bagian-agent)
- [Cara sistem menghindari mengarang](#cara-sistem-menghindari-mengarang)
- [Keputusan arsitektur dan alasannya](#keputusan-arsitektur-dan-alasannya)
- [Menjalankan di komputer sendiri](#menjalankan-di-komputer-sendiri)
- [Struktur folder](#struktur-folder)
- [Job terjadwal](#job-terjadwal)
- [API](#api)
- [Pengujian](#pengujian)
- [Deployment](#deployment)
- [Batasan yang diketahui](#batasan-yang-diketahui)
- [Catatan tentang data](#catatan-tentang-data)

---

## Apa yang bisa dilakukan sistem ini

**Dashboard** — 18 halaman analitik: tren revenue, pemecahan per
destinasi/kanal/segmen, volume dan pembatalan booking, kinerja agen dan
kampanye, profil pelanggan, ulasan, serta halaman proyeksi dan risiko
churn.

**Advisor AI** — panel chat yang menempel di dashboard dan bisa:

| Jenis pertanyaan | Contoh | Sumber jawaban |
|---|---|---|
| Fakta agregat | "Berapa booking bulan lalu?" | Query MongoDB |
| Prediksi | "Proyeksi revenue 3 minggu ke depan?" | Model forecast |
| Risiko pelanggan | "Berapa risiko churn pelanggan X?" | Model churn |
| Sebab & makna | "Kenapa pelanggan kami churn?" | Korpus insight |
| Strategi | "Bagaimana strategi retensi terbaik?" | Playbook bisnis |
| Eksplorasi | "Bagaimana sebaran nilai transaksi?" | Analisis statistik |
| Navigasi | "Buka halaman kampanye" | Katalog halaman |

Jawabannya bisa disertai tabel dan grafik, dan setiap jawaban membawa
informasi teknis yang bisa diperiksa: rute yang dipilih, alat yang
dipakai, waktu proses, versi prompt, dan sumber yang dikutip beserta
skor kemiripannya.

---

## Cara kerjanya, dalam bahasa sederhana

Bayangkan sebuah restoran.

**Dapur bekerja sebelum jam buka.** Setiap malam dan setiap minggu, ada
program terjadwal yang melatih model prediksi, menghitung skor risiko
tiap pelanggan, dan menuliskan temuan analisis ke database. Ini
pekerjaan berat yang butuh menit sampai jam.

**Pelayan bekerja saat tamu datang.** Ketika seseorang bertanya, sistem
tidak menghitung ulang dari nol — dia mengambil hasil yang sudah
disiapkan dapur, lalu menyusunnya jadi jawaban. Itu sebabnya jawaban
datang dalam hitungan detik, bukan menit.

**Ada resepsionis yang menentukan arah.** Pertanyaan masuk dibaca dulu:
apakah ini soal angka masa lalu, prediksi, sebab-akibat, atau permintaan
membuka halaman? Dari situ pertanyaan diarahkan ke bagian yang tepat.

**Dan ada satu penulis untuk semua jawaban.** Bagian mana pun yang
mengerjakan, yang menulis kalimat akhirnya selalu satu — supaya gaya
bahasanya konsisten dan aturan penting (misalnya: angka proyeksi wajib
diberi label proyeksi) ditegakkan di satu tempat.

---

## Arsitektur

```
┌──────────────┐     ┌──────────────────────────────────────────┐
│   Frontend   │     │              Backend (FastAPI)           │
│ React + Vite │────▶│                                          │
│              │     │  ┌────────────┐      ┌────────────────┐  │
│  Dashboard   │     │  │   Routes   │─────▶│    Services    │  │
│  + Chat      │     │  │ (tipis)    │      │ (logika bisnis)│  │
└──────────────┘     │  └────────────┘      └───────┬────────┘  │
                     │         │                    │           │
                     │         ▼                    │           │
                     │  ┌────────────┐              │           │
                     │  │   Agents   │──────────────┘           │
                     │  │ (LangGraph)│                          │
                     │  └─────┬──────┘                          │
                     └────────┼─────────────────────────────────┘
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
        ┌──────────┐   ┌───────────┐   ┌───────────┐
        │ MongoDB  │   │ ChromaDB  │   │ Groq LLM  │
        │ (data +  │   │ (vektor   │   │ (bahasa)  │
        │  hasil)  │   │  insight) │   │           │
        └────▲─────┘   └─────▲─────┘   └───────────┘
             │               │
        ┌────┴───────────────┴─────┐
        │   Jobs terjadwal (cron)  │
        │ latih model, hitung skor,│
        │ tulis insight, indexing  │
        └──────────────────────────┘
```

Prinsip yang dijaga di sepanjang kode:

- **Routes tipis** — hanya validasi input, panggil satu fungsi service,
  petakan hasil ke status HTTP. Tidak ada logika bisnis di sana.
- **Service tidak tahu HTTP** — itu sebabnya agent bisa memanggil fungsi
  yang sama secara langsung tanpa lewat jaringan.
- **Konfigurasi terpusat** — tidak ada `os.getenv` tersebar; semuanya
  lewat `src/config/settings.py`.

---

## Bagian machine learning

Tiga model, semuanya dilatih dari data historis dan diuji secara
*out-of-time* (dilatih pada data lama, diuji pada data yang belum pernah
dilihat model).

**Forecast revenue** — regresi Ridge dengan fitur tren, lag musiman, dan
dummy periode. Tiga granularitas dengan tingkat kesalahan rata-rata
(MAPE): bulanan ~9,5%, mingguan ~7,2%, harian ~15,7%. Ketiganya
mengalahkan baseline naif.

**Prediksi churn** — HistGradientBoosting dengan fitur RFM (recency,
frequency, monetary) plus sinyal ulasan dan pembatalan. Akurasinya
berbeda tajam antar kelompok: AUC 0,93 untuk pelanggan berulang, 0,61
untuk pemesan sekali. Perbedaan ini disampaikan ke pengguna, bukan
disembunyikan — skor untuk pemesan sekali tidak boleh dipakai sendirian
sebagai dasar keputusan.

**Segmentasi pelanggan** — K-Means (k=5, silhouette 0,40) di atas fitur
RFM yang ditransformasi log.

Beberapa keputusan teknis yang menentukan kualitasnya:

- **Kolom agregat pelanggan tidak pernah jadi fitur.** Field seperti
  `total_trips` dan `total_spent` di koleksi customers sudah memuat
  informasi masa depan relatif terhadap titik prediksi — memakainya akan
  menghasilkan akurasi palsu yang runtuh di produksi.
- **Outlier dipertahankan.** Pelanggan dengan recency ekstrem bukan
  gangguan statistik; justru merekalah sinyal churn terkuat.
- **Parser tanggal berlapis dengan pemutus arus.** Data tanggal datang
  dalam beberapa format; parser mencoba tiga strategi dan berhenti
  dengan error bila lebih dari 5% gagal — daripada diam-diam mengisi
  nilai kosong.

---

## Bagian RAG

RAG (*Retrieval-Augmented Generation*) adalah cara membuat model bahasa
menjawab dari dokumen yang kita miliki, bukan dari ingatannya sendiri.

Korpusnya dua macam, dan pemisahan ini disengaja:

| Sumber | Isi | Contoh |
|---|---|---|
| `ml_insights` | Temuan dari analisis data | "Recency adalah pendorong churn terkuat" |
| `business_knowledge` | Kebijakan perusahaan | "Segmen premium tidak diberi diskon massal" |

Setiap potongan teks membawa label `source_type`, sehingga jawaban bisa
membedakan **fakta dari data** dan **kebijakan perusahaan** — dua hal
yang tidak boleh melebur jadi satu suara.

Alur pencariannya, dari yang termurah ke termahal:

1. Validasi pertanyaan (kosong? terlalu panjang?)
2. Sapaan dan ucapan terima kasih dijawab langsung — tanpa LLM
3. Cache persis (pertanyaan identik)
4. Cache semantik (pertanyaan berbeda kata, sama makna)
5. Pencarian vektor di ChromaDB
6. **Gerbang relevansi** — bila kemiripan tertinggi di bawah ambang,
   sistem menjawab "tidak tahu" dan LLM tidak pernah dipanggil
7. Panggil LLM dengan konteks yang ditemukan
8. Verifikasi jawaban terhadap konteks
9. Simpan ke cache

Indexing bersifat **inkremental**: identitas tiap potongan adalah hash
dari isinya, sehingga potongan yang tidak berubah tidak di-proses ulang.
Retrain mingguan yang hanya memperbarui dua-tiga insight hanya membayar
dua-tiga operasi, bukan seluruh korpus.

---

## Bagian agent

Enam agent yang dirangkai dengan LangGraph. Menariknya, **hanya dua yang
memanggil model bahasa** — sisanya deterministik.

| Agent | Tugas | Pakai LLM? |
|---|---|---|
| Supervisor | Menentukan arah, menulis ulang pertanyaan lanjutan | Ya |
| ML Inference | Proyeksi revenue dan skor churn | Tidak |
| Data Analyst | Fakta agregat dari database | Tidak |
| Insight RAG | Sebab, makna, strategi | Tidak (RAG punya LLM sendiri) |
| Visualization | Menyusun spesifikasi grafik | Tidak |
| Synthesizer | Menulis jawaban akhir | Ya |

Polanya *hub-and-spoke*: Supervisor adalah pusatnya, spesialis kembali
kepadanya setelah selesai, dan ada pembatas lompatan agar graf tidak
berputar tanpa henti.

**Percakapan bersambung.** Kirim `thread_id` yang sama, dan pertanyaan
seperti "kalau bulan sebelumnya?" dipahami dari konteks giliran
sebelumnya.

**Navigasi lewat chat.** Agent tidak berpindah halaman sendiri — dia
mengeluarkan *spesifikasi* berisi id halaman dan parameternya, lalu
frontend yang menjalankan. Model hanya boleh memilih dari daftar halaman
yang ada, jadi tidak mungkin mengarang rute.

---

## Cara sistem menghindari mengarang

Ini bagian yang paling banyak menyita perhatian selama pengembangan,
karena kegagalan model bahasa berbeda dari kegagalan perangkat lunak
biasa: **dia tidak melempar error, dia menjawab dengan lancar dan
salah.**

Pertahanannya berlapis:

**Angka tidak pernah berasal dari ingatan model.** Semua angka datang
dari hasil pemanggilan fungsi. Model bertugas merangkai kalimat di
sekitar angka yang sudah benar, bukan mengingatnya.

**Query database ditulis tangan.** Model memilih dari katalog agregasi
yang sudah tersedia; dia tidak pernah menulis query sendiri. Ini menutup
sekaligus dua risiko: query yang salah diam-diam, dan query yang
memberatkan database.

**Gerbang relevansi.** Bila pencarian tidak menemukan konteks yang cukup
mirip, LLM tidak dipanggil sama sekali — sistem langsung menjawab tidak
tahu. Tidak ada konteks lemah yang bisa dijadikan bahan mengarang.

**Verifikasi setelah jawaban lahir.** Tiap kalimat jawaban dicocokkan
dengan konteks yang dipakai, dan tiap angka diperiksa keberadaannya di
konteks. Jawaban yang gagal verifikasi dibuang.

**Keputusan penting dijaga kode, bukan prompt.** Aturan seperti
"pertanyaan revenue selalu ke model forecast" ditegakkan dengan kode
setelah keputusan model, karena hal yang harus selalu benar tidak boleh
bergantung pada kepatuhan model terhadap instruksi.

**Larangan mengarang tentang sistem itu sendiri.** Model dilarang
menjelaskan mengapa sistem berperilaku tertentu kecuali penjelasannya
memang tersedia — karena jenis karangan ini tidak bisa diverifikasi oleh
gerbang mana pun.

**Setiap jawaban membawa jejaknya.** Rute, alat yang dipakai, sumber
beserta skornya, dan penanda bila jawaban memuat proyeksi. Pengguna yang
tidak menguasai data pun bisa melihat dari mana jawaban itu datang.

---

## Keputusan arsitektur dan alasannya

Beberapa hal yang **sengaja tidak dipakai**, dan alasannya:

**Hybrid search (BM25 + vektor) — tersedia, tapi dimatikan.** Korpusnya
puluhan potongan atomik; pencarian vektor saja sudah menemukan
semuanya. Kodenya tetap ada di balik saklar, siap dinyalakan bila korpus
tumbuh.

**Reranking — tidak dipakai.** Berguna saat menyaring 50 kandidat dari
korpus ribuan. Mengurutkan ulang 3 dari 30 kandidat hanya menambah
latensi tanpa mengubah hasil.

**Context compression — tidak dipakai.** Potongan teksnya sudah 2–4
kalimat sejak lahir; tidak ada yang bisa dipangkas.

**Redis — tidak dipakai.** Cache terdistribusi berguna untuk banyak
replika server. Satu instance cukup dengan cache di memori.

**A/B testing dan canary — tidak dipakai.** Keduanya butuh trafik
pengguna nyata untuk dibagi. Padanan offline-nya justru ada: gerbang
champion/challenger saat melatih ulang model churn — model baru harus
membuktikan diri lebih baik sebelum menggantikan yang lama.

**Query rewriting — dipindahkan, bukan dihapus.** Fungsinya tetap ada,
tapi dikerjakan Supervisor yang memang sudah menulis ulang pertanyaan
lanjutan. Dua komponen mengerjakan hal yang sama itu duplikasi.

**Forecast hanya untuk revenue.** Metrik lain (conversion rate per
kanal, kinerja agen) hanya tersedia historis. Alasannya: revenue adalah
satu-satunya deret dengan volume dan stabilitas cukup untuk model yang
lolos uji out-of-time. Memasang model setengah matang untuk sepuluh
kanal akan menghasilkan angka yang terlihat meyakinkan tapi tidak bisa
dipertanggungjawabkan.

**Autentikasi — sengaja ditiadakan.** Ini portofolio publik; registrasi
akan menghalangi orang mencoba. Sebagai gantinya, ancaman yang nyata di
konteks ini (penyalahgunaan biaya LLM) ditangani rate limit per-IP,
kuota harian global, validasi input, dan batas ukuran permintaan.

---

## Menjalankan di komputer sendiri

### Prasyarat

- Python 3.12
- Node.js 22
- MongoDB (lokal atau Atlas)
- API key Groq (gratis di console.groq.com)

### Backend

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env             # lalu isi nilainya
```

Isi `.env` minimal:

```
MODE=dev
MONGO_URL=mongodb://127.0.0.1:27017
DATABASE_NAME=jojoba_travel
GROQ_API_KEY=gsk_...
EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
EMBEDDING_DIM=384
CHROMA_COLLECTION=jojoba_insights
CORS_ORIGINS=http://localhost:5173
PORT=8001
```

Siapkan data dan index — **urutannya penting**:

```bash
python -m jobs.ensure_indexes      # indeks database
python -m jobs.seed_playbook       # kebijakan bisnis ke MongoDB
python -m jobs.retrain_forecast    # latih model forecast
python -m jobs.retrain_churn       # latih model churn
python -m jobs.reindex_rag         # bangun index vektor
```

Jalankan:

```bash
uvicorn src.main:app --reload --port 8001
```

Buka `http://localhost:8001/docs` untuk dokumentasi API interaktif.

### Frontend

```bash
cd frontend
cp .env.example .env
npm install
npm run dev
```

Buka `http://localhost:5173`. Vite mem-proxy `/api` ke backend, jadi
tidak ada masalah CORS saat pengembangan.

### Docker

```bash
docker compose up --build
docker compose --profile jobs run --rm jobs python -m jobs.reindex_rag
```

---

## Struktur folder

```
src/
├── config/          settings terpusat, koneksi database
├── middleware/       logging, request id, rate limit, security headers
├── model/            schema Pydantic (kontrak bentuk data)
├── routes/           endpoint HTTP — tipis, tanpa logika bisnis
│   ├── agent/        endpoint chat
│   ├── dashboard/    endpoint analitik per domain
│   └── ml/           endpoint prediksi
├── services/         logika bisnis — tidak tahu HTTP sama sekali
│   ├── agents/       Supervisor, node spesialis, graf, runtime
│   ├── ml/           pemuat artefak, pembangun fitur, resolver revenue
│   ├── rag/          embeddings, vector store, indexer, retrieval, engine
│   └── ...           booking, campaign, customer, revenue, review, dst
├── utils/            logger, jam bisnis
└── artifacts/        model terlatih (.pkl) dan ekspor CSV/JSON

jobs/                 program terjadwal (cron)
frontend/             dashboard React
test/                 unit test, manual test, evaluasi
```

---

## Job terjadwal

| Job | Ritme | Fungsi |
|---|---|---|
| `retrain_forecast` | Mingguan | Melatih ulang model forecast, menulis proyeksi |
| `retrain_churn --rescore-only` | Harian | Menghitung ulang skor risiko tiap pelanggan |
| `retrain_churn` | Tahunan | Melatih ulang model churn (dengan gerbang mutu) |
| `reindex_rag` | Setelah retrain | Menyinkronkan index vektor dengan MongoDB |
| `seed_playbook` | Saat playbook diedit | Memuat kebijakan bisnis ke database |
| `ensure_indexes` | Saat deploy | Membuat indeks database |

Contoh crontab:

```cron
15 2 * * 1  cd /opt/jojoba && .venv/bin/python -m jobs.retrain_forecast
30 2 * * 1  cd /opt/jojoba && .venv/bin/python -m jobs.reindex_rag
0  3 * * *  cd /opt/jojoba && .venv/bin/python -m jobs.retrain_churn --rescore-only
```

Perbedaan penting: **melatih ulang** menghasilkan model baru;
**menghitung ulang skor** memakai model lama pada data baru. Yang kedua
jauh lebih murah dan itulah yang dibutuhkan tiap hari.

---

## API

Dokumentasi interaktif ada di `/docs`. Ringkasan endpoint utama:

| Endpoint | Fungsi |
|---|---|
| `POST /api/v1/chat` | Bertanya ke advisor |
| `GET /api/v1/chat/status` | Kesiapan sistem, kuota, versi prompt |
| `GET /api/v1/revenue` | Revenue aktual dan/atau proyeksi |
| `POST /api/v1/predict/churn` | Skor risiko pelanggan |
| `GET /api/v1/current` | Ringkasan revenue |
| `GET /api/v1/period` | Revenue per periode |
| `GET /api/v1/customers/churn-risk` | Daftar pelanggan berisiko |
| `GET /health` | Status aplikasi |

Contoh bertanya ke advisor:

```bash
curl -X POST http://localhost:8001/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "Berapa proyeksi revenue 3 minggu ke depan?",
       "thread_id": null, "language": "id"}'
```

Kirim kembali `thread_id` dari jawaban untuk melanjutkan percakapan.

---

## Pengujian

```bash
pytest test/UnitTest -v                        # cepat, tanpa dependensi luar
python test/ManualCodeTest/test_rag_manual.py  # terhadap sistem sungguhan
python test/evaluation/run_evaluation.py       # golden set
```

Unit test tidak menyentuh database maupun LLM, jadi bisa dijalankan
setiap perubahan. Evaluasi golden set menjalankan sekumpulan pertanyaan
yang jawabannya sudah diketahui bentuknya, lalu mencatat hasilnya
bersama versi prompt — sehingga pertanyaan *"apakah perubahan prompt
kemarin memperbaiki atau memperburuk?"* bisa dijawab dengan angka.

---

## Deployment

Frontend cocok di Vercel atau Netlify; backend di VPS mana pun (proyek
ini diuji di AWS Lightsail).

Kalau frontend di Vercel dan backend di VPS tanpa domain, gunakan
rewrite Vercel supaya browser hanya bicara HTTPS ke Vercel:

```json
{
  "rewrites": [
    { "source": "/api/:path*",
      "destination": "http://IP_SERVER:8001/api/:path*" }
  ]
}
```

Di sisi server, jalankan dengan systemd agar otomatis hidup kembali
setelah restart. Catatan: gunakan **satu worker** — cache, checkpointer,
dan koneksi vector store belum dibagi antar-proses.

---

## Batasan yang diketahui

Disebutkan terbuka karena mengetahui batas sistem adalah bagian dari
memahaminya:

- **State di memori.** Riwayat percakapan hilang saat aplikasi
  di-restart. Untuk produksi, checkpointer perlu dipindah ke MongoDB.
- **Satu instance.** Cache dan index vektor belum dibagi antar-replika.
- **Belum ada CI/CD, monitoring, dan alerting** yang berjalan otomatis.
- **Satu langkah per pertanyaan.** Pertanyaan yang butuh rantai
  penalaran panjang ditangani dengan pra-pengambilan periode pembanding,
  bukan perencanaan bertahap.
- **Ekstraksi tanggal bergantung pada model bahasa.** "Tiga minggu ke
  depan" diterjemahkan jadi rentang tanggal oleh LLM, dan bisa meleset
  satu-dua hari. Karena itu setiap jawaban selalu menyebutkan periode
  yang benar-benar dihitung.
- **Tidak ada autentikasi** — keputusan sadar untuk portofolio publik,
  lihat bagian keputusan arsitektur.

---

## Catatan tentang data

Data yang dipakai adalah **data sintetis** untuk perusahaan fiktif,
dibuat mengikuti pola bisnis agen perjalanan yang realistis: musiman
liburan sekolah dan akhir tahun, distribusi status pemesanan yang
bergantung pada apakah perjalanan sudah berlangsung, keterkaitan yang
konsisten antara pemesanan, pembayaran, dan ulasan.

Isi *business playbook* bersifat ilustratif untuk mendemonstrasikan
kemampuan menjawab pertanyaan preskriptif. Di lingkungan sungguhan,
konten seperti ini dimiliki dan dikelola oleh ahli domain — bukan oleh
engineer. Sistem sengaja dirancang agar konten itu bisa diperbarui tanpa
menyentuh satu baris kode pun.

---

## Teknologi

**Backend** — Python 3.12, FastAPI, Motor/PyMongo, LangGraph,
LlamaIndex, ChromaDB, FastEmbed, scikit-learn, pandas, Groq

**Frontend** — React 19, Vite, Tailwind CSS 4, Recharts, React Router

**Infrastruktur** — MongoDB Atlas, Docker, systemd, cron