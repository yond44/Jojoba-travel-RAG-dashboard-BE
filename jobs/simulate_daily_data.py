"""
jobs/simulate_daily_data.py

Simulator data harian untuk mode portofolio: menyuntikkan aktivitas
bisnis sintetis SATU HARI penuh dengan hukum relasi yang dipelajari
dari data nyata — bukan angka acak.

Hukum yang ditegakkan (hasil profiling data asli):
  - nights == nights paket; return_date == travel_date + nights
  - created_at == booking_date; kode berurutan BK-/TXN-/CUST-
  - Payment.amount == total; Refund.amount == -total
  - Review hanya untuk booking Completed (~96%), terbit SETELAH
    return_date (median +10 hari); sentimen fungsi dari rating
  - Status x payment mengikuti tabel gabungan empiris
    (Completed+Paid, Cancelled+Unpaid, Cancelled+Refunded, dst.)
  - Konsistensi paket<->hotel<->flight<->destinasi<->level harga dijaga
    lewat TEMPLATE SAMPLING: booking baru mewarisi relasi dari booking
    historis bulan-yang-sama, lalu identitas/tanggal/harga diregenerasi.

Desain lifecycle: seluruh dokumen turunan (payment, refund, review)
dibuat SAAT booking lahir, dengan tanggal masa depannya masing-masing —
konsisten dengan gaya "final state" dataset asli. Query di sistem selalu
memfilter tanggal, jadi dokumen bertanggal-depan otomatis "belum ada"
sampai harinya tiba.

Returning customer (5-10%): TIDAK membuat dokumen customer baru —
hanya UPDATE agregat (total_trips, total_spent_idr, last_purchase_date,
is_repeat_customer). Bobot pemilihan = kedekatan recency terhadap ritme
personal (avg_interval), bukan skor churn — menghindari lingkaran
umpan-balik model yang menilai data buatan dirinya sendiri.

Jalankan:
  python -m jobs.simulate_daily_data              # backfill s.d. kemarin
  python -m jobs.simulate_daily_data --dry-run    # tanpa menulis DB
Cron harian (SEBELUM rescore churn):
  0 3 * * *  cd /path && .venv/bin/python -m jobs.simulate_daily_data
"""

from __future__ import annotations

import argparse
import logging
import math
import random
import string
from collections import defaultdict
from datetime import date, datetime, time, timedelta

import numpy as np
import pandas as pd

from src.config.settings import get_settings
from src.utils.clock import business_today

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("simulate_daily")

VALID_STATUSES = ["Completed", "Confirmed"]
RETURNING_SHARE = (0.05, 0.10)      # porsi booking dari customer lama
SINGLE_CONVERT_SHARE = 0.25         # dari returning: single -> repeat
SOURCE_TAG = "simulator"


# ---------------------------------------------------------------------------
# 1) KONTEKS: pelajari hukum & distribusi dari database
# ---------------------------------------------------------------------------
def load_context(db) -> dict:
    bk = pd.DataFrame(list(db["bookings"].find({}, {
        "booking_code": 1, "customer_id": 1, "package_id": 1,
        "package_name": 1, "destination_name": 1, "package_type": 1,
        "agent_id": 1, "campaign_id": 1, "hotel_id": 1, "flight_id": 1,
        "booking_date": 1, "travel_date": 1, "pax_count": 1, "nights": 1,
        "base_price_idr": 1, "discount_idr": 1, "tax_idr": 1,
        "status": 1, "payment_status": 1, "payment_method": 1,
        "channel": 1, "customer_segment": 1})))
    bk["booking_date"] = pd.to_datetime(bk["booking_date"]).dt.tz_localize(None)
    bk["travel_date"] = pd.to_datetime(bk["travel_date"]).dt.tz_localize(None)

    cs = pd.DataFrame(list(db["customers"].find({}, {
        "name": 1, "email": 1, "city": 1, "segment": 1, "gender": 1,
        "age": 1, "age_group": 1, "acquisition_channel": 1,
        "customer_code": 1, "preferred_destination_type": 1,
        "preferred_package_type": 1})))
    rv = pd.DataFrame(list(db["reviews"].find({}, {
        "rating": 1, "review_text": 1, "review_date": 1, "booking_id": 1})))

    valid = bk[bk["status"].isin(VALID_STATUSES)]

    # Volume harian: rata-rata per (bulan, hari-dalam-minggu) menjaga
    # musiman; uniform min-max akan meratakan Desember dan Februari.
    daily = bk.set_index("booking_date").resample("D").size()
    vol_mean = daily.groupby(
        [daily.index.month, daily.index.dayofweek]).mean().to_dict()

    # Template per bulan: sumber kombinasi relasional yang sah
    templates_by_month = {
        m: g.to_dict("records")
        for m, g in bk.groupby(bk["booking_date"].dt.month)}

    # Ritme personal customer repeat: bahan bobot returning
    g = valid.groupby("customer_id")["booking_date"]
    rhythm = pd.DataFrame({
        "last": g.max(), "first": g.min(), "freq": g.size()})
    rhythm["avg_interval"] = np.where(
        rhythm["freq"] > 1,
        (rhythm["last"] - rhythm["first"]).dt.days
        / (rhythm["freq"] - 1), np.nan)

    # Distribusi empiris lain
    disc_ratio = (bk["discount_idr"] / bk["base_price_idr"]).dropna()
    status_joint = (bk.groupby(["status", "payment_status"]).size()
                    / len(bk)).to_dict()
    rv["review_date"] = pd.to_datetime(rv["review_date"]).dt.tz_localize(None)
    review_text_pool = (rv.groupby("rating")["review_text"]
                        .apply(lambda s: s.dropna().sample(
                            min(300, len(s)), random_state=42).tolist())
                        .to_dict())

    def next_seq(series: pd.Series, prefix: str) -> int:
        nums = (series.dropna().astype(str)
                .str.extract(rf"{prefix}-?0*(\d+)")[0].dropna().astype(int))
        return int(nums.max()) + 1 if len(nums) else 1

    ctx = {
        "vol_mean": vol_mean,
        "vol_clamp": (int(daily.min()), int(daily.max())),
        "templates_by_month": templates_by_month,
        "rhythm": rhythm,
        "customers_meta": cs.set_index("_id"),
        "pax_dist": bk["pax_count"].value_counts(normalize=True),
        "hour_dist": bk["booking_date"].dt.hour.value_counts(normalize=True),
        "lead_days": (bk["travel_date"] - bk["booking_date"]).dt.days
                     .clip(lower=1).dropna(),
        "channel_dist": bk["channel"].value_counts(normalize=True),
        "paymethod_dist": bk["payment_method"].value_counts(normalize=True),
        "status_joint": status_joint,
        "disc_ratio_pool": disc_ratio.sample(
            min(5000, len(disc_ratio)), random_state=42).tolist(),
        "tax_mu_sigma": (0.0873, 0.0105),          # dari profiling
        "campaign_null_p": float(bk["campaign_id"].isna().mean()),
        "agent_null_p": float(bk["agent_id"].isna().mean()),
        "review_rate": 0.963,
        "review_delay": (2, 10, 18),               # p10/p50/p90 hari
        "refund_delay": (10, 36, 92),
        "rating_dist": rv["rating"].value_counts(normalize=True),
        "review_text_pool": review_text_pool,
        "name_pool": cs["name"].dropna().tolist(),
        "email_domains": (cs["email"].dropna().str.split("@").str[1]
                          .value_counts(normalize=True)),
        "city_dist": cs["city"].value_counts(normalize=True),
        "acq_dist": cs["acquisition_channel"].value_counts(normalize=True),
        "seq_booking": next_seq(bk["booking_code"], "BK"),
        "seq_txn": next_seq(pd.Series(
            [d["transaction_code"] for d in db["transactions"].find(
                {}, {"transaction_code": 1})]), "TXN"),
        "seq_customer": next_seq(cs["customer_code"], "CUST"),
        "last_booking_date": daily.index.max().date(),
    }
    logger.info("Konteks siap: %d template, %d nama, seq BK=%d TXN=%d",
                len(bk), len(ctx["name_pool"]),
                ctx["seq_booking"], ctx["seq_txn"])
    return ctx


# ---------------------------------------------------------------------------
# 2) GENERATOR SATU HARI
# ---------------------------------------------------------------------------
def _sample(dist: pd.Series, rng: random.Random):
    return rng.choices(list(dist.index), weights=list(dist.values))[0]


def _tri(lo, mid, hi, rng):  # sampling segitiga dari p10/p50/p90
    return max(0, int(rng.triangular(lo, hi, mid)))


def _timestamp(day: date, ctx, rng) -> datetime:
    hour = _sample(ctx["hour_dist"], rng)
    return datetime.combine(day, time(hour, rng.randrange(60),
                                      rng.randrange(60)))


def _new_customer_doc(ctx, rng, created: datetime, channel: str) -> dict:
    # Nama baru = kombinasi ulang token nama yang ADA di database,
    # supaya menyaru natural ke populasi (permintaan eksplisit).
    a, b = rng.choice(ctx["name_pool"]).split()[:1], \
           rng.choice(ctx["name_pool"]).split()[-1:]
    name = " ".join(a + b)
    domain = _sample(ctx["email_domains"], rng)
    email = (name.lower().replace(" ", ".").replace(",", "")
             + str(rng.randrange(10, 99)) + "@" + domain)
    age = rng.randrange(18, 66)
    seq = ctx["seq_customer"]; ctx["seq_customer"] += 1
    return {
        "customer_code": f"CUST-{seq:06d}",
        "name": name, "email": email,
        "phone": "08" + "".join(rng.choices(string.digits, k=10)),
        "gender": rng.choice(["Male", "Female"]),
        "age": age,
        "age_group": ("18-25" if age <= 25 else "26-35" if age <= 35
                      else "36-45" if age <= 45 else "46-55" if age <= 55
                      else "56+"),
        "city": _sample(ctx["city_dist"], rng),
        "segment": "Bronze",                      # customer baru mulai Bronze
        "acquisition_channel": channel,
        "join_date": created, "created_at": created,
        "total_trips": 0, "total_spent_idr": 0, "avg_trip_value_idr": 0,
        "preferred_destination_type": None, "preferred_package_type": None,
        "is_repeat_customer": False, "last_purchase_date": None,
        "email_opt_in": rng.random() < 0.7,
        "source": SOURCE_TAG,
    }


def _pick_returning(ctx, n: int, today: date, rng: random.Random) -> list:
    """Pilih customer lama berbobot kedekatan ke ritme personalnya.
    Kuota SINGLE_CONVERT_SHARE disisihkan untuk single booker agar
    dinamika konversi single->repeat tetap hidup."""
    r = ctx["rhythm"].copy()
    r["recency"] = (pd.Timestamp(today) - r["last"]).dt.days
    rep = r[r["freq"] > 1].copy()
    # Bobot puncak saat recency == avg_interval (sudah "waktunya")
    z = (rep["recency"] - rep["avg_interval"]) / rep["avg_interval"].clip(lower=30)
    rep["w"] = np.exp(-0.5 * z ** 2) + 1e-6
    single = r[r["freq"] == 1]

    n_single = min(int(n * SINGLE_CONVERT_SHARE), len(single))
    n_repeat = min(n - n_single, len(rep))
    picked = []
    if n_repeat:
        picked += list(np.random.default_rng(rng.randrange(2**32)).choice(
            rep.index, size=n_repeat, replace=False,
            p=(rep["w"] / rep["w"].sum()).values))
    if n_single:
        picked += rng.sample(list(single.index), n_single)
    return picked


def generate_day(ctx, day: date, rng: random.Random) -> dict:
    mean = ctx["vol_mean"].get((day.month, day.weekday()), 60.0)
    lo, hi = ctx["vol_clamp"]
    n = int(np.clip(rng.gauss(mean, 0.15 * mean), lo, hi))

    n_return = int(n * rng.uniform(*RETURNING_SHARE))
    returning_ids = _pick_returning(ctx, n_return, day, rng)

    out = {"bookings": [], "customers": [], "customer_updates": [],
           "transactions": [], "reviews": []}

    for i in range(n):
        created = _timestamp(day, ctx, rng)
        tpl = rng.choice(ctx["templates_by_month"][day.month])
        pax = _sample(ctx["pax_dist"], rng)
        channel = _sample(ctx["channel_dist"], rng)

        # --- identitas customer ---
        if i < len(returning_ids):
            cust_id = returning_ids[i]
            meta = ctx["customers_meta"].loc[cust_id]
            cust_name, cust_segment = meta["name"], meta["segment"]
            is_repeat, new_cust = True, None
        else:
            new_cust = _new_customer_doc(ctx, rng, created, channel)
            cust_id = None                      # diisi saat insert (ObjectId)
            cust_name, cust_segment = new_cust["name"], new_cust["segment"]
            is_repeat = False

        # --- harga: unit per-pax diwarisi template, pax baru, rasio empiris
        unit = tpl["base_price_idr"] / max(tpl["pax_count"], 1)
        base = round(unit * pax * rng.uniform(0.97, 1.03))
        disc = round(base * rng.choice(ctx["disc_ratio_pool"]))
        mu, sg = ctx["tax_mu_sigma"]
        tax = round((base - disc) * max(rng.gauss(mu, sg), 0.05))
        total = base - disc + tax

        status, pay_status = _sample(
            pd.Series(ctx["status_joint"]), rng)
        lead = int(rng.choice(ctx["lead_days"].values))
        travel = datetime.combine(day + timedelta(days=lead),
                                  time(9, 0)) 
        nights = int(tpl["nights"])
        seq = ctx["seq_booking"]; ctx["seq_booking"] += 1

        booking = {
            "booking_code": f"BK-{seq:06d}",
            "customer_id": cust_id,             # placeholder utk customer baru
            "customer_name": cust_name,
            "customer_segment": cust_segment,
            "package_id": tpl["package_id"],
            "package_name": tpl["package_name"],
            "destination_name": tpl["destination_name"],
            "package_type": tpl["package_type"],
            "agent_id": (None if rng.random() < ctx["agent_null_p"]
                         else tpl.get("agent_id")),
            "campaign_id": (None if rng.random() < ctx["campaign_null_p"]
                            else tpl.get("campaign_id")),
            "booking_date": created, "created_at": created,
            "travel_date": travel,
            "return_date": travel + timedelta(days=nights),
            "pax_count": pax, "nights": nights,
            "base_price_idr": base, "discount_idr": disc,
            "tax_idr": tax, "total_price_idr": total,
            "payment_method": _sample(ctx["paymethod_dist"], rng),
            "payment_status": pay_status, "status": status,
            "channel": channel,
            "hotel_id": tpl.get("hotel_id"),
            "flight_id": tpl.get("flight_id"),
            "is_repeat_customer": is_repeat,
            "source": SOURCE_TAG,
        }
        out["bookings"].append(booking)
        if new_cust is not None:
            out["customers"].append((new_cust, booking))  # link utk _id nanti

        # --- lifecycle: transaksi & review (tanggal masa depan sah) ---
        def txn(ttype, amount, when):
            seq_t = ctx["seq_txn"]; ctx["seq_txn"] += 1
            return {"transaction_code": f"TXN-{seq_t:07d}",
                    "booking_ref": booking["booking_code"],  # resolve -> _id
                    "customer_ref": booking,                 # resolve -> _id
                    "amount_idr": amount, "type": ttype,
                    "payment_method": booking["payment_method"],
                    "status": "Success",
                    "gateway_ref": "".join(rng.choices("0123456789ABCDEF", k=16)),
                    "transaction_date": when, "created_at": when,
                    "year": when.year, "month": when.month,
                    "week": int(when.strftime("%V")),
                    "quarter": (when.month - 1) // 3 + 1,
                    "source": SOURCE_TAG}

        if pay_status in ("Paid", "Refunded"):
            pay_when = created + timedelta(days=rng.randrange(0, 3),
                                           hours=rng.randrange(0, 6))
            out["transactions"].append(txn("Payment", total, pay_when))
        if pay_status == "Refunded":
            d10, d50, d90 = ctx["refund_delay"]
            ref_when = created + timedelta(days=_tri(d10, d50, d90, rng))
            out["transactions"].append(txn("Refund", -total, ref_when))

        if status == "Completed" and rng.random() < ctx["review_rate"]:
            rating = int(_sample(ctx["rating_dist"], rng))
            d10, d50, d90 = ctx["review_delay"]
            rv_when = booking["return_date"] + timedelta(
                days=_tri(d10, d50, d90, rng))
            asp = lambda: int(np.clip(rating + rng.choice([-1, 0, 0, 1]), 1, 5))
            out["reviews"].append({
                "booking_ref": booking["booking_code"],
                "customer_ref": booking,
                "customer_name": cust_name,
                "destination_name": booking["destination_name"],
                "package_type": booking["package_type"],
                "rating": rating,
                "sentiment": ("Positive" if rating >= 4 else
                              "Neutral" if rating == 3 else "Negative"),
                "review_text": rng.choice(
                    ctx["review_text_pool"].get(rating, [""])) or "",
                "is_verified": rng.random() < 0.866,
                "review_date": rv_when, "created_at": rv_when,
                "aspects": {"hotel": asp(), "flight": asp(), "guide": asp(),
                            "value_for_money": asp(), "communication": asp()},
                "source": SOURCE_TAG})

        # --- update agregat customer lama (BUKAN dokumen baru) ---
        if is_repeat and status in VALID_STATUSES:
            out["customer_updates"].append((cust_id, total, created))

    return out


# ---------------------------------------------------------------------------
# 3) PENULISAN — resolve referensi ObjectId dengan urutan yang benar
# ---------------------------------------------------------------------------
def write_day(db, docs: dict, day: date):
    from bson import ObjectId

    # 1. Customer baru dulu -> dapat _id -> tempel ke booking-nya
    for cust, booking in docs["customers"]:
        booking["customer_id"] = db["customers"].insert_one(cust).inserted_id

    # 2. Bookings (customer lama: string _id dari rhythm -> ObjectId)
    for b in docs["bookings"]:
        if isinstance(b["customer_id"], str):
            b["customer_id"] = ObjectId(b["customer_id"])
        for f in ("package_id", "agent_id", "campaign_id",
                  "hotel_id", "flight_id"):
            if isinstance(b.get(f), str):
                b[f] = ObjectId(b[f])
        b["_bid"] = db["bookings"].insert_one(
            {k: v for k, v in b.items() if k != "_bid"}).inserted_id

    by_code = {b["booking_code"]: b for b in docs["bookings"]}

    # 3. Transaksi & review: resolve booking_ref/customer_ref -> _id asli
    for t in docs["transactions"]:
        b = by_code[t.pop("booking_ref")]
        t["booking_id"], t["customer_id"] = b["_bid"], b["customer_id"]
        t.pop("customer_ref")
        db["transactions"].insert_one(t)
    for r in docs["reviews"]:
        b = by_code[r.pop("booking_ref")]
        r["booking_id"], r["customer_id"] = b["_bid"], b["customer_id"]
        r.pop("customer_ref")
        db["reviews"].insert_one(r)

    # 4. Update agregat customer returning
    for cust_id, total, when in docs["customer_updates"]:
        db["customers"].update_one(
            {"_id": ObjectId(cust_id) if isinstance(cust_id, str) else cust_id},
            {"$inc": {"total_trips": 1, "total_spent_idr": int(total)},
             "$set": {"last_purchase_date": when,
                      "is_repeat_customer": True}})

    logger.info("%s: %d bookings, %d customer baru, %d update, "
                "%d txn, %d review", day, len(docs["bookings"]),
                len(docs["customers"]), len(docs["customer_updates"]),
                len(docs["transactions"]), len(docs["reviews"]))


# ---------------------------------------------------------------------------
# 4) MAIN — backfill idempoten dari akhir data s.d. kemarin
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="Generate & laporkan tanpa menulis ke database.")
    args = ap.parse_args()

    settings = get_settings()
    from pymongo import MongoClient
    client = MongoClient(settings.mongo_url, serverSelectionTimeoutMS=8000)
    client.admin.command("ping")
    db = client[settings.database_name]

    ctx = load_context(db)
    start = ctx["last_booking_date"] + timedelta(days=1)
    end = business_today() - timedelta(days=1)
    if start > end:
        logger.info("Tidak ada hari yang perlu diisi (%s > %s).", start, end)
        return

    day = start
    while day <= end:
        # Idempoten: hari yang sudah punya booking dilewati
        exists = db["bookings"].count_documents({
            "booking_date": {"$gte": datetime.combine(day, time.min),
                             "$lt": datetime.combine(day + timedelta(days=1),
                                                     time.min)}}, limit=1)
        if exists:
            logger.info("%s sudah terisi — lewati.", day)
        else:
            rng = random.Random(f"jojoba-{day.isoformat()}")  # deterministik
            docs = generate_day(ctx, day, rng)
            if args.dry_run:
                logger.info("[DRY] %s: %d bookings, %d cust baru, %d txn, "
                            "%d review", day, len(docs["bookings"]),
                            len(docs["customers"]),
                            len(docs["transactions"]), len(docs["reviews"]))
            else:
                write_day(db, docs, day)
        day += timedelta(days=1)


if __name__ == "__main__":
    main()
