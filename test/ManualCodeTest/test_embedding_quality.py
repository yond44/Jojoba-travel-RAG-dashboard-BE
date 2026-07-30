import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.services.rag.config import EMBEDDING_MODEL  # noqa: E402
from src.services.rag.retrieval import retrieve_advanced  # noqa: E402

RELEVANT_QUESTIONS = [
    "strategi retensi pelanggan risiko tinggi",
    "kampanye mana yang paling murah",
    "kenapa pelanggan churn",
]
IRRELEVANT_QUESTIONS = [
    "resep rendang padang",
    "siapa presiden Amerika",
]

print(f"Model: {EMBEDDING_MODEL}\n")
print("RELEVAN (harus tinggi):")
for question in RELEVANT_QUESTIONS:
    result = retrieve_advanced(question)
    top = result["sources"][0] if result["sources"] else {}
    print(f"  {result['top_score']:.3f}  {question[:40]:42s} -> "
          f"{top.get('id', '-')}")

print("\nTIDAK RELEVAN (harus rendah):")
for question in IRRELEVANT_QUESTIONS:
    result = retrieve_advanced(question)
    print(f"  {result['top_score']:.3f}  {question[:40]}")