import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.services.rag.vector_store import get_collection, collection_count

st.set_page_config(page_title="Chroma Viewer", layout="wide")
st.title("🔍 ChromaDB Viewer")

st.metric("Total vektor ter-index", collection_count())

collection = get_collection()
# include embeddings agar kamu lihat dimensinya; buang bila terlalu berat
data = collection.get(include=["documents", "metadatas", "embeddings"])

rows = []
for chunk_id, document, metadata, embedding in zip(
        data["ids"], data["documents"], data["metadatas"],
        data["embeddings"]):
    rows.append({
        "id": chunk_id[:12] + "...",
        "source_type": metadata.get("source_type", ""),
        "topic": metadata.get("topic", ""),
        "logical_id": metadata.get("id", ""),
        "dim": len(embedding),
        "text_preview": document[:120],
        "embedding_head": str([round(value, 3) for value in embedding[:5]]),
    })

frame = pd.DataFrame(rows)

# Filter cepat per source_type
source_filter = st.multiselect(
    "Filter source_type", options=sorted(frame["source_type"].unique()),
    default=list(frame["source_type"].unique()))
filtered = frame[frame["source_type"].isin(source_filter)]

st.caption(f"Menampilkan {len(filtered)} dari {len(frame)} chunk")
st.dataframe(filtered, use_container_width=True, height=500)

# Lihat satu chunk utuh dalam bentuk JSON
st.divider()
selected_index = st.number_input("Lihat detail chunk (baris ke-)",
                                 min_value=0, max_value=len(data["ids"]) - 1,
                                 value=0)
st.json({
    "id": data["ids"][selected_index],
    "metadata": data["metadatas"][selected_index],
    "text": data["documents"][selected_index],
    "embedding_dim": len(data["embeddings"][selected_index]),
    "embedding_first_10": [round(v, 4)
                           for v in data["embeddings"][selected_index][:10]],
})