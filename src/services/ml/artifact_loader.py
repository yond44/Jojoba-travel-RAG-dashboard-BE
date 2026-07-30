from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib

from src.config.settings import get_settings
from src.utils.log import logger


@dataclass(frozen=True)
class MLArtifacts:
    churn_model: Any
    churn_features: list[str]
    kmeans: dict[str, Any]
    churn_keras: Any | None = None
    keras_preprocessor: dict[str, Any] | None = None


_artifacts: MLArtifacts | None = None


def _require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(
            f"Artifact not found: {path}. "
            f"Run the initial training first (see RUNBOOK_ML.md), "
            f"or check ARTIFACTS_DIR in .env."
        )
    return path


def load_artifacts() -> MLArtifacts:
    global _artifacts
    if _artifacts is not None:
        return _artifacts

    base = Path(get_settings().artifacts_dir) / "models"
    logger.info("Initializing ML artifacts from %s ...", base)

    churn_model = joblib.load(_require(base / "churn_model_sklearn.pkl"))
    with open(_require(base / "churn_features.json")) as f:
        churn_features: list[str] = json.load(f)
    kmeans = joblib.load(_require(base / "kmeans_segmentation.pkl"))

 
    churn_keras = None
    keras_preprocessor = None
    keras_path = base / "churn_model.keras"
    if keras_path.exists():
        try:
            from tensorflow import keras
            churn_keras = keras.models.load_model(keras_path)
            keras_preprocessor = joblib.load(
                _require(base / "churn_keras_preprocessor.pkl"))
            logger.info("Keras comparator model loaded.")
        except Exception:
            logger.warning("Found .keras but failed to load it — "
                           "continuing with the sklearn model only.")

    _artifacts = MLArtifacts(
        churn_model=churn_model,
        churn_features=churn_features,
        kmeans=kmeans,
        churn_keras=churn_keras,
        keras_preprocessor=keras_preprocessor,
    )
    logger.info("ML artifacts ready: %d churn features, kmeans.",
                len(churn_features))
    return _artifacts


def get_artifacts() -> MLArtifacts:
    if _artifacts is None:
        raise RuntimeError(
            "Artifacts have not been loaded yet. Make sure load_artifacts() "
            "is called during lifespan startup (main.py).")
    return _artifacts
