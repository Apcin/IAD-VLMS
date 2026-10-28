import re
import os
import json
import urllib.request
import logging
from typing import Optional, Tuple
from difflib import SequenceMatcher

logger = logging.getLogger(__name__)

# URL of the uvicorn judge server (also hosts Nomic embeddings)
EMBEDDING_SERVER_URL = os.environ.get("GEMINI_JUDGE_URL", "http://127.0.0.1:5200")


class NomicEmbeddingModel:
    """Nomic similarity via HTTP call to the uvicorn server."""

    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _lexical_similarity(self, text1: str, text2: str) -> float:
        if not text1 or not text2:
            return 0.0
        if text1 == text2:
            return 1.0
        if text1 in text2 or text2 in text1:
            return 0.85
        return SequenceMatcher(None, text1, text2).ratio()

    def similarity(self, text1: str, text2: str) -> float:
        try:
            endpoint = f"{EMBEDDING_SERVER_URL}/embed_similarity"
            data = json.dumps({"text1": text1, "text2": text2}).encode("utf-8")
            req = urllib.request.Request(endpoint, data=data, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                result = json.loads(resp.read())
                return result.get("similarity", 0.0)
        except Exception as e:
            logger.warning(f"Embedding server call failed, falling back to lexical: {e}")
            return self._lexical_similarity(text1, text2)


class AnomalyRewardCalculator:
    """Type reward using Nomic semantic similarity and fixed threshold bins."""

    def __init__(self):
        self.nomic = NomicEmbeddingModel.get_instance()

    def _normalize_text(self, text: str) -> str:
        if not text:
            return ""
        text = text.lower().strip()
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"[^\w\s-]", "", text)
        return text

    def _threshold_map(self, sim: float) -> float:
        if sim >= 0.90:
            return 1.0
        if sim >= 0.80:
            return 0.9
        if sim >= 0.70:
            return 0.7
        if sim >= 0.55:
            return 0.5
        if sim >= 0.40:
            return 0.2
        return 0.0

    def compute_reward(self, predicted: str, actual: str) -> float:
        """Returns type reward in [0, 1] from Nomic similarity threshold bins."""
        if not predicted or not actual:
            return 0.0

        pred_norm = self._normalize_text(predicted)
        actual_norm = self._normalize_text(actual)
        similarity = self.nomic.similarity(pred_norm, actual_norm)
        return self._threshold_map(similarity)

    def compute_reward_with_explanation(self, predicted: str, actual: str) -> Tuple[float, str]:
        if not predicted or not actual:
            return 0.0, "empty"

        pred_norm = self._normalize_text(predicted)
        actual_norm = self._normalize_text(actual)
        similarity = self.nomic.similarity(pred_norm, actual_norm)
        score = self._threshold_map(similarity)
        return score, f"nomic_similarity={similarity:.4f} -> mapped_score={score:.2f}"