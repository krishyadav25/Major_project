"""Embedding abstractions for dense retrieval.

Provides:
- BaseEmbedder: Abstract interface for embedding models.
- BGEEmbedder: Production embedder using BAAI/bge-m3 via sentence-transformers.
  Lazy-loaded to avoid downloading or instantiating model weights during test runs.
- FakeDeterministicEmbedder: Lightweight, zero-dependency, deterministic embedder
  for fast and reproducible unit testing without network or GPU.
"""

import hashlib
import math
from abc import ABC, abstractmethod
from typing import Any

from app.retrieval.tokenization import tokenize_multilingual


class BaseEmbedder(ABC):
    """Abstract interface for text embedding models."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Embedding dimension size."""
        ...

    @abstractmethod
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Compute embeddings for a list of texts."""
        ...

    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        """Compute embedding for a single search query."""
        ...


class BGEEmbedder(BaseEmbedder):
    """Production BGE-M3 dense embedder using sentence-transformers.

    Uses lazy model loading so merely instantiating this class or importing
    this module does not trigger model weight downloads.
    """

    def __init__(self, model_name: str = "BAAI/bge-m3", device: str | None = None) -> None:
        self.model_name = model_name
        self.device = device
        self._model: Any = None
        self._dimension: int = 1024

    @property
    def dimension(self) -> int:
        return self._dimension

    def _load_model(self) -> Any:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        model = self._load_model()
        embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return [e.tolist() for e in embeddings]

    def embed_query(self, text: str) -> list[float]:
        if not text.strip():
            return [0.0] * self._dimension
        model = self._load_model()
        embedding = model.encode(text, normalize_embeddings=True, show_progress_bar=False)
        return embedding.tolist()


class FakeDeterministicEmbedder(BaseEmbedder):
    """Lightweight, deterministic embedder for unit testing.

    Generates L2-normalized pseudo-embeddings from token hashes without
    downloading any models or loading PyTorch weights.
    """

    def __init__(self, dimension: int = 64) -> None:
        if dimension <= 0:
            raise ValueError(f"dimension must be positive, got {dimension}")
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    def _text_to_vector(self, text: str) -> list[float]:
        vec = [0.0] * self._dimension
        tokens = tokenize_multilingual(text)

        if not tokens:
            # Fall back to character-level hashing if no word tokens found
            tokens = [c for c in text.lower().strip() if not c.isspace()]

        if not tokens:
            return vec

        for token in tokens:
            # Deterministic integer digest from md5
            digest = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16)
            idx = digest % self._dimension
            vec[idx] += 1.0

        # L2-normalize the vector
        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 0.0:
            vec = [v / norm for v in vec]

        return vec

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [self._text_to_vector(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._text_to_vector(text)
