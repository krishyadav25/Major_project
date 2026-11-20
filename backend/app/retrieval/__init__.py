"""Hybrid Retrieval module for Adapted Reasoning RAG.

Provides:
- Multilingual tokenization for English, Hindi, Hinglish, numbers, and currency.
- Embedding abstractions (BaseEmbedder, BGEEmbedder, FakeDeterministicEmbedder).
- BM25 sparse keyword retrieval with safe JSON persistence (BM25Retriever).
- ChromaDB dense vector retrieval with cosine similarity (ChromaDenseRetriever).
- Reciprocal Rank Fusion (reciprocal_rank_fusion).
- End-to-end hybrid retrieval orchestration (HybridRetriever).
- Retrieval data models (SearchResult, RetrievalQuery, FusionConfig).
"""

from app.retrieval.dense import ChromaDenseRetriever
from app.retrieval.embeddings import BaseEmbedder, BGEEmbedder, FakeDeterministicEmbedder
from app.retrieval.fusion import reciprocal_rank_fusion
from app.retrieval.models import FusionConfig, RetrievalQuery, SearchResult
from app.retrieval.pipeline import HybridRetriever
from app.retrieval.sparse import BM25Retriever
from app.retrieval.tokenization import tokenize_multilingual

__all__ = [
    "BGEEmbedder",
    "BM25Retriever",
    "BaseEmbedder",
    "ChromaDenseRetriever",
    "FakeDeterministicEmbedder",
    "FusionConfig",
    "HybridRetriever",
    "RetrievalQuery",
    "SearchResult",
    "reciprocal_rank_fusion",
    "tokenize_multilingual",
]
