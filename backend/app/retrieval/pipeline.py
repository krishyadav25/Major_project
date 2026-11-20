"""Hybrid retrieval pipeline orchestrating dense, sparse, and fusion stages.

Coordinates:
- Indexing into both ChromaDB (dense) and BM25 (sparse).
- Multilingual query execution across both retrieval branches.
- Overfetching candidates to ensure robust fusion recall.
- Reciprocal Rank Fusion (RRF) deduplication and reranking.
- Scheme metadata filtering.
"""

from app.ingestion.models import TextChunk
from app.retrieval.dense import ChromaDenseRetriever
from app.retrieval.fusion import reciprocal_rank_fusion
from app.retrieval.models import RetrievalQuery, SearchResult
from app.retrieval.sparse import BM25Retriever


class HybridRetriever:
    """Orchestrates hybrid retrieval combining dense vector search and BM25 sparse search."""

    def __init__(
        self,
        dense_retriever: ChromaDenseRetriever,
        sparse_retriever: BM25Retriever,
        default_rrf_k: int = 60,
    ) -> None:
        """Initialize the hybrid retriever with dense and sparse backends.

        Args:
            dense_retriever: Initialized ChromaDenseRetriever instance.
            sparse_retriever: Initialized BM25Retriever instance.
            default_rrf_k: Default RRF smoothing constant (default: 60).
        """
        self.dense_retriever = dense_retriever
        self.sparse_retriever = sparse_retriever
        self.default_rrf_k = default_rrf_k

    def index_chunks(self, chunks: list[TextChunk]) -> dict[str, int]:
        """Index a batch of text chunks into both dense and sparse indexes.

        Args:
            chunks: List of TextChunk objects to index.

        Returns:
            Dictionary with counts of indexed items in dense and sparse indexes.
        """
        dense_count = self.dense_retriever.index_chunks(chunks)
        sparse_count = self.sparse_retriever.index_chunks(chunks)
        return {"dense": dense_count, "sparse": sparse_count}

    def delete_document(self, doc_id: str) -> dict[str, int]:
        """Delete all chunks for a document synchronously across dense and sparse indexes.

        Ensures dense and sparse indices stay consistent when documents are removed or re-indexed.

        Args:
            doc_id: Identifier of the document to delete.

        Returns:
            Dictionary with remaining counts in both dense and sparse indexes.
        """
        dense_count = self.dense_retriever.delete_document(doc_id)
        sparse_count = self.sparse_retriever.delete_document(doc_id)
        return {"dense": dense_count, "sparse": sparse_count}

    def retrieve(
        self,
        query: str | RetrievalQuery,
        top_k: int | None = None,
        scheme_name: str | None = None,
        rrf_k: int | None = None,
    ) -> list[SearchResult]:
        """Execute hybrid search using dense retrieval, BM25, and RRF fusion.

        Args:
            query: Either a query string or a typed RetrievalQuery instance.
            top_k: Optional override for number of top results to return.
            scheme_name: Optional override for scheme filter.
            rrf_k: Optional override for RRF constant k.

        Returns:
            List of SearchResult objects ordered by fused RRF score descending.
        """
        if isinstance(query, RetrievalQuery):
            query_str = query.query
            target_top_k = top_k if top_k is not None else query.top_k
            target_scheme = scheme_name if scheme_name is not None else query.scheme_name
            target_rrf_k = rrf_k if rrf_k is not None else query.rrf_k
        else:
            query_str = str(query)
            target_top_k = top_k if top_k is not None else 5
            target_scheme = scheme_name
            target_rrf_k = rrf_k if rrf_k is not None else self.default_rrf_k

        if not query_str.strip() or target_top_k <= 0:
            return []

        # Overfetch candidates from each branch to ensure robust fusion overlap
        candidate_k = max(target_top_k * 3, 20)

        dense_candidates = self.dense_retriever.search(
            query=query_str,
            top_k=candidate_k,
            scheme_name=target_scheme,
        )

        sparse_candidates = self.sparse_retriever.search(
            query=query_str,
            top_k=candidate_k,
            scheme_name=target_scheme,
        )

        return reciprocal_rank_fusion(
            dense_results=dense_candidates,
            sparse_results=sparse_candidates,
            k=target_rrf_k,
            top_k=target_top_k,
        )
