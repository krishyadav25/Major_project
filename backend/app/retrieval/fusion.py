"""Reciprocal Rank Fusion (RRF) for combining dense and sparse retrieval results.

Implements the standard RRF algorithm:
    RRF_score(d) = sum_{m in M} (1 / (k + rank_m(d)))

Features:
- Configurable constant k (standard default: 60).
- 1-based ranks.
- Merges provenance, ranks, and raw scores from both retrieval modalities.
- Deterministic tie-breaking on identical RRF scores via chunk_id.
"""

from app.retrieval.models import SearchResult


def reciprocal_rank_fusion(
    dense_results: list[SearchResult],
    sparse_results: list[SearchResult],
    k: int = 60,
    top_k: int = 5,
) -> list[SearchResult]:
    """Fuse ranked candidate lists from dense and sparse retrievers using RRF.

    Args:
        dense_results: Ranked list of SearchResult items from dense retrieval.
        sparse_results: Ranked list of SearchResult items from sparse retrieval.
        k: Smoothing constant parameterizing the rank dampening (default: 60).
        top_k: Maximum number of fused results to return (default: 5).

    Returns:
        Deduplicated and reranked list of SearchResult items, ordered by rrf_score descending.
    """
    if top_k <= 0:
        return []
    if k <= 0:
        raise ValueError(f"RRF smoothing constant k must be positive, got {k}")

    # Map of chunk_id -> merged SearchResult candidate
    merged: dict[str, SearchResult] = {}
    rrf_scores: dict[str, float] = {}

    # Process dense results
    for rank, res in enumerate(dense_results, start=1):
        chunk_id = res.chunk_id
        rrf_contribution = 1.0 / (k + rank)
        rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + rrf_contribution

        # Clone or record candidate
        merged[chunk_id] = SearchResult(
            chunk_id=res.chunk_id,
            doc_id=res.doc_id,
            filename=res.filename,
            page_number=res.page_number,
            chunk_index=res.chunk_index,
            text=res.text,
            scheme_name=res.scheme_name,
            dense_score=res.dense_score if res.dense_score is not None else res.score,
            dense_rank=rank,
            sparse_score=None,
            sparse_rank=None,
        )

    # Process sparse results
    for rank, res in enumerate(sparse_results, start=1):
        chunk_id = res.chunk_id
        rrf_contribution = 1.0 / (k + rank)
        rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + rrf_contribution

        if chunk_id in merged:
            existing = merged[chunk_id]
            existing.sparse_score = res.sparse_score if res.sparse_score is not None else res.score
            existing.sparse_rank = rank
            if existing.scheme_name is None and res.scheme_name is not None:
                existing.scheme_name = res.scheme_name
        else:
            merged[chunk_id] = SearchResult(
                chunk_id=res.chunk_id,
                doc_id=res.doc_id,
                filename=res.filename,
                page_number=res.page_number,
                chunk_index=res.chunk_index,
                text=res.text,
                scheme_name=res.scheme_name,
                dense_score=None,
                dense_rank=None,
                sparse_score=res.sparse_score if res.sparse_score is not None else res.score,
                sparse_rank=rank,
            )

    # Set final rrf_score and score on all candidates
    candidate_list: list[SearchResult] = []
    for chunk_id, candidate in merged.items():
        total_score = rrf_scores[chunk_id]
        candidate.rrf_score = total_score
        candidate.score = total_score
        candidate_list.append(candidate)

    # Sort descending by rrf_score; break ties deterministically by chunk_id
    candidate_list.sort(key=lambda item: (-item.score, item.chunk_id))

    return candidate_list[:top_k]
