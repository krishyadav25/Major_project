"""Unit tests for Milestone 3: Hybrid Retrieval.

Tests:
1. Deterministic multilingual tokenization (English, Hindi Devanagari, Hinglish, currency, numbers).
2. Embedder abstractions (FakeDeterministicEmbedder, lazy BGEEmbedder).
3. BM25Retriever indexing, search, metadata filtering, and safe JSON persistence.
4. ChromaDenseRetriever in-memory/persistent indexing, cosine similarity search, and scheme filtering.
5. Reciprocal Rank Fusion (RRF) mathematical accuracy, tie-breaking, and ranking.
6. HybridRetriever end-to-end indexing, retrieval, and parameter handling.
7. Source provenance citations and document collision resistance.
"""

import math
from pathlib import Path

import pytest
from app.ingestion.models import (
    ChunkMetadata,
    TextChunk,
    compute_content_hash,
    make_doc_id,
)
from app.retrieval.dense import ChromaDenseRetriever
from app.retrieval.embeddings import BGEEmbedder, FakeDeterministicEmbedder
from app.retrieval.fusion import reciprocal_rank_fusion
from app.retrieval.models import RetrievalQuery, SearchResult
from app.retrieval.pipeline import HybridRetriever
from app.retrieval.sparse import BM25Retriever
from app.retrieval.tokenization import tokenize_multilingual

# ==============================================================================
# 1. Multilingual Tokenizer Tests
# ==============================================================================


def test_tokenize_english_basic():
    text = "The PM Kisan scheme provides direct financial support to farmers."
    tokens = tokenize_multilingual(text)
    assert "pm" in tokens
    assert "kisan" in tokens
    assert "scheme" in tokens
    assert "farmers" in tokens
    # All Latin tokens lowercased
    for t in tokens:
        assert t == t.lower()


def test_tokenize_hindi_devanagari():
    text = "प्रधानमंत्री किसान सम्मान निधि योजना के तहत किसानों को वित्तीय सहायता मिलती है।"
    tokens = tokenize_multilingual(text)
    assert "प्रधानमंत्री" in tokens
    assert "किसान" in tokens
    assert "सम्मान" in tokens
    assert "निधि" in tokens
    assert "किसानों" in tokens


def test_tokenize_hinglish():
    text = "PM Kisan yojana me apply kaise kare eligibility kya hai?"
    tokens = tokenize_multilingual(text)
    assert "pm" in tokens
    assert "kisan" in tokens
    assert "yojana" in tokens
    assert "apply" in tokens
    assert "eligibility" in tokens


def test_tokenize_currency_normalization():
    # Rs. 6,000, rs 6000, INR 6000, and ₹6,000 should all produce ₹ and 6000
    texts = [
        "Annual benefit is Rs. 6,000 per year.",
        "Annual benefit is Rs 6,000 per year.",
        "Annual benefit is INR 6000 per year.",
        "Annual benefit is ₹6,000 per year.",
    ]
    for text in texts:
        tokens = tokenize_multilingual(text)
        assert "₹" in tokens, f"Failed currency symbol in: {text}"
        assert "6000" in tokens, f"Failed normalized amount in: {text}"


def test_tokenize_numerical_thresholds():
    text = "Cultivable land up to 2.0 hectares and age between 18-40 years with 100% subsidy."
    tokens = tokenize_multilingual(text)
    assert "2.0" in tokens
    assert "18" in tokens
    assert "40" in tokens
    assert "100" in tokens


def test_tokenize_empty_and_whitespace():
    assert tokenize_multilingual("") == []
    assert tokenize_multilingual("   \n\t  ") == []
    assert tokenize_multilingual("!@#$%^&*()_+{}[]:;\"'<>?,./~`") == []


# ==============================================================================
# 2. Embedder Tests
# ==============================================================================


def test_fake_embedder_dimension_and_validation():
    embedder = FakeDeterministicEmbedder(dimension=32)
    assert embedder.dimension == 32

    with pytest.raises(ValueError, match="dimension must be positive"):
        FakeDeterministicEmbedder(dimension=0)


def test_fake_embedder_deterministic_and_normalized():
    embedder = FakeDeterministicEmbedder(dimension=64)
    vec1 = embedder.embed_query("PM Kisan eligibility criteria 2000")
    vec2 = embedder.embed_query("PM Kisan eligibility criteria 2000")
    assert vec1 == vec2
    assert len(vec1) == 64

    # Verify L2 normalization: sum(v^2) == 1.0
    norm = math.sqrt(sum(v * v for v in vec1))
    assert pytest.approx(norm, abs=1e-5) == 1.0


def test_fake_embedder_empty_text():
    embedder = FakeDeterministicEmbedder(dimension=64)
    vec = embedder.embed_query("")
    assert len(vec) == 64
    assert all(v == 0.0 for v in vec)


def test_fake_embedder_batch_matches_single():
    embedder = FakeDeterministicEmbedder(dimension=64)
    texts = ["First scheme document", "Second scheme document"]
    batch = embedder.embed_texts(texts)
    single0 = embedder.embed_query(texts[0])
    single1 = embedder.embed_query(texts[1])
    assert batch[0] == single0
    assert batch[1] == single1
    assert embedder.embed_texts([]) == []


def test_bge_embedder_lazy_loading():
    embedder = BGEEmbedder(model_name="BAAI/bge-m3")
    assert embedder.dimension == 1024
    # The internal model is NOT instantiated yet (no network or weight loading)
    assert embedder._model is None


def test_bge_embedder_mocked_inference(monkeypatch):
    import numpy as np

    class MockModel:
        def __init__(self, model_name, device=None):
            self.model_name = model_name
            self.device = device

        def encode(self, texts, normalize_embeddings=True, show_progress_bar=False):
            if isinstance(texts, str):
                return np.ones((1024,), dtype=float)
            return np.ones((len(texts), 1024), dtype=float)

    import sentence_transformers

    monkeypatch.setattr(sentence_transformers, "SentenceTransformer", MockModel)

    embedder = BGEEmbedder(model_name="BAAI/bge-m3")
    assert embedder.embed_texts([]) == []

    batch = embedder.embed_texts(["test text"])
    assert len(batch) == 1
    assert len(batch[0]) == 1024

    assert embedder.embed_query("") == [0.0] * 1024
    query_vec = embedder.embed_query("test query")
    assert len(query_vec) == 1024


# ==============================================================================
# 3. BM25 Sparse Retriever Tests
# ==============================================================================


@pytest.fixture
def sample_chunks() -> list[TextChunk]:
    return [
        TextChunk(
            text="PM-Kisan provides ₹6,000 per year in three installments to farmer families.",
            metadata=ChunkMetadata(
                chunk_id="pm_kisan::chunk::0001",
                doc_id="pm_kisan",
                filename="pm_kisan_guidelines.pdf",
                page_number=1,
                chunk_index=0,
                char_count=77,
                scheme_name="PM-Kisan",
            ),
        ),
        TextChunk(
            text="Institutional landholders and income tax payees are excluded from PM-Kisan benefits.",
            metadata=ChunkMetadata(
                chunk_id="pm_kisan::chunk::0002",
                doc_id="pm_kisan",
                filename="pm_kisan_guidelines.pdf",
                page_number=2,
                chunk_index=1,
                char_count=85,
                scheme_name="PM-Kisan",
            ),
        ),
        TextChunk(
            text="Ayushman Bharat provides health insurance coverage of ₹5,00,000 per family per year.",
            metadata=ChunkMetadata(
                chunk_id="ayushman::chunk::0001",
                doc_id="ayushman",
                filename="pmjay_guidelines.pdf",
                page_number=1,
                chunk_index=0,
                char_count=85,
                scheme_name="Ayushman Bharat",
            ),
        ),
    ]


def test_bm25_indexing_and_search(sample_chunks):
    retriever = BM25Retriever()
    count = retriever.index_chunks(sample_chunks)
    assert count == 3

    results = retriever.search("PM-Kisan ₹6,000 installment", top_k=2)
    assert len(results) > 0
    top = results[0]
    assert top.chunk_id == "pm_kisan::chunk::0001"
    assert top.sparse_rank == 1
    assert top.sparse_score is not None
    assert top.sparse_score > 0.0
    assert top.scheme_name == "PM-Kisan"


def test_bm25_scheme_pre_filtering(sample_chunks):
    retriever = BM25Retriever()
    retriever.index_chunks(sample_chunks)

    # Search for health insurance with PM-Kisan filter -> must return empty
    res_kisan = retriever.search("health insurance", scheme_name="PM-Kisan")
    assert len(res_kisan) == 0

    # Search with Ayushman Bharat filter -> returns chunk
    res_ayushman = retriever.search("health insurance", scheme_name="Ayushman Bharat")
    assert len(res_ayushman) == 1
    assert res_ayushman[0].chunk_id == "ayushman::chunk::0001"


def test_bm25_empty_query_or_index():
    retriever = BM25Retriever()
    assert retriever.search("anything") == []
    assert retriever.index_chunks([]) == 0

    retriever.index_chunks(
        [
            TextChunk(
                text="sample text",
                metadata=ChunkMetadata(
                    chunk_id="doc::chunk::0001",
                    doc_id="doc",
                    filename="doc.pdf",
                    page_number=1,
                    chunk_index=0,
                    char_count=11,
                ),
            )
        ]
    )
    assert retriever.search("") == []
    assert retriever.search("   ") == []
    assert retriever.search("!@#$%^&*()") == []


def test_bm25_json_persistence(sample_chunks, tmp_path: Path):
    filepath = tmp_path / "bm25_index.json"
    retriever = BM25Retriever()
    retriever.index_chunks(sample_chunks)
    retriever.save_to_json(filepath)

    assert filepath.exists()

    # Load into a new retriever
    loaded_retriever = BM25Retriever.from_json(filepath)
    assert len(loaded_retriever.chunks) == 3

    results = loaded_retriever.search("Ayushman 500000", top_k=1)
    assert len(results) == 1
    assert results[0].chunk_id == "ayushman::chunk::0001"


def test_bm25_load_from_json_missing_file(tmp_path: Path):
    retriever = BM25Retriever()
    with pytest.raises(FileNotFoundError):
        retriever.load_from_json(tmp_path / "non_existent.json")


def test_bm25_delete_document(sample_chunks):
    retriever = BM25Retriever()
    retriever.index_chunks(sample_chunks)
    assert len(retriever.chunks) == 3

    # Delete pm_kisan (2 chunks)
    remaining = retriever.delete_document("pm_kisan")
    assert remaining == 1
    assert len(retriever.chunks) == 1
    assert retriever.chunks[0].metadata.doc_id == "ayushman"

    # Search for deleted chunk returns []
    assert retriever.search("PM-Kisan ₹6,000") == []

    # Deleting non-existent doc returns current count
    assert retriever.delete_document("non_existent") == 1


# ==============================================================================
# 4. ChromaDenseRetriever Tests
# ==============================================================================


def test_chroma_dense_indexing_and_search(sample_chunks):
    embedder = FakeDeterministicEmbedder(dimension=64)
    retriever = ChromaDenseRetriever(
        embedder=embedder,
        persist_directory=None,
        collection_name="dense_search_test",
    )

    count = retriever.index_chunks(sample_chunks)
    assert count == 3
    assert retriever.count() == 3

    results = retriever.search("PM-Kisan ₹6,000", top_k=2)
    assert len(results) == 2
    assert results[0].dense_rank == 1
    assert results[0].dense_score is not None
    assert -1.0 <= results[0].dense_score <= 1.0
    assert results[0].source_citation.startswith("[PM-Kisan]")


def test_chroma_dense_scheme_filter(sample_chunks):
    embedder = FakeDeterministicEmbedder(dimension=64)
    retriever = ChromaDenseRetriever(
        embedder=embedder,
        persist_directory=None,
        collection_name="dense_filter_test",
    )
    retriever.index_chunks(sample_chunks)

    # Filter to Ayushman Bharat only
    results = retriever.search("farmer assistance", scheme_name="Ayushman Bharat", top_k=5)
    for r in results:
        assert r.scheme_name == "Ayushman Bharat"


def test_chroma_dense_persistent_client(sample_chunks, tmp_path: Path):
    persist_dir = tmp_path / "chroma_test_db"
    embedder = FakeDeterministicEmbedder(dimension=64)

    # 1. Create and populate persistent retriever
    r1 = ChromaDenseRetriever(embedder=embedder, persist_directory=persist_dir)
    r1.index_chunks(sample_chunks)
    assert r1.count() == 3

    # 2. Open new retriever pointing to same persistent directory
    r2 = ChromaDenseRetriever(embedder=embedder, persist_directory=persist_dir)
    assert r2.count() == 3
    res = r2.search("PM-Kisan", top_k=1)
    assert len(res) == 1


def test_chroma_dense_empty_query_and_empty_collection():
    embedder = FakeDeterministicEmbedder(dimension=64)
    retriever = ChromaDenseRetriever(
        embedder=embedder,
        persist_directory=None,
        collection_name="isolated_empty_collection",
    )
    assert retriever.count() == 0
    assert retriever.search("test") == []
    assert retriever.index_chunks([]) == 0

    retriever.index_chunks(
        [
            TextChunk(
                text="some document",
                metadata=ChunkMetadata(
                    chunk_id="doc::chunk::0001",
                    doc_id="doc",
                    filename="f.pdf",
                    page_number=1,
                    chunk_index=0,
                    char_count=13,
                ),
            )
        ]
    )
    assert retriever.search("") == []
    assert retriever.search("   ", top_k=0) == []


def test_chroma_dense_query_exception_handled(sample_chunks, monkeypatch):
    embedder = FakeDeterministicEmbedder(dimension=64)
    retriever = ChromaDenseRetriever(
        embedder=embedder,
        persist_directory=None,
        collection_name="exception_test_collection",
    )
    retriever.index_chunks(sample_chunks)

    def mock_query(*args, **kwargs):
        raise RuntimeError("Simulated query failure")

    monkeypatch.setattr(retriever._collection, "query", mock_query)
    assert retriever.search("test") == []


def test_chroma_dense_delete_document(sample_chunks):
    embedder = FakeDeterministicEmbedder(dimension=64)
    retriever = ChromaDenseRetriever(
        embedder=embedder,
        persist_directory=None,
        collection_name="delete_test_collection",
    )
    retriever.index_chunks(sample_chunks)
    assert retriever.count() == 3

    # Delete pm_kisan (2 chunks)
    remaining = retriever.delete_document("pm_kisan")
    assert remaining == 1
    assert retriever.count() == 1

    # Deleting non-existent doc returns current count
    assert retriever.delete_document("non_existent") == 1
    assert retriever.delete_document("") == 1


# ==============================================================================
# 5. Reciprocal Rank Fusion (RRF) Tests
# ==============================================================================


def test_rrf_mathematical_accuracy():
    # Construct synthetic results for Doc A and Doc B
    # Doc A: dense rank 1, sparse rank 2
    # Doc B: dense rank 2, sparse rank 1
    # Constant k = 60
    # Expected RRF(A) = 1/(60+1) + 1/(60+2) = 1/61 + 1/62 = 0.01639344 + 0.01612903 = 0.03252247
    # Expected RRF(B) = 1/(60+2) + 1/(60+1) = 1/62 + 1/61 = 0.03252247
    # Ties broken alphabetically: "chunk_a" before "chunk_b"

    dense_res = [
        SearchResult(
            chunk_id="chunk_a",
            doc_id="doc",
            filename="a.pdf",
            page_number=1,
            chunk_index=0,
            text="A",
            score=0.9,
            dense_score=0.9,
            dense_rank=1,
        ),
        SearchResult(
            chunk_id="chunk_b",
            doc_id="doc",
            filename="b.pdf",
            page_number=1,
            chunk_index=1,
            text="B",
            score=0.8,
            dense_score=0.8,
            dense_rank=2,
        ),
    ]

    sparse_res = [
        SearchResult(
            chunk_id="chunk_b",
            doc_id="doc",
            filename="b.pdf",
            page_number=1,
            chunk_index=1,
            text="B",
            score=10.0,
            sparse_score=10.0,
            sparse_rank=1,
        ),
        SearchResult(
            chunk_id="chunk_a",
            doc_id="doc",
            filename="a.pdf",
            page_number=1,
            chunk_index=0,
            text="A",
            score=8.0,
            sparse_score=8.0,
            sparse_rank=2,
        ),
    ]

    fused = reciprocal_rank_fusion(dense_res, sparse_res, k=60, top_k=5)
    assert len(fused) == 2

    expected_score = (1.0 / 61) + (1.0 / 62)
    assert pytest.approx(fused[0].rrf_score, abs=1e-7) == expected_score
    assert pytest.approx(fused[1].rrf_score, abs=1e-7) == expected_score

    # Deterministic tie-breaking on identical RRF score by chunk_id
    assert fused[0].chunk_id == "chunk_a"
    assert fused[1].chunk_id == "chunk_b"

    # Verify both ranks and scores are populated
    assert fused[0].dense_rank == 1
    assert fused[0].sparse_rank == 2
    assert fused[1].dense_rank == 2
    assert fused[1].sparse_rank == 1


def test_rrf_disjoint_candidates():
    # Doc A only in dense (rank 1) -> 1 / 61
    # Doc C only in sparse (rank 1) -> 1 / 61
    dense_res = [
        SearchResult(
            chunk_id="chunk_a",
            doc_id="doc",
            filename="a.pdf",
            page_number=1,
            chunk_index=0,
            text="A",
            score=0.9,
            dense_rank=1,
        )
    ]
    sparse_res = [
        SearchResult(
            chunk_id="chunk_c",
            doc_id="doc",
            filename="c.pdf",
            page_number=1,
            chunk_index=0,
            text="C",
            score=12.0,
            sparse_rank=1,
        )
    ]

    fused = reciprocal_rank_fusion(dense_res, sparse_res, k=60, top_k=1)
    assert len(fused) == 1
    assert fused[0].chunk_id == "chunk_a"  # 'chunk_a' < 'chunk_c' tie-break
    assert fused[0].dense_rank == 1
    assert fused[0].sparse_rank is None


def test_rrf_scheme_name_backfill():
    # Dense result has scheme_name=None, sparse result has scheme_name="PM-Kisan"
    dense_res = [
        SearchResult(
            chunk_id="chunk_a",
            doc_id="doc",
            filename="a.pdf",
            page_number=1,
            chunk_index=0,
            text="A",
            score=0.9,
            scheme_name=None,
            dense_rank=1,
        )
    ]
    sparse_res = [
        SearchResult(
            chunk_id="chunk_a",
            doc_id="doc",
            filename="a.pdf",
            page_number=1,
            chunk_index=0,
            text="A",
            score=5.0,
            scheme_name="PM-Kisan",
            sparse_rank=1,
        )
    ]
    fused = reciprocal_rank_fusion(dense_res, sparse_res, k=60, top_k=1)
    assert len(fused) == 1
    assert fused[0].scheme_name == "PM-Kisan"


def test_rrf_invalid_k_and_top_k():
    with pytest.raises(ValueError, match="k must be positive"):
        reciprocal_rank_fusion([], [], k=0)

    assert reciprocal_rank_fusion([], [], k=60, top_k=0) == []


# ==============================================================================
# 6. HybridRetriever Pipeline Tests
# ==============================================================================


def test_hybrid_retriever_pipeline(sample_chunks):
    embedder = FakeDeterministicEmbedder(dimension=64)
    dense_retriever = ChromaDenseRetriever(
        embedder=embedder,
        persist_directory=None,
        collection_name="hybrid_test_collection",
    )
    sparse_retriever = BM25Retriever()

    hybrid = HybridRetriever(
        dense_retriever=dense_retriever,
        sparse_retriever=sparse_retriever,
        default_rrf_k=60,
    )

    counts = hybrid.index_chunks(sample_chunks)
    assert counts == {"dense": 3, "sparse": 3}

    # Query with plain string
    results = hybrid.retrieve("PM-Kisan ₹6,000", top_k=2)
    assert len(results) <= 2
    assert len(results) > 0
    assert results[0].rrf_score is not None

    # Query with typed RetrievalQuery
    rq = RetrievalQuery(
        query="PM-Kisan benefits",
        scheme_name="PM-Kisan",
        top_k=2,
        rrf_k=60,
    )
    rq_results = hybrid.retrieve(rq)
    assert len(rq_results) > 0
    for r in rq_results:
        assert r.scheme_name == "PM-Kisan"

    # Empty query returns []
    assert hybrid.retrieve("") == []
    assert hybrid.retrieve("   ", top_k=0) == []


def test_hybrid_retriever_delete_document(sample_chunks):
    embedder = FakeDeterministicEmbedder(dimension=64)
    dense_retriever = ChromaDenseRetriever(
        embedder=embedder,
        persist_directory=None,
        collection_name="hybrid_delete_test_collection",
    )
    sparse_retriever = BM25Retriever()

    hybrid = HybridRetriever(
        dense_retriever=dense_retriever,
        sparse_retriever=sparse_retriever,
    )
    hybrid.index_chunks(sample_chunks)

    # Delete pm_kisan (2 chunks)
    counts = hybrid.delete_document("pm_kisan")
    assert counts == {"dense": 1, "sparse": 1}


# ==============================================================================
# 7. Source Provenance and Collision Resistance Tests
# ==============================================================================


def test_source_citation_format():
    res_with_scheme = SearchResult(
        chunk_id="kisan::chunk::0001",
        doc_id="kisan",
        filename="pm_kisan.pdf",
        page_number=3,
        chunk_index=2,
        text="Sample text",
        scheme_name="PM-Kisan",
    )
    assert res_with_scheme.source_citation == "[PM-Kisan] pm_kisan.pdf (Page 3)"

    res_without_scheme = SearchResult(
        chunk_id="guideline::chunk::0001",
        doc_id="guideline",
        filename="generic.pdf",
        page_number=1,
        chunk_index=0,
        text="Sample text",
        scheme_name=None,
    )
    assert res_without_scheme.source_citation == "generic.pdf (Page 1)"


def test_document_id_collision_resistance(tmp_path: Path):
    # Two PDF files with identical filenames but in different directories with different contents
    dir1 = tmp_path / "dept1"
    dir2 = tmp_path / "dept2"
    dir1.mkdir()
    dir2.mkdir()

    file1 = dir1 / "guidelines.pdf"
    file2 = dir2 / "guidelines.pdf"

    content1 = b"Department 1 guidelines for agriculture welfare."
    content2 = b"Department 2 guidelines for healthcare insurance."

    file1.write_bytes(content1)
    file2.write_bytes(content2)

    hash1 = compute_content_hash(content1)
    hash2 = compute_content_hash(content2)

    doc_id_1 = make_doc_id(file1, content_hash=hash1)
    doc_id_2 = make_doc_id(file2, content_hash=hash2)

    # Must be distinct because hashes differ
    assert doc_id_1 != doc_id_2
    assert hash1[:8] in doc_id_1
    assert hash2[:8] in doc_id_2

    # Chunks generated from both documents will have non-colliding chunk_ids
    chunk_id_1 = f"{doc_id_1}::chunk::0000"
    chunk_id_2 = f"{doc_id_2}::chunk::0000"
    assert chunk_id_1 != chunk_id_2
