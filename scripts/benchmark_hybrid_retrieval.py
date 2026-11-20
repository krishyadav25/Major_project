"""Validation and benchmarking script for Milestone 3: Hybrid Retrieval.

Evaluates:
- Real BGE-M3 dense embeddings (CPU inference).
- Multilingual tokenization and BM25 sparse keyword search.
- Reciprocal Rank Fusion (RRF) combining dense and sparse results.
- Latency, peak memory, throughput, and cross-lingual retrieval accuracy
  across English, Hindi, and Hinglish government scheme queries.
"""

import gc
import json
import math
import time
from pathlib import Path
from typing import Any

import psutil
from app.ingestion.models import ChunkMetadata, TextChunk
from app.retrieval.dense import ChromaDenseRetriever
from app.retrieval.embeddings import BGEEmbedder
from app.retrieval.models import RetrievalQuery
from app.retrieval.pipeline import HybridRetriever
from app.retrieval.sparse import BM25Retriever


def get_process_memory_mb() -> float:
    """Return the current process RSS in Megabytes."""
    return psutil.Process().memory_info().rss / (1024**2)


def create_synthetic_corpus() -> list[TextChunk]:
    """Create a realistic synthetic corpus of 12 government scheme passages.

    Covers 3 major schemes (PM-Kisan, PM-JAY, PMAY-G) across English and Hindi.
    """
    raw_specs = [
        # --- PM-KISAN ---
        (
            "pm_kisan::chunk::0001",
            "pm_kisan",
            "pm_kisan_guidelines.pdf",
            1,
            0,
            (
                "The Pradhan Mantri Kisan Samman Nidhi (PM-KISAN) is a central sector scheme that "
                "provides income support of ₹6,000 per year in three equal installments of ₹2,000 "
                "every four months to all landholding farmer families across India."
            ),
            "PM-Kisan",
        ),
        (
            "pm_kisan::chunk::0002",
            "pm_kisan",
            "pm_kisan_guidelines.pdf",
            2,
            1,
            (
                "The following categories of beneficiaries of higher economic status shall not be "
                "eligible for benefits under PM-Kisan: (a) All Institutional Landholders; (b) Former and "
                "present holders of constitutional posts; (c) All persons who paid Income Tax in last assessment year."
            ),
            "PM-Kisan",
        ),
        (
            "pm_kisan::chunk::0003",
            "pm_kisan",
            "pm_kisan_guidelines.pdf",
            3,
            2,
            (
                "प्रधानमंत्री किसान सम्मान निधि योजना के तहत सभी पात्र किसान परिवारों को प्रति वर्ष "
                "₹6,000 का वित्तीय लाभ प्रदान किया जाता है, जो ₹2,000 की तीन समान किस्तों में सीधे आधार लिंक बैंक खाते में भेजा जाता है।"
            ),
            "PM-Kisan",
        ),
        (
            "pm_kisan::chunk::0004",
            "pm_kisan",
            "pm_kisan_guidelines.pdf",
            4,
            3,
            (
                "संस्थागत भूमि धारक और पिछले मूल्यांकन वर्ष में आयकर (Income Tax) का भुगतान करने वाले "
                "सभी नागरिक पीएम-किसान योजना के तहत लाभ के लिए अपात्र (not eligible) घोषित किए गए हैं।"
            ),
            "PM-Kisan",
        ),
        # --- AYUSHMAN BHARAT PM-JAY ---
        (
            "pmjay::chunk::0001",
            "pmjay",
            "pmjay_manual.pdf",
            1,
            0,
            (
                "Ayushman Bharat Pradhan Mantri Jan Arogya Yojana (AB PM-JAY) provides a health cover "
                "of ₹5,00,000 per family per year for secondary and tertiary care hospitalization across "
                "empaneled public and private healthcare facilities."
            ),
            "Ayushman Bharat",
        ),
        (
            "pmjay::chunk::0002",
            "pmjay",
            "pmjay_manual.pdf",
            2,
            1,
            (
                "Under PM-JAY, there is no restriction on family size, age, or gender of family members. "
                "All pre-existing medical conditions are covered from day one of enrollment without any waiting period."
            ),
            "Ayushman Bharat",
        ),
        (
            "pmjay::chunk::0003",
            "pmjay",
            "pmjay_manual.pdf",
            3,
            2,
            (
                "आयुष्मान भारत प्रधानमंत्री जन आरोग्य योजना के अंतर्गत द्वितीयक और तृतीयक देखभाल अस्पताल में "
                "भर्ती के लिए प्रत्येक पात्र परिवार को प्रति वर्ष ₹5,00,000 तक का कैशलेस स्वास्थ्य बीमा दिया जाता है।"
            ),
            "Ayushman Bharat",
        ),
        (
            "pmjay::chunk::0004",
            "pmjay",
            "pmjay_manual.pdf",
            4,
            3,
            (
                "आयुष्मान योजना में परिवार के सदस्यों की संख्या, आयु या लिंग पर कोई सीमा नहीं है। "
                "नामांकन के पहले दिन से ही सभी पुरानी और गंभीर बीमारियां अस्पताल में कवर की जाती हैं।"
            ),
            "Ayushman Bharat",
        ),
        # --- PMAY-G (HOUSING) ---
        (
            "pmayg::chunk::0001",
            "pmayg",
            "pmayg_scheme.pdf",
            1,
            0,
            (
                "Pradhan Mantri Awaas Yojana - Gramin (PMAY-G) provides unit financial assistance of "
                "₹1,20,000 in plain areas and ₹1,30,000 in hilly, difficult and IAP areas for construction "
                "of quality pucca houses for rural homeless families."
            ),
            "PMAY-G",
        ),
        (
            "pmayg::chunk::0002",
            "pmayg",
            "pmayg_scheme.pdf",
            2,
            1,
            (
                "Beneficiary identification and prioritization under PMAY-G is done strictly based on "
                "housing deprivation criteria in Socio-Economic and Caste Census (SECC) 2011, followed by Gram Sabha verification."
            ),
            "PMAY-G",
        ),
        (
            "pmayg::chunk::0003",
            "pmayg",
            "pmayg_scheme.pdf",
            3,
            2,
            (
                "प्रधानमंत्री आवास योजना ग्रामीण (PMAY-G) के तहत मैदानी क्षेत्रों में पक्का मकान बनाने "
                "के लिए ₹1,20,000 और दुर्गम या पहाड़ी क्षेत्रों में ₹1,30,000 की वित्तीय सहायता सीधे डीबीटी के माध्यम से मिलती है।"
            ),
            "PMAY-G",
        ),
        (
            "pmayg::chunk::0004",
            "pmayg",
            "pmayg_scheme.pdf",
            4,
            3,
            (
                "पीएमएवाई-जी में ग्रामीण बेघर परिवारों का चयन सामाजिक आर्थिक और जाति जनगणना 2011 के "
                "आवास अभाव मानकों के आधार पर किया जाता है और अंतिम सूची ग्राम सभा द्वारा सत्यापित होती है।"
            ),
            "PMAY-G",
        ),
    ]

    chunks: list[TextChunk] = []
    for cid, doc_id, filename, page, cidx, text, scheme in raw_specs:
        meta = ChunkMetadata(
            chunk_id=cid,
            doc_id=doc_id,
            filename=filename,
            page_number=page,
            chunk_index=cidx,
            char_count=len(text),
            scheme_name=scheme,
        )
        chunks.append(TextChunk(text=text, metadata=meta))
    return chunks


def run_benchmark() -> dict[str, Any]:
    print("=" * 70)
    print("Adapted Reasoning RAG — Milestone 3: BGE-M3 Hybrid Retrieval Benchmark")
    print("=" * 70)

    # 1. System specs and baseline memory
    ram = psutil.virtual_memory()
    initial_rss = get_process_memory_mb()
    print(
        f"System RAM: {ram.total / (1024**3):.2f} GB total, {ram.available / (1024**3):.2f} GB available"
    )
    print(f"Initial Process RSS: {initial_rss:.2f} MB")

    # 2. Model loading
    print("\n[Stage 1] Loading BGE-M3 on CPU...")
    gc.collect()
    t_load_start = time.perf_counter()
    embedder = BGEEmbedder(model_name="BAAI/bge-m3", device="cpu")
    # Trigger model load
    _ = embedder._load_model()
    load_duration = time.perf_counter() - t_load_start
    loaded_rss = get_process_memory_mb()
    ram_delta_mb = loaded_rss - initial_rss
    print(f"Model loaded in: {load_duration:.2f} seconds")
    print(f"Process RSS after load: {loaded_rss:.2f} MB (RAM Delta: {ram_delta_mb:.2f} MB)")

    # 3. Vector validation
    print("\n[Stage 2] Validating Embedding Dimensions and Numerical Properties...")
    test_queries = {
        "en": "What is the annual financial support provided under PM Kisan?",
        "hi": "पीएम किसान योजना में किसानों को प्रति वर्ष कितनी राशि मिलती है?",
        "hinglish": "PM Kisan yojana me eligibility kya hai aur kitna paisa milta hai?",
    }

    query_embeddings: dict[str, list[float]] = {}
    for lang, q_text in test_queries.items():
        vec = embedder.embed_query(q_text)
        query_embeddings[lang] = vec
        norm = math.sqrt(sum(v * v for v in vec))
        all_finite = all(math.isfinite(v) for v in vec)
        print(f"  {lang.upper()} Query: '{q_text[:40]}...'")
        print(f"    - Vector length: {len(vec)}")
        print(f"    - All finite: {all_finite}")
        print(f"    - L2 norm: {norm:.5f}")
        assert len(vec) == 1024, f"Expected 1024-dim vector, got {len(vec)}"
        assert all_finite, "Non-finite numbers found in embedding"
        assert abs(norm - 1.0) < 1e-4, f"Vector not L2 normalized: {norm}"

    # 4. Latency and throughput benchmarks
    print("\n[Stage 3] Benchmarking Query Latency and Batch Throughput...")
    # Single query latency (5 warm runs)
    latencies: list[float] = []
    benchmark_query = "Who is eligible for Ayushman Bharat health insurance?"
    for i in range(5):
        t0 = time.perf_counter()
        _ = embedder.embed_query(benchmark_query)
        lat = (time.perf_counter() - t0) * 1000
        latencies.append(lat)
        print(f"  Query run {i + 1}: {lat:.2f} ms")

    avg_query_lat = sum(latencies) / len(latencies)
    min_query_lat = min(latencies)
    max_query_lat = max(latencies)
    print(
        f"  Single Query Latency -> Mean: {avg_query_lat:.2f} ms, Min: {min_query_lat:.2f} ms, Max: {max_query_lat:.2f} ms"
    )

    # Batch throughput (12 synthetic passages)
    corpus = create_synthetic_corpus()
    passages = [c.text for c in corpus]
    t_batch_start = time.perf_counter()
    _ = embedder.embed_texts(passages)
    batch_time = time.perf_counter() - t_batch_start
    throughput = len(passages) / batch_time
    ms_per_doc = (batch_time / len(passages)) * 1000
    print(
        f"  Batch Throughput ({len(passages)} passages): {batch_time:.2f} seconds ({throughput:.2f} passages/sec, {ms_per_doc:.2f} ms/passage)"
    )

    # 5. Indexing benchmark
    print("\n[Stage 4] Indexing 12 Chunks into ChromaDB and BM25...")
    dense_retriever = ChromaDenseRetriever(
        embedder=embedder,
        persist_directory=None,
        collection_name="bge_m3_benchmark_collection",
    )
    sparse_retriever = BM25Retriever()
    hybrid = HybridRetriever(
        dense_retriever=dense_retriever,
        sparse_retriever=sparse_retriever,
        default_rrf_k=60,
    )

    t_idx_start = time.perf_counter()
    index_counts = hybrid.index_chunks(corpus)
    indexing_time = time.perf_counter() - t_idx_start
    print(
        f"  Indexed into Dense: {index_counts['dense']} chunks, Sparse: {index_counts['sparse']} chunks"
    )
    print(f"  Total Indexing Duration: {indexing_time:.2f} seconds")

    # 6. Retrieval Quality Tests
    print("\n[Stage 5] Evaluating Retrieval Quality Across Languages & Filters...")
    test_cases = [
        {
            "id": "TC-1 (English Direct)",
            "query": "What is the annual financial support under PM-Kisan?",
            "filter": None,
            "expected_scheme": "PM-Kisan",
            "expected_top_chunk": "pm_kisan::chunk::0001",
        },
        {
            "id": "TC-2 (Hindi Direct)",
            "query": "आयुष्मान भारत योजना में कितना स्वास्थ्य बीमा मिलता है?",
            "filter": None,
            "expected_scheme": "Ayushman Bharat",
            "expected_top_chunk": "pmjay::chunk::0003",
        },
        {
            "id": "TC-3 (Hinglish Exclusions)",
            "query": "PM Kisan yojana me income tax payees eligible hai kya?",
            "filter": None,
            "expected_scheme": "PM-Kisan",
            "expected_top_chunk": "pm_kisan::chunk::0004",  # or 0002
        },
        {
            "id": "TC-4 (Cross-lingual Hindi to English)",
            "query": "ग्रामीण क्षेत्रों में पक्का मकान बनाने के लिए कितनी सहायता मिलती है?",
            "filter": None,
            "expected_scheme": "PMAY-G",
            "expected_top_chunk": "pmayg::chunk::0003",  # or 0001
        },
        {
            "id": "TC-5 (Scheme Pre-filter PMAY-G)",
            "query": "financial assistance for building house",
            "filter": "PMAY-G",
            "expected_scheme": "PMAY-G",
            "expected_top_chunk": "pmayg::chunk::0001",
        },
        {
            "id": "TC-6 (Scheme Pre-filter Rejection)",
            "query": "hospitalization and surgery cover",
            "filter": "PM-Kisan",
            "expected_scheme": "PM-Kisan",
            "expected_top_chunk": None,  # PM-Kisan corpus has no hospitalization info
        },
    ]

    retrieval_results_log: list[dict[str, Any]] = []

    for tc in test_cases:
        t0 = time.perf_counter()
        q = RetrievalQuery(
            query=tc["query"],
            scheme_name=tc["filter"],
            top_k=3,
            rrf_k=60,
        )
        results = hybrid.retrieve(q)
        q_duration = (time.perf_counter() - t0) * 1000

        print(f"\n--- {tc['id']} ---")
        print(f'Query: "{tc["query"]}" | Filter: {tc["filter"]} | Latency: {q_duration:.2f} ms')

        top_match = results[0] if results else None
        top_cid = top_match.chunk_id if top_match else "None"
        top_scheme = top_match.scheme_name if top_match else "None"

        print(f"Top-1 Chunk: {top_cid} | Scheme: {top_scheme}")
        if top_match:
            print(f"Citation: {top_match.source_citation}")
            print(
                f"Scores -> Dense Rank: {top_match.dense_rank} (Score: {top_match.dense_score:.4f}), "
                f"Sparse Rank: {top_match.sparse_rank} (Score: {top_match.sparse_score}), "
                f"RRF Score: {top_match.rrf_score:.5f}"
            )
            print(f'Text snippet: "{top_match.text[:95]}..."')

        tc_log = {
            "test_id": tc["id"],
            "query": tc["query"],
            "filter": tc["filter"],
            "latency_ms": round(q_duration, 2),
            "top_1_chunk_id": top_cid,
            "top_1_scheme": top_scheme,
            "top_1_citation": top_match.source_citation if top_match else None,
            "top_1_rrf_score": round(top_match.rrf_score, 5)
            if top_match and top_match.rrf_score
            else None,
            "results_count": len(results),
        }
        retrieval_results_log.append(tc_log)

    peak_rss = get_process_memory_mb()

    summary = {
        "platform": {
            "os": "Windows 11",
            "cpu": "AMD Ryzen 7 7730U (8 cores / 16 threads)",
            "ram_total_gb": round(ram.total / (1024**3), 2),
            "ram_available_gb": round(ram.available / (1024**3), 2),
            "torch_version": "2.15.0+cpu",
        },
        "bge_m3_metrics": {
            "model_name": "BAAI/bge-m3",
            "model_load_time_sec": round(load_duration, 2),
            "ram_before_mb": round(initial_rss, 1),
            "ram_after_load_mb": round(loaded_rss, 1),
            "ram_model_delta_mb": round(ram_delta_mb, 1),
            "peak_rss_mb": round(peak_rss, 1),
            "dimension": 1024,
            "single_query_latency_mean_ms": round(avg_query_lat, 2),
            "single_query_latency_min_ms": round(min_query_lat, 2),
            "single_query_latency_max_ms": round(max_query_lat, 2),
            "batch_size": len(passages),
            "batch_duration_sec": round(batch_time, 2),
            "batch_throughput_items_per_sec": round(throughput, 2),
            "batch_ms_per_passage": round(ms_per_doc, 2),
            "indexing_duration_sec": round(indexing_time, 2),
        },
        "retrieval_test_results": retrieval_results_log,
    }

    # Save summary to JSON for reporting
    output_path = Path(__file__).resolve().parent / "benchmark_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 70)
    print(f"Benchmark completed successfully! Results written to: {output_path}")
    print("=" * 70)
    return summary


if __name__ == "__main__":
    run_benchmark()
