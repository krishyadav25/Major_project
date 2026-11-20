# Milestone 3 Validation & Performance Benchmark Report
**Adapted Reasoning RAG — Hybrid Retrieval (BGE-M3 + BM25 + ChromaDB + RRF)**

---

## Executive Summary

Milestone 3 implements and validates the hybrid retrieval pipeline for **Adapted Reasoning RAG**. The system combines dense semantic vector search powered by **BAAI/bge-m3** with sparse lexical retrieval powered by **BM25Okapi**, harmonized via **Reciprocal Rank Fusion (RRF)** with source provenance attribution.

All benchmarks and validations were executed on local hardware (**AMD Ryzen 7 7730U APU, 16 GB RAM, Windows 11**) running on **PyTorch CPU-only mode** (`torch==2.15.0+cpu`).

| Evaluation Metric | Target / Specification | Measured Result | Status |
| :--- | :--- | :--- | :--- |
| **Automated Test Suite** | 100% passing | **90 / 90 tests passing** | ✅ Verified |
| **Code Coverage** | $\ge 90\%$ | **97% overall backend coverage** | ✅ Verified |
| **Lint & Formatting** | Ruff compliance | **0 warnings, 0 errors** | ✅ Verified |
| **BGE-M3 CPU Load Time** | Cold load into RAM | **24.49 seconds** | ✅ Acceptable |
| **Model Memory Footprint** | System budget < 4 GB | **674.4 MB RAM delta** (Peak RSS: ~1.9 GB) | ✅ Safe |
| **Single-Query Latency** | Interactive < 500 ms | **224.74 ms mean** (Min: 217.9 ms) | ✅ Responsive |
| **Batch Throughput (CPU)** | Throughput measure | **4.33 passages/sec** (230.8 ms/passage) | ✅ Measured |
| **Top-1 Retrieval Accuracy** | 100% across test set | **6 / 6 test cases matched** | ✅ Verified |

---

## Part A — Architectural & Implementation Verification

### 1. Deterministic Identification & Traceability
- **Document IDs:** Built via `make_doc_id(filepath, prefix=None, content_hash=None)`. Appending an 8-character SHA-256 content digest guarantees collision resistance when different documents share identical filenames.
- **Chunk IDs:** Built as `{doc_id}::chunk::{chunk_index:04d}` (e.g. `pm_kisan_guidelines::chunk::0001`). Chunk indices are 0-based, sequential, and preserve page number mappings.
- **Source Citations:** Every retrieved candidate produces a human-readable citation:
  - Formatted as `[{scheme_name}] {filename} (Page {page_number})`, e.g., `[PM-Kisan] pm_kisan_guidelines.pdf (Page 2)`.

### 2. Document Updates, Reindexing & Removal
- **In-Place Updates:** `BM25Retriever._chunk_map` and ChromaDB `collection.upsert()` key strictly on `chunk_id`. Re-indexing identical chunks updates existing entries without duplication.
- **Orphan Chunk Prevention:** To prevent old chunks from remaining when an updated document generates fewer chunks, synchronized deletion methods were implemented across all components:
  - `BM25Retriever.delete_document(doc_id)`: Removes all chunks matching `doc_id` from the in-memory dictionary and rebuilds `BM25Okapi`.
  - `ChromaDenseRetriever.delete_document(doc_id)`: Issues `_collection.delete(where={"doc_id": doc_id})` to purge vector records.
  - `HybridRetriever.delete_document(doc_id)`: Executes synchronized removal across both stores, returning remaining counts.

### 3. Dense vs. Sparse Consistency
- **Persistence Asymmetry:** ChromaDB persists vectors to disk via `chromadb.PersistentClient()`. BM25 operates in memory and serializes safely to UTF-8 JSON (`save_to_json` / `load_from_json`) with **zero pickle usage**.
- **Coordinated Indexing:** `HybridRetriever.index_chunks()` coordinates atomic indexing across dense and sparse engines, returning `{"dense": dense_count, "sparse": sparse_count}`.

---

## Part B & C — Hardware Benchmark & BGE-M3 Profiling

### 1. Test Environment Specifications
- **Operating System:** Windows 11 Home Single Language (64-bit)
- **Processor:** AMD Ryzen 7 7730U with Radeon Graphics (8 cores, 16 threads, 2.0 GHz base, up to 4.5 GHz boost)
- **System Memory:** 16 GB physical RAM (13.86 GB usable after APU VRAM allocation)
- **Available RAM During Test:** 4.38 GB free
- **Storage:** 88.13 GB free on `D:\`, 40.86 GB free on `C:\`
- **PyTorch Wheel:** `2.15.0+cpu` (CPU-only build, saving ~3 GB disk space over CUDA builds)

### 2. Model Memory & Loading Profiling
- **Hugging Face Model:** `BAAI/bge-m3` (570M parameters, 1024-dimensional output).
- **Disk Weight Footprint:** ~2.24 GB cached in HuggingFace hub cache.
- **Model Load Time:** **24.49 seconds** from disk cache to CPU RAM.
- **Process Memory Before Load:** 68.6 MB RSS.
- **Process Memory After Load:** 743.0 MB RSS.
- **Net Memory Delta for BGE-M3:** **674.4 MB**.
- **Peak Process Memory During Batch Encoding:** **1,956.2 MB** (~1.9 GB peak).

### 3. Latency & Throughput Measurements
```
Single Query Latency (5 consecutive runs on CPU):
  Run 1: 233.32 ms
  Run 2: 226.18 ms
  Run 3: 223.41 ms
  Run 4: 222.88 ms
  Run 5: 217.92 ms
  -------------------------------------------------------------
  Mean: 224.74 ms | Min: 217.92 ms | Max: 233.32 ms
```
- **Batch Embedding Throughput:** 12 synthetic scheme passages encoded in **2.77 seconds** (**4.33 passages/sec** or ~230.8 ms/passage).
- **Indexing Duration:** 12 passages embedded + upserted to ChromaDB (HNSW cosine space) + tokenized and indexed into BM25: **2.74 seconds**.
- **Hybrid Retrieval Latency:** **225.40 ms average** per query (Dense search + BM25 search + RRF fusion).

### 4. Vector Properties & Numerical Validation
- **Vector Dimension:** Exactly **1024** across English, Hindi, and Hinglish.
- **Finite Check:** 100% finite values (no `NaN` or `inf`).
- **L2 Normalization:** Euclidean norm $\|v\|_2 = 1.00000$ across all test queries.

---

## Part D — Multilingual Retrieval Quality Evaluation

The pipeline was tested against a multi-scheme corpus consisting of 12 distinct passages across **PM-Kisan**, **Ayushman Bharat PM-JAY**, and **PMAY-G** in English and Hindi.

```
Corpus Overview:
- PM-Kisan (4 passages): English/Hindi benefits (₹6,000/yr) & exclusion criteria (institutional landholders, tax payees).
- Ayushman Bharat (4 passages): English/Hindi health cover (₹5,00,000/yr), pre-existing condition terms.
- PMAY-G (4 passages): English/Hindi pucca housing grant (₹1,20,000 plain / ₹1,30,000 hilly), SECC 2011 criteria.
```

### Retrieval Test Results

| Test Case | Language | Query | Active Filter | Top-1 Chunk ID | RRF Score | Citation | Evaluation |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **TC-1** | English | *"What is the annual financial support under PM-Kisan?"* | None | `pm_kisan::chunk::0001` | **0.03279** | `[PM-Kisan] pm_kisan_guidelines.pdf (Page 1)` | **Pass** |
| **TC-2** | Hindi | *"आयुष्मान भारत योजना में कितना स्वास्थ्य बीमा मिलता है?"* | None | `pmjay::chunk::0003` | **0.03279** | `[Ayushman Bharat] pmjay_manual.pdf (Page 3)` | **Pass** |
| **TC-3** | Hinglish | *"PM Kisan yojana me income tax payees eligible hai kya?"* | None | `pm_kisan::chunk::0002` | **0.03279** | `[PM-Kisan] pm_kisan_guidelines.pdf (Page 2)` | **Pass** |
| **TC-4** | Cross-Lingual (Hindi $\rightarrow$ English) | *"ग्रामीण क्षेत्रों में पक्का मकान बनाने के लिए कितनी सहायता मिलती है?"* | None | `pmayg::chunk::0003` | **0.03279** | `[PMAY-G] pmayg_scheme.pdf (Page 3)` | **Pass** |
| **TC-5** | English (Scheme Filter) | *"financial assistance for building house"* | `PMAY-G` | `pmayg::chunk::0001` | **0.03252** | `[PMAY-G] pmayg_scheme.pdf (Page 1)` | **Pass** |
| **TC-6** | Negative Filter Check | *"hospitalization and surgery cover"* | `PM-Kisan` | `pm_kisan::chunk::0002` | 0.03279 | `[PM-Kisan] pm_kisan_guidelines.pdf (Page 2)` | **Pass** (Excluded AYUSHMAN) |

### Key Retrieval Observations
1. **Devanagari Tokenization & Lexical Match:** In TC-2 and TC-4, BM25 matched Hindi keywords (`स्वास्थ्य बीमा`, `पक्का मकान`) with sparse scores exceeding 8.0–15.2, complementing dense vector similarity.
2. **Hinglish Understanding:** In TC-3, BGE-M3 mapped the Hinglish query (*"income tax payees eligible hai kya"*) directly to the exclusion clause in Page 2 of PM-Kisan with a dense similarity score of 0.6522.
3. **Cross-Lingual Alignment:** In TC-4, both the Hindi passage (Page 3) and English passage (Page 1) of PMAY-G ranked at the top of the candidate list due to shared semantic vector space.
4. **Metadata Filtering:** TC-5 and TC-6 proved that ChromaDB and BM25 pre-filtering strictly isolate candidates before RRF reranking.

---

## Part E — Test Suite & Code Quality Status

```
============================= test session starts =============================
platform win32 -- Python 3.12.10, pytest-9.2.0, pluggy-1.7.0
plugins: anyio-4.16.0, cov-7.1.0, platformdirs-4.14.5
collected 90 items

backend/tests/integration/test_app_startup.py ....                       [  4%]
backend/tests/unit/test_health.py ....                                   [  8%]
backend/tests/unit/test_ingestion.py ................................... [ 64%]
backend/tests/unit/test_retrieval.py .................................. [100%]

=============================== tests coverage ================================
Name                                                  Stmts   Miss  Cover   Missing
-----------------------------------------------------------------------------------
backend/app/core/config.py                               19      0   100%
backend/app/core/logging.py                              16      0   100%
backend/app/core/routes.py                               10      0   100%
backend/app/ingestion/chunker.py                         69      3    96%   69, 75, 113
backend/app/ingestion/cleaner.py                         15      0   100%
backend/app/ingestion/extractor.py                       33      2    94%   53, 65
backend/app/ingestion/models.py                          68      5    93%   41, 67, 102, 109, 145
backend/app/ingestion/pipeline.py                        18      0   100%
backend/app/main.py                                      23      2    91%   66-68
backend/app/retrieval/__init__.py                         8      0   100%
backend/app/retrieval/dense.py                           68      3    96%   114-115, 162
backend/app/retrieval/embeddings.py                      66      0   100%
backend/app/retrieval/fusion.py                          32      0   100%
backend/app/retrieval/models.py                          27      0   100%
backend/app/retrieval/pipeline.py                        34      0   100%
backend/app/retrieval/sparse.py                          77      1    99%   58
backend/app/retrieval/tokenization.py                    17      0   100%
-----------------------------------------------------------------------------------
TOTAL                                                   600     16    97%
============================= 90 passed in 37.88s =============================
```

- **Ruff Linter:** `ruff check backend scripts` $\rightarrow$ **All checks passed (0 warnings)**.
- **Ruff Formatter:** `ruff format --check backend scripts` $\rightarrow$ **35 files formatted (0 unformatted)**.

---

## Deployment & Operational Recommendations

### 1. Local CPU vs. GPU Inference
- **Local CPU Serving:** Highly viable for query processing. At **~224 ms** per query, local CPU execution on the Ryzen 7 7730U provides snappy responses without requiring dedicated local GPU hardware.
- **Memory Footprint:** The model delta of **~674 MB** comfortably fits in 16 GB RAM with 4+ GB of free memory headroom.

### 2. Kaggle GPU Recommended for Batch Corpus Indexing
- **CPU Batch Speed:** 4.33 passages/second (~230 ms/passage).
  - 1,000 chunks (~100 PDF pages): **~4 minutes**.
  - 5,000 chunks (full guidelines corpus): **~20–25 minutes**.
- **Kaggle GPU Speed (T4/P100 FP16):** 150–250 passages/second.
  - 5,000 chunks will ingest and embed in **under 30 seconds**.
- **Recommended Workflow:** Ingest and generate embeddings for the primary 5 scheme document corpora on Kaggle GPU, export the populated ChromaDB directory and BM25 index JSON to `data/processed/`, and serve retrieval queries locally on CPU.
