# Adapted Reasoning RAG

A bilingual (Hindi, English, Hinglish), voice-enabled Retrieval-Augmented Generation assistant that provides source-cited eligibility verdicts for Indian government welfare schemes.

## Covered Schemes

| # | Scheme |
|---|--------|
| 1 | PM-Kisan |
| 2 | Ayushman Bharat (PM-JAY) |
| 3 | PM Mudra Yojana |
| 4 | PM Awas Yojana (PMAY) |
| 5 | Sukanya Samriddhi Yojana |

## Key Features

- **Claim-Level Evidence Sufficiency** — decomposes questions into verifiable claims, checks each against official documents, and abstains when evidence is missing.
- **Hybrid Retrieval** — BGE-M3 dense embeddings + BM25 keyword search, fused with Reciprocal Rank Fusion.
- **Deterministic Eligibility Engine** — rule-based verdicts: Eligible, Not Eligible, or Need More Information.
- **Multi-turn Dialogue** — collects user profile details (age, income, occupation, landholding) across conversation turns.
- **Voice I/O** — Whisper ASR and AI4Bharat / Coqui TTS for Hindi, English, and Hinglish.

## Project Structure

```
Major_Project/
├── backend/
│   ├── app/
│   │   ├── core/            # FastAPI app, router, settings, pipeline orchestrator
│   │   ├── ingestion/       # PDF extraction, cleaning, chunking
│   │   ├── retrieval/       # Hybrid search (BGE-M3 + BM25), targeted retrieval
│   │   ├── understanding/   # Language detection, intent, query rewriting
│   │   ├── reasoning/       # Claim decomposition, verification, per-claim gate
│   │   ├── eligibility/     # Deterministic rule engine for 5 schemes
│   │   ├── dialogue/        # Multi-turn dialogue manager, profile extraction
│   │   ├── generation/      # Cited answer generator with abstention
│   │   ├── voice/           # ASR (Whisper) and TTS
│   │   └── evaluation/      # RAGAS, gold set scoring, comparison systems
│   └── tests/
│       ├── unit/            # Per-module isolated tests
│       └── integration/     # Cross-module pipeline tests
├── frontend/                # React.js / Next.js + TailwindCSS
├── data/
│   ├── raw/                 # Original scheme PDFs
│   ├── processed/           # Chunks, embeddings, indices
│   └── evaluation/          # Gold set, labelled claim–passage pairs
├── configs/                 # YAML/JSON configs, scheme metadata
├── prompts/                 # Versioned LLM prompt templates
├── notebooks/               # Jupyter experiments and EDA
├── scripts/                 # Utility and deployment scripts
├── docs/                    # SRS, architecture docs, user guide
├── .gitignore
├── .env.example             # Environment variable template
├── pyproject.toml           # Python project config
└── README.md
```

## Setup

```bash
# 1. Clone the repository
git clone https://github.com/krishyadav25/Major_project.git
cd Major_project

# 2. Create a virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows

# 3. Install dependencies (when added)
pip install -e ".[dev]"

# 4. Copy and configure environment variables
cp .env.example .env
# Edit .env with your settings
```

## Team

- **Krish Yadav** (500121939)
- **Pranjal Seluriyal** (500122853)
- **Isha** (500119683)

**Mentor:** Dr. Mansi Gupta
**Institution:** University of Petroleum & Energy Studies, Dehradun

## License

MIT
