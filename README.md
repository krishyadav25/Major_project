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

```text
Major_Project/
├── backend/
│   ├── app/
│   │   ├── core/            # FastAPI app, router, settings, logging, schemas
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
│       ├── unit/            # Per-module isolated tests (e.g. test_health.py)
│       └── integration/     # Cross-module pipeline tests (e.g. test_app_startup.py)
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

## Getting Started (Windows)

### Prerequisites
- Windows 10/11
- Python 3.10+ (Recommended: **Python 3.12.10**)
- Git

### 1. Clone the Repository
```powershell
git clone https://github.com/krishyadav25/Major_project.git
cd Major_project
```

### 2. Set Up Virtual Environment
```powershell
# Create virtual environment
python -m venv .venv

# Activate on Windows (PowerShell)
.venv\Scripts\Activate.ps1

# Or activate on Windows (Command Prompt)
.venv\Scripts\activate.bat
```

### 3. Install Dependencies
```powershell
# Upgrade pip
python -m pip install --upgrade pip

# Install project dependencies and development tools
pip install -e ".[dev]"
```

### 4. Configure Environment Variables
```powershell
# Copy the example environment template
Copy-Item .env.example .env

# Edit .env with your local settings (e.g., host, port, Ollama URL)
```

### 5. Run the FastAPI Backend Server
```powershell
# Option A: Run directly with Uvicorn (with live reload)
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# Option B: Run via Python module
python -m app.main
```

The API will be available at:
- **Base URL:** http://localhost:8000
- **Health Check:** http://localhost:8000/health

### 6. Interactive API Documentation (Swagger / OpenAPI)
Once the server is running, open your browser:
- **Swagger UI:** http://localhost:8000/docs
- **ReDoc UI:** http://localhost:8000/redoc
- **OpenAPI JSON:** http://localhost:8000/openapi.json

### 7. Run Automated Tests
```powershell
# Run the entire test suite
pytest

# Run tests with detailed verbose output
pytest -v

# Run with test coverage report
pytest --cov=app backend/tests
```

## Team

- **Krish Yadav** (500121939)
- **Pranjal Seluriyal** (500122853)
- **Isha** (500119683)

**Mentor:** Dr. Mansi Gupta  
**Institution:** University of Petroleum & Energy Studies, Dehradun

## License

MIT
