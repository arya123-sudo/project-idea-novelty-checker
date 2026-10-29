# What Was Used — Full Inventory

Every component of the Novelty Checker, what it does, and why it's there.

## 1. Core ML / AI

| Component | What it is | Role in the system |
|---|---|---|
| **Sentence-BERT `all-MiniLM-L6-v2`** (Reimers & Gurevych lineage) | Transformer sentence-embedding model, 384-dimensional vectors | Converts title+abstract into a semantic vector positioned by *meaning* — this is what catches paraphrased duplicates |
| **fastembed (ONNX Runtime)** | Inference engine running the same MiniLM model in ONNX form | Production embedding on the server — same model, same scores, no PyTorch, fits 512 MB RAM |
| **Cosine similarity** | Angle between embedding vectors (via L2-normalised inner product) | The similarity measure behind every comparison |
| **FAISS `IndexFlatIP`** (Meta) | Vector similarity-search library | Exact nearest-neighbour search over the department corpus; scales to millions of vectors later |
| **TF-IDF** (scikit-learn) | Keyword-weighting baseline | Two roles: automatic fallback if SBERT fails, and the *baseline* in our evaluation that proves why semantic embeddings are needed |
| **Neighbourhood density** (own implementation, after Wang et al. 2025) | Mean similarity to the 5 nearest corpus projects | The crowdedness lens — flags saturated topic areas even without an exact duplicate |

## 2. The score (our own logic)

- **Novelty Score = 100 × (1 − max similarity)**, clamped 0–100
- **Verdict bands:** ≥70 Highly Novel · 45–69 Moderately Novel · <45 Low Novelty
- **Duplicate threshold 0.55** — chosen from the 60-pair evaluation (costlier error = falsely branding a new idea a duplicate)
- **Evidence explanation generator** (own code): verdict justification text + shared-concept extraction per match (stopword-filtered token overlap)

## 3. Backend & Frontend

| Component | Role |
|---|---|
| **Python 3.12** | Language |
| **FastAPI + Uvicorn** | REST API (`POST /api/check`, `GET /api/health`) and web server |
| **Pydantic** | Request validation |
| **httpx** | Async calls to arXiv/GitHub APIs |
| **Single-file HTML/CSS/JS** (`static/index.html`) | Student/faculty UI — score gauge, verdict, explanation box, crowdedness line, match cards with concept chips |
| **numpy** | Vector maths, density computation |

## 4. External data sources (live APIs, no keys needed)

| Source | How used |
|---|---|
| **Department corpus** (`data/dept_projects.json`, 36 projects) | ⚠️ SAMPLE data — representative topics standing in for the real Acharya AIML archive; replacing this one file upgrades the system to real data |
| **arXiv API** (export.arxiv.org) | Live candidate papers by keyword, then re-ranked by embedding similarity |
| **GitHub Search API** | Live candidate repositories by keyword, then re-ranked by embedding similarity |

## 5. Evaluation (all measured, none assumed)

| Experiment | Tooling | Headline result |
|---|---|---|
| Pair classification, 60 labeled pairs (30 paraphrase-similar / 30 different) | scikit-learn metrics (`evaluation/evaluate.py`) | ROC-AUC 1.000, Precision 1.000, Recall 0.933, F1 0.966 @ 0.55 |
| SBERT vs TF-IDF comparison | same script | TF-IDF F1 collapses 0.868 → 0.125 across thresholds; SBERT separation: similar mean 0.680 vs different mean 0.056 |
| Retrieval test (DPR-style) | `evaluation/retrieval_eval.py` | SBERT Recall@3 1.000, MRR 0.711 · TF-IDF Recall@3 0.867 (misses the original entirely in 4/30) |
| Smoke tests | pytest (`tests/test_smoke.py`) | 3/3 pass |

## 6. Deployment & infrastructure

| Component | Detail |
|---|---|
| **Render** (free tier, Ohio region) | Public hosting — https://project-idea-novelty-checker.onrender.com; `render.yaml` blueprint; build = `requirements-deploy.txt` (fastapi, uvicorn, httpx, numpy, scikit-learn, faiss-cpu, fastembed — deliberately no PyTorch) |
| **GitHub** | Source of truth — https://github.com/arya123-sudo/project-idea-novelty-checker |
| Note | Repo connected to Render as a *public git repository*, so pushes do **not** auto-redeploy — redeploys are one Manual Deploy click in the dashboard |

## 7. Research base (12 verified papers — full pack in `RESEARCH_PAPERS.md`)

- **Stack foundations:** Sentence-BERT (2019) · FAISS (2019) · Dense Passage Retrieval (2020)
- **Novelty science:** Shibayama et al. (2021) — embedding-distance novelty, validated · Uzzi et al. (2013, Science) — atypical combinations predict impact
- **Paper embeddings:** SPECTER (2020) · SPECTER2 / SciRepEval (2023)
- **Modern novelty assessment:** Si et al. (ICLR 2025) · Wang et al. RND (2025) · NoveltyRank (2025) · What Is Novel? (2026) · All That Glitters is Not Novel (2025)

## 8. Documents in this repo

| File | What it is |
|---|---|
| `REPORT.md` / `REPORT.pdf` | Full project report (problem, literature, methodology, results, limitations) |
| `RESEARCH_PAPERS.md` / `.pdf` | Annotated 12-paper literature pack |
| `EXPLANATION.md` / `.pdf` | Pitch, technique justifications, usefulness, viva Q&A |
| `STACK.md` | This inventory |
| `README.md` | Overview + live demo link |
