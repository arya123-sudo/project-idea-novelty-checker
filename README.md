# AI-Based Project Idea Novelty Checker

Mini Project — Department of AI & ML, Acharya Institute of Technology, Bengaluru

**Team:** Arya R Yaligar (1AY24AI012) · Bhargav Raj (1AY24AI019) · D Bargav (1AY24AI028) · G Vajresh Shenoy (1AY24AI034)
**Guide:** Mr. Praveen Arokiaraj, Assistant Professor, Dept. of AI & ML

## Problem
Project ideas are often rejected late as "already done". Students cannot easily check past department projects, arXiv papers and GitHub repos, and keyword search misses reworded / paraphrased versions of the same idea. Late rejection wastes weeks.

## What this does
Enter a new project **title + abstract**. The system:
1. Embeds it with Sentence-Transformers (`all-MiniLM-L6-v2`)
2. Searches a **FAISS** index of past department projects + live **arXiv** and **GitHub** results
3. Returns a **Novelty Score (0–100)**, a verdict, and the **closest matches** with cosine similarity — so a student/faculty can see *why* an idea looks novel or duplicate, early.

Verdicts: `>=70` Highly Novel · `45–69` Moderately Novel — differentiate clearly · `<45` Low Novelty — very similar work exists

## Architecture (as in synopsis)
```
New Idea (Title + Abstract)
        ↓
Sentence-Transformer Embedding
        ↓
FAISS Search: Dept corpus + arXiv + GitHub
        ↓
Cosine Similarity Score
        ↓
Novelty Verdict + Closest Matches
```
Backend: **FastAPI** · Evaluation: **ROC-AUC, Precision, Recall, F1** on labeled similar/different pairs

> Honest fallback: if `sentence-transformers` is not installed, the app automatically falls back to TF-IDF embeddings so the demo still runs offline. Install the full requirements to use SBERT (the synopsis model).

## Run it
```bash
cd novelty-checker
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt        # full: SBERT + FAISS + FastAPI
# light demo only: pip install -r requirements-light.txt
uvicorn app.main:app --reload --port 8000
```
Open http://localhost:8000 — paste a title + abstract and click **Check Novelty**.

API: `POST /api/check`  `{"title": "...", "abstract": "...", "check_external": true}`

## Evaluation
```bash
python evaluation/evaluate.py
```
Uses `data/labeled_pairs.csv` — **60 labeled pairs** (30 similar/paraphrased, 30 different) — and prints ROC-AUC, Precision, Recall, F1 at the operating threshold.

**Measured results (SBERT, threshold 0.55): ROC-AUC 1.000 · Precision 1.000 · Recall 0.933 · F1 0.966**
TF-IDF fallback on the same set: ROC-AUC 0.933 · F1 0.868 — see `REPORT.md` §6 for the full comparison and why SBERT matters for paraphrases.

Live check example — "Crop Disease Detection using Deep Learning" → **Novelty 8.7 (Low Novelty)**, matched against the dept corpus (0.913), GitHub repos (0.822) and arXiv papers (0.807) in one result list.

## Data
- `data/dept_projects.json` — sample department corpus (replace/extend with the real department archive — this is the project's unique corpus no public tool has)
- `data/labeled_pairs.csv` — labeled pairs for evaluation (extend to 60–100 pairs for the final report)

## Structure
```
app/main.py        FastAPI app + /api/check
app/embedder.py    SBERT embedder with TF-IDF fallback
app/indexer.py     FAISS index over the dept corpus
app/external.py    Live arXiv + GitHub search
static/index.html  Web UI (student/faculty)
evaluation/        Metrics script
tests/             Smoke tests
```
