"""AI-Based Project Idea Novelty Checker — FastAPI backend.

POST /api/check {title, abstract, check_external}
  -> novelty_score (0-100), verdict, closest_matches[] across
     Department (FAISS) + arXiv + GitHub.
Novelty = 100 * (1 - max_similarity), clamped. The closest match drives the
score: one near-duplicate is enough to make an idea not novel.
"""
from pathlib import Path
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .embedder import Embedder
from .indexer import DeptIndex, load_corpus
from .external import search_arxiv, search_github

BASE = Path(__file__).resolve().parent.parent
app = FastAPI(title="Project Idea Novelty Checker")

_corpus = load_corpus()
embedder = Embedder(corpus_texts=[f"{p['title']}. {p['abstract']}" for p in _corpus])
dept = DeptIndex(embedder)


class CheckRequest(BaseModel):
    title: str
    abstract: str
    check_external: bool = True


def verdict_for(score: float) -> str:
    if score >= 70:
        return "Highly Novel"
    if score >= 45:
        return "Moderately Novel — differentiate clearly from the closest matches"
    return "Low Novelty — very similar work already exists"


@app.get("/api/health")
def health():
    return {"status": "ok", "embedder": embedder.backend, "dept_projects": len(_corpus)}


@app.post("/api/check")
async def check(req: CheckRequest):
    query_vec = embedder.encode(f"{req.title}. {req.abstract}")
    matches = dept.search(query_vec, k=5)
    if req.check_external:
        matches += await search_arxiv(req.title, req.abstract, embedder, query_vec, k=3)
        matches += await search_github(req.title, req.abstract, embedder, query_vec, k=3)
    matches.sort(key=lambda m: m["similarity"], reverse=True)
    max_sim = matches[0]["similarity"] if matches else 0.0
    score = round(max(0.0, min(100.0, (1 - max_sim) * 100)), 1)
    return {
        "novelty_score": score,
        "verdict": verdict_for(score),
        "max_similarity": max_sim,
        "embedder": embedder.backend,
        "closest_matches": matches[:8],
    }


app.mount("/static", StaticFiles(directory=BASE / "static"), name="static")


@app.get("/")
def home():
    return FileResponse(BASE / "static" / "index.html")
