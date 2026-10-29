"""AI-Based Project Idea Novelty Checker — FastAPI backend.

POST /api/check {title, abstract, check_external}
  -> novelty_score (0-100), verdict, closest_matches[] across
     Department (FAISS) + arXiv + GitHub.
Novelty = 100 * (1 - max_similarity), clamped. The closest match drives the
score: one near-duplicate is enough to make an idea not novel.
"""
import re
from pathlib import Path
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .embedder import Embedder
from .indexer import DeptIndex, load_corpus
from .external import search_arxiv, search_github

STOPWORDS = set("a an the of for to in on and or with using use used based system systems application app project model models data user users".split())
DUPLICATE_THRESHOLD = 0.55  # operating point from the 60-pair evaluation


def shared_terms(query_text: str, match_text: str, n: int = 6):
    """Distinctive content words shared by the query and a match — the
    plain-language evidence for *why* the two are close."""
    def words(t):
        return [w for w in re.findall(r"[a-z]{3,}", t.lower()) if w not in STOPWORDS]
    q, m = set(words(query_text)), words(match_text)
    shared = [w for w in dict.fromkeys(m) if w in q]
    return shared[:n]

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
    query_text = f"{req.title}. {req.abstract}"
    query_vec = embedder.encode(query_text)
    matches = dept.search(query_vec, k=5)
    if req.check_external:
        matches += await search_arxiv(req.title, req.abstract, embedder, query_vec, k=3)
        matches += await search_github(req.title, req.abstract, embedder, query_vec, k=3)
    matches.sort(key=lambda m: m["similarity"], reverse=True)
    for m in matches:
        m["shared_terms"] = shared_terms(query_text, f"{m['title']}. {m.get('abstract') or ''}")
    max_sim = matches[0]["similarity"] if matches else 0.0
    score = round(max(0.0, min(100.0, (1 - max_sim) * 100)), 1)
    density = dept.density(query_vec)
    top = matches[0] if matches else None
    if top and max_sim >= DUPLICATE_THRESHOLD:
        explanation = (
            f"Closest prior work: “{top['title']}” ({top['source']}) at {max_sim:.0%} similarity — "
            f"above the {DUPLICATE_THRESHOLD:.0%} duplicate threshold set by our 60-pair evaluation. "
            f"As submitted, this idea would read as a repeat of that work."
        )
    elif top:
        explanation = (
            f"No near-duplicate found. The closest prior work is “{top['title']}” ({top['source']}) "
            f"at {max_sim:.0%} similarity, below the {DUPLICATE_THRESHOLD:.0%} duplicate threshold. "
            f"Differentiation should be argued against this match."
        )
    else:
        explanation = "No prior work retrieved from any source."
    if density["local_crowding"] >= 0.35:
        explanation += (
            f" Note: the topic area itself is crowded — average similarity to the 5 nearest "
            f"department projects is {density['local_crowding']:.0%}, so the proposal must state "
            f"its difference explicitly."
        )
    return {
        "novelty_score": score,
        "verdict": verdict_for(score),
        "max_similarity": max_sim,
        "embedder": embedder.backend,
        "closest_matches": matches[:8],
        "density": density,
        "explanation": explanation,
    }


app.mount("/static", StaticFiles(directory=BASE / "static"), name="static")


@app.get("/")
def home():
    return FileResponse(BASE / "static" / "index.html")
