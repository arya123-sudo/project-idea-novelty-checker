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
from .external import (search_arxiv, search_github, search_semantic_scholar,
                       search_openalex)

STOPWORDS = set("a an the of for to in on and or with using use used based system systems application app project model models data user users time one two new also can will helps help early many much different inside within among across via per".split())
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


def gap_analysis(req, matches, density, external: bool):
    """Evidence-based research-gap analysis. Every line is computed from the
    retrieved matches themselves — no generative opinion: concepts in the
    idea that no close prior work covers, sources where nothing close
    exists, and the implementation gap when no similar code exists."""
    gaps = []
    corpus_text = " ".join(f"{m['title']} {m.get('abstract') or ''}" for m in matches[:8]).lower()

    def root(w):  # crude singular so crops~crop, photos~photographs match
        return w[:-1] if len(w) > 4 and w.endswith("s") else w

    corpus_roots = {root(w) for w in re.findall(r"[a-z]{4,}", corpus_text)}
    concepts = []
    for w in re.findall(r"[a-z]{4,}", f"{req.title} {req.abstract}".lower()):
        if w not in STOPWORDS and w not in concepts:
            concepts.append(w)
    uncovered = [c for c in concepts
                 if root(c) not in corpus_roots
                 and not any(r.startswith(root(c)[:5]) for r in corpus_roots)][:4]
    for c in uncovered:
        gaps.append(f"Concept “{c}” appears in your idea but in none of the closest prior work — "
                    f"this is a differentiating angle; state it explicitly in the synopsis.")
    sources = ["Department"] + (["arXiv", "GitHub", "Semantic Scholar", "OpenAlex"] if external else [])
    for s in sources:
        best = max([m["similarity"] for m in matches if m["source"] == s], default=0.0)
        if best < 0.45:
            gaps.append(f"No close prior work retrieved from {s} in this check (best match {best:.0%}) — that channel looks open.")
    gh = max([m["similarity"] for m in matches if m["source"] == "GitHub"], default=0.0)
    if external and gh < 0.45:
        gaps.append("No close GitHub implementation exists — building a working implementation is itself a contribution.")
    if density["local_crowding"] >= 0.35:
        gaps.append("Caution: the department topic area is crowded — novelty must come from method or application, not the topic alone.")
    return gaps


@app.post("/api/check")
async def check(req: CheckRequest):
    query_text = f"{req.title}. {req.abstract}"
    query_vec = embedder.encode(query_text)
    matches = dept.search(query_vec, k=5)
    if req.check_external:
        # All four sources in parallel — sequential calls made a full check
        # take ~30 s. Each source still fails soft on rate limits.
        import asyncio
        extra = await asyncio.gather(
            search_arxiv(req.title, req.abstract, embedder, query_vec, k=3),
            search_github(req.title, req.abstract, embedder, query_vec, k=3),
            search_semantic_scholar(req.title, req.abstract, embedder, query_vec, k=3),
            search_openalex(req.title, req.abstract, embedder, query_vec, k=3),
            return_exceptions=True,
        )
        for lst in extra:
            if isinstance(lst, list):
                matches += lst
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
        "closest_matches": matches[:10],
        "density": density,
        "explanation": explanation,
        "gap_analysis": gap_analysis(req, matches, density, req.check_external),
    }


app.mount("/static", StaticFiles(directory=BASE / "static"), name="static")


@app.get("/")
def home():
    return FileResponse(BASE / "static" / "index.html")
