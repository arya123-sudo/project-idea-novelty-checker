"""Live external search: arXiv + GitHub.

These sources have no embedding index on our side, so candidates returned by
their keyword search are re-ranked by embedding cosine similarity against the
query — that is what catches paraphrases keyword ranking alone would bury.
Failures (offline, rate-limit) return [] instead of breaking the check.
"""
import os
import re
import httpx

TIMEOUT = 12.0


def _client(**kw):
    # httpx crashes parsing NO_PROXY values that contain bracketed IPv6
    # entries like [::1] (some sandboxes/CI set them). Strip only those
    # entries, in-process; on a normal machine this is a no-op.
    for var in ("no_proxy", "NO_PROXY"):
        v = os.environ.get(var)
        if v and "[" in v:
            os.environ[var] = ",".join(p for p in v.split(",") if "[" not in p)
    return httpx.AsyncClient(timeout=TIMEOUT, follow_redirects=True, trust_env=True, **kw)


def _keywords(title: str, abstract: str, n=8):
    words = re.findall(r"[a-zA-Z]{4,}", f"{title} {abstract}".lower())
    stop = {"using", "based", "system", "project", "approach", "model", "with", "from", "that", "this"}
    seen, out = set(), []
    for w in words:
        if w not in stop and w not in seen:
            seen.add(w); out.append(w)
        if len(out) >= n:
            break
    return out


HEADERS = {"User-Agent": "NoveltyChecker/1.0 (student mini-project)"}


async def search_arxiv(title, abstract, embedder, query_vec, k=3):
    words = _keywords(title, abstract, 5)
    if not words:
        return []
    q = "+AND+".join(f"all:{w}" for w in words[:3])
    url = f"https://export.arxiv.org/api/query?search_query={q}&start=0&max_results=8"
    try:
        async with _client(headers=HEADERS) as c:
            r = await c.get(url)
            r.raise_for_status()
        import xml.etree.ElementTree as ET
        ns = {"a": "http://www.w3.org/2005/Atom"}
        root = ET.fromstring(r.text)
        cands = []
        for e in root.findall("a:entry", ns):
            t = " ".join(e.findtext("a:title", "", ns).split())
            s = " ".join(e.findtext("a:summary", "", ns).split())
            link = e.findtext("a:id", "", ns)
            if t:
                cands.append({"source": "arXiv", "title": t, "abstract": s[:400], "year": None, "url": link})
    except Exception:
        return []
    return _rerank(cands, embedder, query_vec, k)


async def search_github(title, abstract, embedder, query_vec, k=3):
    q = "+".join(_keywords(title, abstract, 5))
    if not q:
        return []
    try:
        async with _client(headers={"Accept": "application/vnd.github+json", **HEADERS}) as c:
            r = await c.get(f"https://api.github.com/search/repositories?q={q}&per_page=8")
            r.raise_for_status()
        cands = [{
            "source": "GitHub", "title": it["full_name"],
            "abstract": (it.get("description") or "")[:400],
            "year": (it.get("created_at") or "")[:4] or None,
            "url": it["html_url"],
        } for it in r.json().get("items", [])]
    except Exception:
        return []
    return _rerank(cands, embedder, query_vec, k)


async def search_semantic_scholar(title, abstract, embedder, query_vec, k=3):
    """Semantic Scholar indexes 200M+ papers including IEEE and Elsevier
    content, so this one free API extends coverage to those publishers
    without their key-gated APIs."""
    q = " ".join(_keywords(title, abstract, 6))
    if not q:
        return []
    try:
        async with _client(headers=HEADERS) as c:
            r = await c.get(
                "https://api.semanticscholar.org/graph/v1/paper/search",
                params={"query": q, "limit": 8,
                        "fields": "title,abstract,year,venue,url"},
            )
            r.raise_for_status()
        cands = [{
            "source": "Semantic Scholar", "title": p["title"],
            "abstract": (p.get("abstract") or "")[:400],
            "year": p.get("year"), "url": p.get("url"),
            "venue": p.get("venue") or None,
        } for p in r.json().get("data", []) if p.get("title")]
    except Exception:
        return []
    return _rerank(cands, embedder, query_vec, k)


async def search_openalex(title, abstract, embedder, query_vec, k=3):
    """OpenAlex (250M+ works, fully open) — abstracts arrive as an inverted
    index and are reconstructed; venue names surface IEEE/Elsevier journals."""
    q = " ".join(_keywords(title, abstract, 6))
    if not q:
        return []
    try:
        async with _client(headers=HEADERS) as c:
            r = await c.get("https://api.openalex.org/works",
                            params={"search": q, "per-page": 8})
            r.raise_for_status()
        cands = []
        for w in r.json().get("results", []):
            inv = w.get("abstract_inverted_index") or {}
            words = sorted(((pos, word) for word, poss in inv.items() for pos in poss))
            text = " ".join(word for _, word in words)[:400]
            src = ((w.get("primary_location") or {}).get("source") or {}).get("display_name")
            cands.append({
                "source": "OpenAlex", "title": w.get("title") or "",
                "abstract": text, "year": w.get("publication_year"),
                "url": w.get("doi") or w.get("id"), "venue": src,
            })
        cands = [c for c in cands if c["title"]]
    except Exception:
        return []
    return _rerank(cands, embedder, query_vec, k)


def _rerank(cands, embedder, query_vec, k):
    if not cands:
        return []
    vecs = embedder.encode([f"{c['title']}. {c['abstract']}" for c in cands])
    sims = (vecs @ query_vec.T).ravel()
    order = sims.argsort()[::-1][:k]
    for i in order:
        cands[i]["similarity"] = round(float(sims[i]), 4)
    return [cands[i] for i in order]
