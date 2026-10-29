"""Live external search: arXiv + GitHub.

These sources have no embedding index on our side, so candidates returned by
their keyword search are re-ranked by embedding cosine similarity against the
query — that is what catches paraphrases keyword ranking alone would bury.
Failures (offline, rate-limit) return [] instead of breaking the check.
"""
import re
import httpx

TIMEOUT = 12.0


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
        async with httpx.AsyncClient(timeout=TIMEOUT, follow_redirects=True, headers=HEADERS) as c:
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
        async with httpx.AsyncClient(timeout=TIMEOUT, follow_redirects=True, headers={"Accept": "application/vnd.github+json", **HEADERS}) as c:
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


def _rerank(cands, embedder, query_vec, k):
    if not cands:
        return []
    vecs = embedder.encode([f"{c['title']}. {c['abstract']}" for c in cands])
    sims = (vecs @ query_vec.T).ravel()
    order = sims.argsort()[::-1][:k]
    for i in order:
        cands[i]["similarity"] = round(float(sims[i]), 4)
    return [cands[i] for i in order]
