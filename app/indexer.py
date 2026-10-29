"""FAISS index over the department corpus."""
import json
from pathlib import Path
import numpy as np

try:
    import faiss
    HAS_FAISS = True
except Exception:
    HAS_FAISS = False

DATA = Path(__file__).resolve().parent.parent / "data" / "dept_projects.json"


def load_corpus():
    return json.loads(DATA.read_text(encoding="utf-8"))


class DeptIndex:
    def __init__(self, embedder):
        self.corpus = load_corpus()
        texts = [f"{p['title']}. {p['abstract']}" for p in self.corpus]
        self.vectors = embedder.encode(texts)
        self.index = None
        if HAS_FAISS:
            self.index = faiss.IndexFlatIP(self.vectors.shape[1])
            self.index.add(self.vectors)

    def search(self, query_vec: np.ndarray, k: int = 5):
        k = min(k, len(self.corpus))
        if self.index is not None:
            scores, idx = self.index.search(query_vec, k)
            scores, idx = scores[0], idx[0]
        else:  # numpy fallback = same cosine maths
            sims = (self.vectors @ query_vec.T).ravel()
            idx = np.argsort(-sims)[:k]
            scores = sims[idx]
        out = []
        for s, i in zip(scores, idx):
            p = self.corpus[int(i)]
            out.append({
                "source": "Department",
                "title": p["title"], "abstract": p["abstract"],
                "year": p.get("year"), "url": None,
                "similarity": round(float(s), 4),
            })
        return out
