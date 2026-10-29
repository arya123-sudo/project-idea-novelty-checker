"""Retrieval evaluation — the deployment-realistic test.

For each labeled *similar* pair, the paraphrase (idea_a) is used as a query
against a corpus containing the original (idea_b) plus all department
projects as distractors. We measure whether the true original is retrieved
at rank 1 / in the top 3, and the mean reciprocal rank (MRR) — for both the
SBERT embeddings used by the system and a TF-IDF baseline. This mirrors
Dense Passage Retrieval-style evaluation: can the system find the original
work when the idea is reworded and hidden among other projects?

Run:  EMBEDDER_BACKEND=sbert python evaluation/retrieval_eval.py
"""
import csv, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from app.embedder import Embedder
from app.indexer import load_corpus

DATA = Path(__file__).resolve().parent.parent / "data" / "labeled_pairs.csv"


def ranks_for(sims_matrix):
    """sims_matrix[i] = similarities of query i to corpus; target is index 0."""
    ranks = []
    for row in sims_matrix:
        order = np.argsort(-row)
        ranks.append(int(np.where(order == 0)[0][0]) + 1)
    return ranks


def report(name, ranks):
    r = np.array(ranks)
    print(f"{name}: Recall@1 {np.mean(r == 1):.3f} | Recall@3 {np.mean(r <= 3):.3f} | "
          f"MRR {np.mean(1.0 / r):.3f} | queries {len(r)}")


def main():
    rows = [r for r in csv.DictReader(open(DATA, encoding="utf-8")) if int(r["label"]) == 1]
    dept = load_corpus()
    distractors = [f"{p['title']}. {p['abstract']}" for p in dept]
    queries = [r["idea_a"] for r in rows]
    targets = [r["idea_b"] for r in rows]
    corpus = targets[0:1]  # placeholder, rebuilt per query below

    # Success definition: the labeled pairs were written from real project
    # topics, so a department project can be the *same underlying idea* as the
    # pair target. We therefore score the rank of the first corpus item from
    # the target's idea family: the target itself (index 0) plus any
    # department project whose SBERT similarity to the target is >= 0.70.
    # Corpus = [target] + all department projects (no removal).
    emb = Embedder(corpus_texts=queries + targets + distractors)
    print(f"Embedder backend: {emb.backend}")
    qv = emb.encode(queries)
    tv_all = emb.encode(targets)
    dv = emb.encode(distractors)
    ranks = []
    for i in range(len(targets)):
        family = set([0]) | {j + 1 for j, s in enumerate((dv @ tv_all[i].T).ravel()) if s >= 0.70}
        sims = (np.vstack([tv_all[i], dv]) @ qv[i].T).ravel()
        order = np.argsort(-sims)
        ranks.append(min(int(np.where(order == f)[0][0]) + 1 for f in family))
    report("SBERT (system)", ranks)

    # TF-IDF baseline, same corpora and same SBERT-defined families
    from sklearn.feature_extraction.text import TfidfVectorizer
    vec = TfidfVectorizer().fit(queries + targets + distractors)
    qm = vec.transform(queries)
    tm_all = vec.transform(targets)
    dm = vec.transform(distractors)
    ranks = []
    for i in range(len(targets)):
        family = set([0]) | {j + 1 for j, s in enumerate((dv @ tv_all[i].T).ravel()) if s >= 0.70}
        target_sim = np.asarray((tm_all[i] @ qm[i].T).todense()).ravel()
        distr_sim = np.asarray((dm @ qm[i].T).todense()).ravel()
        all_sims = np.concatenate([target_sim, distr_sim])
        order = np.argsort(-all_sims)
        ranks.append(min(int(np.where(order == f)[0][0]) + 1 for f in family))
    report("TF-IDF (baseline)", ranks)


if __name__ == "__main__":
    main()
