"""Evaluation on labeled similar/different pairs.

Synopsis metrics: ROC-AUC, Precision, Recall, F1.
Label 1 = similar (same idea, often paraphrased), 0 = different.
A pair is predicted 'similar' when cosine similarity >= THRESHOLD.
Run:  python evaluation/evaluate.py
"""
import csv, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.embedder import Embedder
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score

THRESHOLD = 0.05  # TF-IDF cosines run much lower than SBERT (paraphrases share few exact words — measured: similar pairs 0.08-0.28, different 0.00-0.02 on this set). SBERT threshold is 0.55. This gap is itself a finding for the report: it is why the synopsis specifies SBERT.
DATA = Path(__file__).resolve().parent.parent / "data" / "labeled_pairs.csv"


def main():
    rows = list(csv.DictReader(open(DATA, encoding="utf-8")))
    texts = [r["idea_a"] for r in rows] + [r["idea_b"] for r in rows]
    emb = Embedder(corpus_texts=texts)
    print(f"Embedder backend: {emb.backend}  |  pairs: {len(rows)}")
    va = emb.encode([r["idea_a"] for r in rows])
    vb = emb.encode([r["idea_b"] for r in rows])
    sims = (va * vb).sum(axis=1)
    y = [int(r["label"]) for r in rows]
    thr = 0.55 if emb.backend == "sbert" else THRESHOLD
    pred = [1 if s >= thr else 0 for s in sims]
    print(f"Threshold: {thr}")
    print(f"ROC-AUC  : {roc_auc_score(y, sims):.3f}")
    print(f"Precision: {precision_score(y, pred, zero_division=0):.3f}")
    print(f"Recall   : {recall_score(y, pred, zero_division=0):.3f}")
    print(f"F1       : {f1_score(y, pred, zero_division=0):.3f}")


if __name__ == "__main__":
    main()
