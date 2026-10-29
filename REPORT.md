# AI-Based Project Idea Novelty Checker

**Mini Project Report — Draft**

Department of Artificial Intelligence and Machine Learning
Acharya Institute of Technology, Bengaluru

| | |
|---|---|
| Arya R Yaligar | 1AY24AI012 |
| Bhargav Raj | 1AY24AI019 |
| D Bargav | 1AY24AI028 |
| G Vajresh Shenoy | 1AY24AI034 |

**Guide:** Mr. Praveen Arokiaraj, Assistant Professor, Dept. of AI & ML

---

## 1. Introduction / Problem Statement

Every semester, student project ideas are rejected — often late, after weeks of work — with the reason "already done". Students have no easy way to check a new idea against three scattered sources: past department projects, research papers on arXiv, and code repositories on GitHub. Keyword search, the only tool normally available, misses reworded or paraphrased versions of the same idea, because a duplicate idea rarely reuses the same words.

This project builds an automated, objective way to measure the novelty of a project idea **early**: a student submits a title and abstract, and the system returns a novelty score with the closest existing matches, so the student (and guide) can see *why* an idea looks novel or duplicate before committing to it.

## 2. Objectives

1. Build a semantic similarity engine that compares a new idea with past department projects.
2. Extend the check to public arXiv papers and GitHub repositories.
3. Output a novelty score with closest matches, validated on labeled similar/different pairs using ROC-AUC, Precision, Recall and F1.

## 3. Literature Survey (summary)

| Work | Relevance |
|---|---|
| Reimers & Gurevych, 2019 — Sentence-BERT | Siamese BERT networks producing sentence embeddings comparable with cosine similarity — the embedding method used here. |
| Johnson et al., 2019 — FAISS | Billion-scale similarity search; used here for nearest-neighbour retrieval over the project corpus. |
| Karpukhin et al., 2020 — Dense Passage Retrieval | Shows dense embeddings outperform keyword (BM25-style) retrieval for semantic matching. |
| Shibayama et al., 2021 | Embedding-based measurement of scientific novelty — novelty as distance in embedding space, the principle behind our score. |
| Liu et al., 2025 | LLM-based novelty assessment — qualitative verdicts; our work differs by giving a deterministic, evaluated pipeline. |

**Differentiation.** Plagiarism checkers (e.g. Turnitin) match exact text, not ideas. Patent tools work in a different domain. No public tool covers a department's own past-project archive — the corpus where most "already done" rejections actually come from. Our defensible claim is the combination: **department corpus + arXiv + GitHub, with a calibrated, evaluated similarity pipeline**. We do not claim semantic novelty checking itself is new.

## 4. Proposed Methodology

```
New Idea (Title + Abstract)
        ↓
Sentence-Transformer Embedding (all-MiniLM-L6-v2, 384-dim)
        ↓
FAISS Search: Department corpus + live arXiv + GitHub (re-ranked by embedding similarity)
        ↓
Cosine Similarity against closest matches
        ↓
Novelty Score = 100 × (1 − max similarity)  →  Verdict + Closest Matches
```

Verdict bands: **≥ 70** Highly Novel · **45–69** Moderately Novel (differentiate clearly) · **< 45** Low Novelty (very similar work exists).

Backend: FastAPI. Frontend: single-page web UI for students/faculty. If SBERT is unavailable the embedding layer falls back to TF-IDF so the demo still runs — Section 6 shows why SBERT is nevertheless necessary.

## 5. Implementation

- `app/embedder.py` — SBERT embedder (TF-IDF fallback), L2-normalised vectors so inner product = cosine similarity.
- `app/indexer.py` — FAISS `IndexFlatIP` over the department corpus (36 sample projects in the demo dataset; to be replaced by the real department archive).
- `app/external.py` — live arXiv API and GitHub Search API calls; keyword candidates are **re-ranked by embedding similarity**, which is what surfaces paraphrases that keyword ranking alone buries.
- `app/main.py` — `POST /api/check` returning score, verdict and top matches across all three sources.
- `evaluation/evaluate.py` — metrics on labeled pairs.

## 6. Results and Evaluation

Evaluation set: **60 labeled pairs** (30 similar — same idea, mostly paraphrased; 30 different — unrelated ideas).

<!-- RESULTS:SBERT — filled from the actual SBERT run -->
_(To be filled from the SBERT evaluation run.)_

<!-- RESULTS:TFIDF -->
TF-IDF fallback (20-pair pilot set): ROC-AUC 1.000, Precision 1.000, Recall 1.000, F1 1.000 — but with a revealing weakness: paraphrase pairs scored only **0.08–0.28** cosine similarity (vs 0.00–0.02 for unrelated pairs). TF-IDF separates the two classes on this small set only because unrelated pairs share almost no words; its margins are thin and it cannot recognise a paraphrase written in completely different vocabulary. This is the empirical justification for SBERT in the methodology.

Worked examples (live system):
- "Face Recognition Attendance" (a known past project, reworded) → **Novelty 45.3**, closest match: the department's own *Smart Attendance System using Face Recognition*.
- A genuinely new idea (beehive health monitoring via acoustic sensors) → high novelty, no close department match.

## 7. Limitations

- The demo department corpus is a 36-project sample; novelty verdicts are only as complete as the real archive fed into it.
- External search depends on the arXiv/GitHub keyword pre-filter; a paper using entirely different terminology may not enter the candidate set.
- The novelty score is a decision-support signal, not a final verdict — the guide's judgement remains final.

## 8. Future Scope

- Replace the sample corpus with the full department archive (all batches).
- LLM-based query expansion to improve external recall.
- Faculty dashboard with batch screening of a whole class's proposals.
- Cross-encoder re-ranking for higher precision on borderline pairs.

## 9. Conclusion

The system demonstrates that idea-level semantic matching catches duplicates that keyword search misses, gives students an early, objective novelty signal with evidence (the closest matches), and is evaluated with standard classification metrics rather than anecdotal examples.

## References

1. Reimers, N. & Gurevych, I. (2019). Sentence-BERT. EMNLP-IJCNLP.
2. Johnson, J., Douze, M. & Jégou, H. (2019). Billion-scale similarity search with GPUs (FAISS). IEEE TBD.
3. Karpukhin, V. et al. (2020). Dense Passage Retrieval for Open-Domain QA. EMNLP.
4. Shibayama, S. et al. (2021). Measuring novelty in science with word embedding. PLoS ONE / Scientometrics line of work.
5. Liu et al. (2025). LLM-based research novelty detection.
