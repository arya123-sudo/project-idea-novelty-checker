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
| Shibayama et al., 2021 | Embedding-based measurement of scientific novelty — novelty as distance in embedding space, validated against self-reported novelty and future citations; the principle behind our score. |
| Uzzi et al., 2013 — Science | Atypical combinations of prior work predict high impact across 17.9M papers — novelty is quantitatively measurable against a prior-work corpus. |
| Cohan et al., 2020 — SPECTER (+ SPECTER2, 2023) | Scientific-document embeddings from title+abstract trained with citation signals; the named upgrade for our arXiv side. |
| Si, Yang & Hashimoto, 2025 — ICLR | 100+ researchers, human vs LLM ideas: LLM ideas judged more novel, but ~95% of generated ideas were duplicates after deduplication, and LLM self-evaluation of novelty fails — motivation for an independent checker. |
| Wang et al., 2025 — Relative Neighbor Density | Embedding-neighbourhood-density novelty scoring (AUROC 0.820 CS); documents that LLM-as-judge novelty ratings flip when the same idea is rephrased — support for a deterministic pipeline. |
| NoveltyRank, 2025 | Closest published system: retrieval + SPECTER2 embeddings + novelty estimation over 60k papers; venue acceptance as proxy label, chronological splits. Differs from ours in target: research papers vs student project ideas against an institutional archive. |
| What Is Novel?, 2026; All That Glitters is Not Novel, 2025 | Hybrid retrieval + fine-tuned-judge novelty evaluation with idea-level plagiarism detection; and evidence that LLM-generated research often duplicates prior work in ways LLM detectors miss but corpus similarity catches. |

Full annotated pack: `RESEARCH_PAPERS.md` in this repository.

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
- `app/indexer.py` also computes **neighbourhood density**: mean similarity to the 5 nearest department projects (topic *crowdedness*, inspired by Wang et al. 2025), reported alongside the score.
- `app/main.py` generates an **evidence explanation** for every verdict — closest prior work vs the 0.55 duplicate threshold, crowdedness reading, and the distinctive shared concepts per match — so the verdict can be argued with, not just accepted.
- `evaluation/evaluate.py` — metrics on labeled pairs; `evaluation/retrieval_eval.py` — DPR-style retrieval evaluation.

## 6. Results and Evaluation

Evaluation set: **60 labeled pairs** (30 similar — same idea, mostly paraphrased; 30 different — unrelated ideas).

**SBERT results (all-MiniLM-L6-v2, threshold 0.55):**

| Metric | SBERT | TF-IDF (same 60 pairs) |
|---|---|---|
| ROC-AUC | **1.000** | 0.933 |
| Precision | **1.000** | 1.000 |
| Recall | **0.933** | 0.767 |
| F1 | **0.966** | 0.868 |

Similarity distributions explain the gap. SBERT places similar pairs at **0.498–0.885** (mean 0.680) and different pairs at **−0.064–0.280** (mean 0.056) — a wide, safe separation margin. TF-IDF compresses similar pairs to **0.000–0.260**: a paraphrase written in different words is nearly invisible to it, and its F1 collapses from 0.868 to 0.125 as the threshold rises from 0.05 to 0.15. This is the empirical justification for SBERT in the methodology: **the novelty signal lives in meaning, not shared words.**

At the operating threshold 0.55, SBERT missed 2 of 30 similar pairs (college-enquiry chatbot paraphrase, 0.512; phishing-detection paraphrase, 0.498). A threshold scan shows F1 = 1.000 at 0.30 on this set; we keep 0.55 as the operating point because wrongly branding a student's idea a "duplicate" is the costlier error, and borderline cases are exactly what the closest-matches display is for — the guide sees the evidence, not just the number.

**Retrieval evaluation (deployment-realistic).** Each paraphrased idea from the 30 similar pairs was queried against the department corpus plus its original; we measured the rank of the first item from the same idea family:

| Method | Recall@1 | Recall@3 | MRR |
|---|---|---|---|
| SBERT (this system) | 0.433 | **1.000** | **0.711** |
| TF-IDF (baseline) | 0.500 | 0.867 | 0.697 |

TF-IDF occasionally wins rank 1 on exact title wording, but it loses the original entirely from the top 3 in 4 of 30 cases. For a novelty checker the missed duplicate is the costly error — SBERT never misses: a same-idea item is in the top 3 for every reworded query, at 0.85–0.88 similarity against 0.22–0.39 for the nearest unrelated project.

**Crowdedness in practice.** The density lens separates cases the headline score cannot: crop-disease scores local crowding 0.63 (a saturated area — differentiation is mandatory), while beehive acoustics scores 0.30 (open area). Both readings are shown to the user with the verdict.

**Deployment:** the system is deployed publicly on Render (free tier) at https://project-idea-novelty-checker.onrender.com, running the same MiniLM model in ONNX form (`sbert-onnx` backend — no torch, fits 512 MB RAM). The live deployment reproduces the local results exactly: the crop-disease check below returns novelty 8.7 with max similarity 0.9127 on both the local torch backend and the deployed ONNX backend, and the full three-source check completes live in ~16 s.

Worked examples (live system, all three sources):
- "Crop Disease Detection using Deep Learning" (reworded abstract) → **Novelty 8.7 / Low Novelty**. Closest matches: Department *Crop Disease Detection using Deep Learning* (0.913), Department *Plant Leaf Disease Classification with CNN* (0.840), GitHub `Shubham-Jain-09/Crop-Disease-Detection` (0.822), arXiv *Detecting Multiple Diseases in Multiple Crops Using Deep Learning* (0.807).
- A genuinely new idea (beehive health monitoring via acoustic sensors) → **Novelty 56.6 / Moderately Novel**, closest department match only 0.434 — no close prior work in any source.

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

1. Reimers, N. & Gurevych, I. (2019). Sentence-BERT: Sentence Embeddings using Siamese BERT Networks. EMNLP-IJCNLP. arXiv:1908.10084.
2. Johnson, J., Douze, M. & Jégou, H. (2019). Billion-scale similarity search with GPUs. IEEE Transactions on Big Data. arXiv:1702.08734.
3. Karpukhin, V. et al. (2020). Dense Passage Retrieval for Open-Domain Question Answering. EMNLP. arXiv:2004.04906.
4. Shibayama, S., Yin, D. & Matsumoto, K. (2021). Measuring novelty in science with word embedding. PLoS ONE, 16(7), e0254034.
5. Uzzi, B., Mukherjee, S., Stringer, M. & Jones, B. (2013). Atypical combinations and scientific impact. Science, 342(6157), 468–472.
6. Cohan, A. et al. (2020). SPECTER: Document-level Representation Learning using Citation-informed Transformers. ACL. arXiv:2004.07180.
7. Singh, A. et al. (2023). SciRepEval: A Multi-Format Benchmark for Scientific Document Representations (SPECTER2). arXiv:2211.13308.
8. Si, C., Yang, D. & Hashimoto, T. (2025). Can LLMs Generate Novel Research Ideas? A Large-Scale Human Study with 100+ NLP Researchers. ICLR. arXiv:2409.04109.
9. Wang, Y., Cui, M., Jiang, A. & Yan, J. (2025). Enabling AI Scientists to Recognize Innovation: A Domain-Agnostic Algorithm for Assessing Novelty. arXiv:2503.01508.
10. NoveltyRank: A Retrieval-Augmented Framework for Conceptual Novelty Estimation in AI Research (2025). arXiv:2512.14738.
11. What Is Novel? A Knowledge-Driven Framework for Bias-Aware Literature Originality Evaluation (2026). arXiv:2602.06054.
12. All That Glitters is Not Novel: Plagiarism in AI-Generated Research (2025). arXiv:2502.16487.
