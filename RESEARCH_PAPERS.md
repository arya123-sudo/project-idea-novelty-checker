# Research Papers — AI-Based Project Idea Novelty Checker

Literature pack for the mini project. Every paper below was verified against its publisher/arXiv page. Organised the way the report's literature survey is: (A) the foundations our stack is built on, (B) the science of measuring novelty, (C) scientific-document embeddings, (D) the modern automated-novelty-assessment literature — including the papers that justify *our* design choices over the obvious alternative (asking an LLM).

---

## A. Foundations of our stack

### A1. Sentence-BERT: Sentence Embeddings using Siamese BERT Networks
**Reimers & Gurevych, 2019 — EMNLP-IJCNLP.** https://arxiv.org/abs/1908.10084
Derives sentence embeddings from BERT with siamese/triplet networks so that semantically similar sentences are close under cosine similarity. Finding a similar sentence pair drops from ~65 hours (BERT cross-encoding) to ~5 seconds, with better accuracy.
**Use in our project:** this is our embedding method (via `all-MiniLM-L6-v2`, a Sentence-Transformers model). It is why a paraphrased duplicate scores high similarity even with no shared keywords — the exact failure mode of keyword search that our problem statement names.

### A2. Billion-scale similarity search with GPUs (FAISS)
**Johnson, Douze & Jégou, 2019 — IEEE Transactions on Big Data.** https://arxiv.org/abs/1702.08734
Introduces the FAISS approach to exact and approximate k-nearest-neighbour search over dense vectors at very large scale, with GPU implementations.
**Use in our project:** our retrieval layer (`IndexFlatIP` over L2-normalised vectors = cosine search). FAISS is what lets the corpus grow from 36 sample projects to a full department archive without changing the design.

### A3. Dense Passage Retrieval for Open-Domain Question Answering
**Karpukhin et al., 2020 — EMNLP.** https://arxiv.org/abs/2004.04906
Shows dense dual-encoder embeddings decisively outperform strong keyword retrieval (BM25) when the query and the answer are worded differently.
**Use in our project:** published evidence for the core design bet — semantic retrieval over keyword retrieval — which our own SBERT-vs-TF-IDF experiment reproduces (Section 6 of the report: TF-IDF F1 collapses 0.868 → 0.125 as threshold rises; SBERT holds 0.966).

## B. The science of measuring novelty

### B1. Measuring novelty in science with word embedding
**Shibayama, Yin & Matsumoto, 2021 — PLoS ONE 16(7): e0254034.** https://doi.org/10.1371/journal.pone.0254034
Defines a paper's novelty through semantic distances in embedding space (distance between the combinations of work it builds on). Validated three ways: agreement with an established bibliometric distance, correlation with authors' self-reported novelty, and prediction of future citations.
**Use in our project:** the theoretical licence for our score — *novelty = distance from prior work in embedding space* — with a validation template (compare against human judgements) we mirror with our labeled-pair evaluation.

### B2. Atypical Combinations and Scientific Impact
**Uzzi, Mukherjee, Stringer & Jones, 2013 — Science 342(6157).**
The classic bibliometric novelty result: papers that combine prior work in atypical ways are roughly twice as likely to be highly cited, across 17.9M papers. Novelty is measurable from how unusual a work's combination of predecessors is.
**Use in our project:** background/motivation — novelty is not a vibe; it has been quantitatively measured against prior-work corpora for a decade. Our system brings the same idea down from journals to student project proposals.

## C. Scientific-document embeddings

### C1. SPECTER: Document-level Representation Learning using Citation-informed Transformers
**Cohan, Feldman, Beltagy, Downey & Weld, 2020 — ACL.** https://arxiv.org/abs/2004.07180
Trains a transformer on title+abstract with a citation-based triplet objective, so papers that cite/are related to each other embed closely. Outperforms general embeddings on scientific-document tasks (SciDocs benchmark). Successor: **SPECTER2** (Singh et al., 2023), trained on 10× more data with task adapters, evaluated on SciRepEval.
**Use in our project:** the strongest named future upgrade. MiniLM is general-purpose; SPECTER2 embeds *papers* specifically, so swapping it in for the arXiv side of our pipeline is a principled improvement, not a guess.

## D. Modern automated novelty assessment (2024–2025)

### D1. Can LLMs Generate Novel Research Ideas? A Large-Scale Human Study with 100+ NLP Researchers
**Si, Yang & Hashimoto, 2024 — ICLR 2025.** https://arxiv.org/abs/2409.04109
100+ expert researchers wrote and blind-reviewed human vs LLM ideas: LLM ideas were judged *more novel* but slightly less feasible. Two findings matter for us: (1) at scale, generation saturates — of ~4,000 generated ideas per topic only ~5% were non-duplicate after deduplication; (2) LLM self-evaluation of novelty fails.
**Use in our project:** motivation and scale argument. If even AI ideation tools produce 95% duplicates, an independent, deterministic novelty checker is a necessary piece of infrastructure — and it must not be the LLM judging itself.

### D2. Enabling AI Scientists to Recognize Innovation: A Domain-Agnostic Algorithm for Assessing Novelty (Relative Neighbor Density)
**Wang, Cui, Jiang & Yan, 2025.** https://arxiv.org/abs/2503.01508
Proposes RND: judge an idea's novelty by the *density of its embedding neighbourhood* relative to its neighbours' densities, instead of a single nearest-neighbour distance. AUROC 0.820 (CS) / 0.765 (biomedical), and it holds up cross-domain (0.795 vs 0.597 for LLM judges/metrics). The paper also documents why LLM-as-judge is unreliable for novelty: identical ideas phrased differently get contradictory ratings.
**Use in our project:** (1) independent support for choosing a deterministic embedding pipeline over an LLM judge; (2) our named algorithmic future-scope — an RND-style density score alongside the max-similarity score.

### D3. NoveltyRank: A Retrieval-Augmented Framework for Conceptual Novelty Estimation in AI Research
**2025.** https://arxiv.org/abs/2512.14738
The closest published system to ours: retrieve related work, embed title+abstract with **SPECTER2**, and estimate conceptual novelty against a 60,294-paper corpus (arXiv + ICLR 2017–2025), using venue acceptance as a proxy novelty label with a chronological train/test split to prevent leakage.
**Use in our project:** proof that the retrieve-embed-compare architecture is the field's current shape — and a methodological upgrade path (proxy labels + time-based splits) if our labeled set grows. Note the honest difference: they assess *research papers*; we assess *student project ideas against an institutional archive*.

### D4. What Is Novel? A Knowledge-Driven Framework for Bias-Aware Literature Originality Evaluation
**2025/2026.** https://arxiv.org/abs/2602.06054
Fine-tunes an LLM on human novelty judgements, combined with structured contribution extraction and graph-based retrieval over related work, producing calibrated novelty scores with grounded justifications; reports reliable **idea-level plagiarism detection**.
**Use in our project:** the hybrid camp — retrieval *plus* a learned judge. Supports our design decision to always show the closest matches (grounding) rather than a bare score, and marks the direction a larger version of this project could take.

### D5. All That Glitters is Not Novel: Plagiarism in AI-Generated Research
**2025.** https://arxiv.org/abs/2502.16487
Shows several showcased LLM-generated research outputs were plagiarised or substantially similar to existing work — and that built-in LLM-based detectors missed it, while embedding/retrieval-based methods (Semantic Scholar retrieval, OpenScholar-style embeddings) are the ones evaluated for catching it.
**Use in our project:** the cautionary tale for the viva — surface-level originality fools LLM judges; similarity against an actual corpus is the check that works.

---

## What the research says, in one page (synthesis for the report/viva)

1. **Novelty is legitimately measurable as embedding distance.** Shibayama et al. validated it against human self-reports and citations; Uzzi et al. showed predecessor-combination novelty predicts impact. Our score is an application of this principle, not an invention of it.
2. **There are three camps in automated novelty assessment today:** LLM-as-judge (flexible, but unstable — rephrasing the same idea flips verdicts, per Wang et al. and Si et al.), embedding-distance (deterministic, reproducible, evaluated — our camp), and hybrids (retrieval + fine-tuned judge, e.g. *What Is Novel?*). For a student-facing screening tool that must give the *same verdict for the same idea*, determinism is a feature.
3. **The duplicate problem is real and large.** Si et al.'s ~95% duplication rate in generated ideas, and the plagiarism findings of *All That Glitters*, show "already done" is the default state of new-looking ideas — exactly the late-rejection pain in our problem statement.
4. **Retrieval quality is the ceiling.** NoveltyRank and *What Is Novel?* both stress that the comparison corpus decides everything — which is why our project's distinctive asset is the **department archive**, the corpus where student rejections actually come from, and why replacing the 36-project sample with the real archive is the highest-value next step.
5. **Named, paper-backed future scope:** SPECTER2 embeddings for the paper corpus (Cohan et al.; Singh et al.), RND-style neighbourhood-density scoring (Wang et al.), proxy-label evaluation with chronological splits (NoveltyRank), and hybrid LLM explanation *on top of* — never instead of — the deterministic score.

## Reference list (report-ready)

1. Reimers, N. & Gurevych, I. (2019). Sentence-BERT: Sentence Embeddings using Siamese BERT Networks. *EMNLP-IJCNLP 2019*. arXiv:1908.10084.
2. Johnson, J., Douze, M. & Jégou, H. (2019). Billion-scale similarity search with GPUs. *IEEE Transactions on Big Data*. arXiv:1702.08734.
3. Karpukhin, V. et al. (2020). Dense Passage Retrieval for Open-Domain Question Answering. *EMNLP 2020*. arXiv:2004.04906.
4. Shibayama, S., Yin, D. & Matsumoto, K. (2021). Measuring novelty in science with word embedding. *PLoS ONE*, 16(7), e0254034.
5. Uzzi, B., Mukherjee, S., Stringer, M. & Jones, B. (2013). Atypical combinations and scientific impact. *Science*, 342(6157), 468–472.
6. Cohan, A., Feldman, S., Beltagy, I., Downey, D. & Weld, D. S. (2020). SPECTER: Document-level Representation Learning using Citation-informed Transformers. *ACL 2020*. arXiv:2004.07180.
7. Singh, A. et al. (2023). SciRepEval: A Multi-Format Benchmark for Scientific Document Representations (SPECTER2). arXiv:2211.13308.
8. Si, C., Yang, D. & Hashimoto, T. (2025). Can LLMs Generate Novel Research Ideas? A Large-Scale Human Study with 100+ NLP Researchers. *ICLR 2025*. arXiv:2409.04109.
9. Wang, Y., Cui, M., Jiang, A. & Yan, J. (2025). Enabling AI Scientists to Recognize Innovation: A Domain-Agnostic Algorithm for Assessing Novelty. arXiv:2503.01508.
10. NoveltyRank: A Retrieval-Augmented Framework for Conceptual Novelty Estimation in AI Research (2025). arXiv:2512.14738.
11. What Is Novel? A Knowledge-Driven Framework for Bias-Aware Literature Originality Evaluation (2026). arXiv:2602.06054.
12. All That Glitters is Not Novel: Plagiarism in AI-Generated Research (2025). arXiv:2502.16487.
