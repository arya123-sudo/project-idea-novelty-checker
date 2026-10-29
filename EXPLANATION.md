# Explaining This Project — Pitch, Techniques, Usefulness, and Tough Questions

*For the team: read this before any presentation or viva. Everything claimed here is backed by the system, the evaluation, or a cited paper — nothing is decoration.*

---

## 1. The 30-second pitch

"Every year, students lose weeks of work when their project idea is rejected at review — not because it was bad, but because it was **already done**, sometimes in their own department the previous year. Keyword search cannot save them, because the duplicate is always *reworded*. Our system checks an idea the way a human expert would — by **meaning**. It converts the idea into a semantic fingerprint, searches three corpora at once — the department archive, arXiv research papers, and GitHub implementations — and returns a novelty score with the closest prior work as evidence. It is deterministic, evaluated, and live."

## 2. The 2-minute explanation

1. **The problem is information, not effort.** A student cannot check what they cannot find. Department archives sit in filing cabinets and shared drives; arXiv has 2M+ papers; GitHub has millions of repos. Manual checking does not scale, and keyword search fails exactly where it matters — a paraphrased duplicate shares *meaning*, not words.
2. **The core technique is semantic embedding.** A transformer model (Sentence-BERT, MiniLM) maps title+abstract to a 384-dimensional vector positioned by meaning. "Camera identifies students and marks attendance" and "automatic attendance where a camera recognises faces" land almost on top of each other in that space — despite sharing almost no vocabulary.
3. **Novelty = distance from prior work.** This is not our invention: Shibayama et al. (2021) validated embedding-distance novelty against researchers' own novelty judgements and future citations. Our score is 100 × (1 − closest similarity).
4. **Retrieval is FAISS** — the same class of index used at billion scale (Johnson et al.) — so the corpus can grow from our 36-project sample to the full department archive and beyond without redesign.
5. **We check three worlds at once.** An idea can be new to the department but old on GitHub, or unpublished as a project but published as a paper. Department + arXiv + GitHub together approximate "has this been done anywhere a reviewer would look."
6. **We added a second lens: crowdedness.** Beyond the single closest match, a neighbourhood-density analysis (inspired by Wang et al. 2025) measures how packed the topic area is. An idea can have no exact duplicate yet sit in a saturated area — the system says so.
7. **Every verdict comes with evidence.** Closest matches, similarity percentages, and the shared concepts behind each match. A score a student can argue with; a verdict a guide can verify. That is deliberate — the research (What Is Novel?, 2026) shows grounded, retrieval-based justification is what makes automated novelty assessment trustworthy.
8. **It is evaluated like a research system, not a demo.** ROC-AUC 1.000 and F1 0.966 on 60 labeled similar/different pairs; a retrieval test where reworded ideas must find their original among distractors; and a baseline comparison showing exactly where keyword methods collapse.

## 3. Technique-by-technique justification (the "why this, not that" table)

| Choice | Alternative rejected | Why |
|---|---|---|
| Sentence-BERT embeddings | TF-IDF / keyword search | Paraphrases share meaning, not words. Measured: TF-IDF's F1 collapses 0.868 → 0.125 as threshold tightens; SBERT holds 0.966. DPR (Karpukhin 2020) showed the same gap at research scale. |
| Deterministic embedding score | "Ask an LLM if it's novel" | Wang et al. 2025: LLM novelty ratings flip when the same idea is rephrased. Si et al. 2025: LLM self-evaluation of novelty fails. A screening tool must give the same verdict for the same idea, every time — and ours runs free, offline-capable, no API key. |
| Max-similarity drives the score | Average similarity | One near-duplicate is enough to sink an idea. Averaging would let a copy hide among unrelated projects. Density is reported separately instead of diluting the main score. |
| FAISS exact inner-product index | Brute-force loops / a database | Exact cosine via IndexFlatIP today; the same library scales to millions of vectors with approximate indexes when the corpus grows. |
| Threshold 0.55 as duplicate line | A round, guessed number | Chosen from the evaluation: it is the operating point balancing the two errors, where falsely accusing a novel idea (precision cost) is treated as worse than missing a borderline duplicate — because the system shows closest matches as evidence either way. |
| Title + abstract as input | Title only / full report | Title-only is too thin to disambiguate (Shibayama used titles only for references; SPECTER showed title+abstract is the informative unit for papers). Full reports don't exist at ideation time — the abstract is what exists when the decision is made. |
| ONNX MiniLM for deployment | Full PyTorch service | Same model, same scores (verified: live deployment reproduces local results exactly, 0.9127 max similarity on the test case), at a fraction of the memory — engineering the system to run on free infrastructure is part of the work. |

## 4. Usefulness — who gains what

- **Students:** a 30-second check *before* the synopsis is finalised — while changing the idea is still free. The closest matches double as a reading list and as the differentiation paragraph of their synopsis ("unlike X, we do Y").
- **Guides/faculty:** a consistent first screen across all teams instead of memory-based judgement ("I think someone did this in 2022…"). Borderline calls still go to the human — the system supplies the evidence, the guide decides.
- **The department:** every checked idea can grow the archive; the corpus gets more valuable each year, and repeated-topic statistics show which areas are saturated.
- **Hackathon teams (same students, different hat):** a same-day originality check before building — our own team's experience is that "already exists" is the most common silent killer of hackathon ideas too.
- **The cost asymmetry it fixes:** a rejected-at-review idea costs weeks and morale; a check costs seconds. Even catching one duplicate per batch pays for the system.

## 5. Anticipated tough questions — and the strong answers

**"Isn't this just plagiarism checking?"**
Plagiarism checkers compare a *document* against sources to find copied text — a misconduct tool used after writing. We compare an *idea* against prior ideas to find conceptual duplication — a planning tool used before building. Different input (title+abstract, not full text), different question (novelty, not copying), different moment (ideation, not submission). "All That Glitters is Not Novel" (2025) shows the two genuinely differ: work can pass plagiarism checks and still be substantially unoriginal.

**"Why not just use ChatGPT/Gemini to judge novelty?"**
Three reasons, all published: (1) instability — rephrase the idea and the verdict flips (Wang et al. 2025); (2) self-evaluation failure — LLMs cannot reliably judge novelty, including their own output (Si et al. 2025); (3) no corpus grounding — an LLM judges from memory of its training data, which does not include our department archive, the corpus where rejections actually come from. Our system retrieves from the actual corpora and shows its evidence.

**"Your corpus is only 36 projects. Isn't that toy-scale?"**
The corpus is a *sample standing in for* the department archive — the one input only the department can provide, and the architecture is corpus-agnostic: FAISS + embeddings means swapping in 3,600 real projects changes a data file, not the system. We evaluated the *method* rigorously (60 labeled pairs, retrieval test, baseline comparison) precisely so the claim "it works" doesn't depend on the sample's size. NoveltyRank (2025) validates the same architecture at 60,294 papers.

**"A similarity score isn't novelty. A similar idea can still be a novel contribution."**
Correct — and the system is built on that admission. It reports *similarity to prior work* with evidence and a crowdedness reading; it never claims to judge contribution quality, feasibility, or execution novelty. That is why the output is "here is the closest work — differentiate from it," not a rejection stamp. Shibayama's validation makes the same distinction: embedding distance measures one real component of novelty (distance from prior art), not all of it.

**"What if a student games it by rewording?"**
Rewording is exactly what semantic embeddings are insensitive to — that's the point of the system. To beat it, the *meaning* must change, at which point the idea genuinely has changed. The residual gaming risk is padding with unrelated buzzwords; the density check and shared-concept evidence make that visible to a human reviewer.

**"How is this different from NoveltyRank / existing research systems?"**
Target and corpus. Published systems assess *research papers* against *paper corpora* for reviewers and AI-scientists. Ours assesses *student project ideas* against the *institutional archive plus papers plus code* — the three places a student idea can already exist — packaged as a free, live tool a department can actually adopt. Same scientific lineage, different and unserved user.

**"What are the honest limitations?"**
(1) Sample corpus until the real archive is supplied. (2) Two borderline paraphrases scored just under the duplicate threshold — shown in our evaluation, not hidden. (3) Abstract-level only: a novel *method* inside a familiar problem won't be credited — the verdict text says "differentiate," for this reason. (4) GitHub/arXiv coverage is best-effort via public APIs.

## 6. The roadmap (future scope, each item paper-backed)

1. **Real department archive** — the single highest-value step; method already validated.
2. **SPECTER2 embeddings for the paper corpus** (Cohan 2020; Singh 2023) — citation-aware paper representations.
3. **Full RND density scoring** (Wang et al. 2025) alongside max-similarity.
4. **Cross-encoder re-ranking** of top matches for finer discrimination at the borderline band (0.45–0.65).
5. **Proxy-label evaluation at scale** with chronological splits, per NoveltyRank's methodology.
6. **LLM explanation layer** — an LLM may *narrate* the evidence (never produce the score), per the hybrid design of What Is Novel? (2026).
