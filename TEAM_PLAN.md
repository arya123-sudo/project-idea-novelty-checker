# Team Plan — Remaining Work (4 members)

The engine, evaluation, research pack and deployment are **already done and live**
(https://project-idea-novelty-checker.onrender.com). This plan covers what is
actually left: real data, verification, documentation and presentation.
Roles follow each member's strength; every task has a visible deliverable.

## Status snapshot

| Area | State |
|---|---|
| Core engine (SBERT + FAISS + scoring) | ✅ Built, tested |
| Sources: Department, arXiv, GitHub, Semantic Scholar, OpenAlex | ✅ Live (S2/OpenAlex best-effort — public APIs rate-limit) |
| Density/crowdedness + explanations + gap analysis | ✅ Live |
| Evaluation (60 pairs + retrieval test) | ✅ Done — AUC 1.000, F1 0.966, Recall@3 1.000 |
| Research pack (12 verified papers) | ✅ Done — `RESEARCH_PAPERS.md` |
| Deployment (Render) | ✅ Live |
| **Real department archive** | ❌ **The one missing input** |
| Final report + PPT | 🟡 Drafts exist (`REPORT.md`, synopsis deck) — need final polish |

## Member assignments

### Arya — Lead / Systems
- Own the live deployment: keep the Render service warm before demos; redeploy after data updates (Manual Deploy in dashboard).
- Own the "ask" for the real archive (with Bhargav): one message to the guide requesting past mini-project titles/abstracts.
- Final integration check: run 5 test ideas through the live site before any review; record screenshots.

### Bhargav Raj — Data (corpus owner)
- Collect the real department archive: past AIML mini-project titles + abstracts (target: 2–3 batches, 60+ projects) from the guide/department records/seniors.
- Convert to the `data/dept_projects.json` format (title, abstract, year) — format example is in the repo.
- Re-run the two evaluations after the swap and note any metric changes for the report.
- Backup task: extend labeled pairs from 60 → 80 with pairs drawn from the real archive.

### D Bargav — Verification & Testing
- Human-judgement study: pick 15 ideas, have 2–3 faculty/peers rate novelty blind, compare with system scores (agreement % for the report — this mirrors Shibayama's validation method).
- Regression checklist: run the standard test ideas (crop disease, attendance, beehive + 5 own) after every change; log results.
- Edge cases: empty abstract, very long abstract, non-English title, exact-duplicate title — document behaviour.

### G Vajresh Shenoy — Documentation & Presentation
- Final report assembly from `REPORT.md` (college format, guide name confirmed on cover).
- PPT from `EXPLANATION.md`: problem → technique → evaluation → demo → limitations → future scope (10–12 slides).
- Demo script: 2 ideas live (one duplicate, one novel) + the crowdedness contrast; rehearse the tough Q&A in `EXPLANATION.md` §5 with the team.
- Screenshots of the live UI for the report appendix.

## Sequence (suggested 10 working days)

1. **Days 1–2:** Archive request sent; Bhargav starts collection from seniors in parallel. Vajresh confirms report format + guide name spelling.
2. **Days 3–5:** Corpus swap + evaluations re-run (Bhargav + Arya). Bargav starts the human-judgement study.
3. **Days 6–8:** Report finalised with real-corpus numbers; PPT built; edge-case log closed.
4. **Days 9–10:** Full-team demo rehearsal + viva Q&A practice; final screenshots; submission.

## Rules the team agreed implicitly — keep them
- Never cite a paper that isn't in `RESEARCH_PAPERS.md` (all 12 verified; a ChatGPT-suggested list shared on 12 Nov contained fabricated citations — do not use it).
- The novelty score is never replaced by an LLM's opinion; explanations come from retrieved evidence.
- Claims stay honest: the corpus is a sample until the real archive lands — say so until it does.
