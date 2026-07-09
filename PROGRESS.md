# Progress Log

## NEXT SESSION — start here
- Task A (local SLM) is BLOCKED on this machine: CPU inference corrupts output
  (both 3B and 1B — see docs/slm_local_feasibility_note.md). NOT an SLM verdict.
- Resume on Kaggle (GPU, free): clean 3B pull → tiered extraction probe vs
  known-correct specs → LoRA fine-tune on 175 Q→spec pairs if 3B works.
- Need to bring to Kaggle: _EXTRACTION_PROMPT + a few Q→spec example pairs.
- Groq remains extraction backend meanwhile. Daily token cap ~500K (~one 100Q
  eval/day) — this is WHY the SLM path matters.
- Belt-and-suspenders TODO (non-blocking): full 100Q Groq rerun after quota
  reset, to reconfirm the regression gate empirically (already closed by proof).

## DONE THIS SESSION
- Froze award state as tag v1.0-iasnlp-award; future-work branch, main protected.
- Interval-overlap logic built + verified (2 anchors + hand-trace).
- 20-question comorbidity benchmark, deterministic gold, committed.
- Router wired additively; regression gate closed by proof (0/100 match).
- Comorbidity eval: 20/20, fully offline. Closes finding F4 (was 0%).
- Documentation + 4 figures in docs/. SLM feasibility note (inconclusive).

## 2026-07-07 — Comorbidity (co-active conditions) gap closed end-to-end

Built interval-overlap logic (`comorbidity_handlers.py`, H35/H36/H37) to fix the 0%/0%
comorbidity gap (finding F4): the old system only asked "was condition X diagnosed in
wave W" (event-based), never "was X *active* during W" (interval-based), so a condition
diagnosed once and never resolved was invisible to every later wave. Verified the fix
against two anchor patients (Kelly223, Sanford861), including reconciling a count
discrepancy down to a single legitimately-active acute condition and confirming
`nunique()` correctly collapses recurring episodes (e.g. Sanford's two Viral sinusitis
bouts, 20 raw rows -> 19 distinct) rather than inflating burden. Generated a 20-question
comorbidity benchmark (`qa_pairs_comorbidity.csv`) with every gold answer computed by
running the real handlers — never hand-typed — covering count, boolean co-occurrence,
duration, and 4 contrast questions quantifying the event-based undercount (e.g. Wave 6
Sanford: event=3 vs interval=21). Wired a deterministic regex router
(`comorbidity_router.py`) that dispatches these 3 question types straight to the interval
handlers, bypassing LLM extraction entirely — added as a pure prepend to
`text_to_pandas_answer`, verified as a structural no-op on the locked 100-question
benchmark (0/100 locked questions match a comorbidity template, so the regression gate
closed by proof rather than by a live rerun, since the shared Groq daily token quota was
exhausted mid-verification). Final result: **20/20 on the comorbidity benchmark,
confirmed fully offline** (Groq client patched to raise on any call — never invoked).
