"""
eval_comorbidity.py
--------------------
Evaluates text_to_pandas_answer() on benchmark/qa_pairs_comorbidity.csv.

Every question in this benchmark matches one of the 3 fixed comorbidity
templates, so extract_comorbidity_spec() intercepts all 20 before the Groq
call and dispatches directly to H35/H36/H37 (comorbidity_handlers.py). To
prove that at eval time (not just by construction), the Groq client is
patched to raise if it's ever invoked -- a silent LLM fallback would fail
the run loudly instead of passing quietly.

Grading uses the same fuzzy_grade() as eval_t2p_v2.py (the v2 benchmark
runner): substring match, yes/no-prefix match, or numeric-token match,
falling back to a difflib similarity ratio.

Saves: benchmark/comorbidity_eval_results.csv
"""

import os
import re
import sys
import difflib

import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
_BENCH = os.path.join(_ROOT, "benchmark")

sys.path.insert(0, _HERE)
import text_to_pandas_fix as t2p


def _forbidden_groq_call(*_args, **_kwargs):
    raise AssertionError(
        "Groq API was called during the comorbidity eval -- the regex "
        "prepend failed to intercept a question."
    )


_NUM_RE = re.compile(r'\b\d+\.?\d*\b')


def fuzzy_grade(gold: str, pred: str) -> bool:
    g = str(gold).lower().strip()
    p = str(pred).lower().strip()
    if p.startswith("error:") or p.startswith("insufficient"):
        return False
    if g in p or p in g:
        return True
    for word in ("yes", "no"):
        if g.startswith(word) and p.startswith(word):
            return True
    nums = _NUM_RE.findall(g)
    if nums and all(n in p for n in nums):
        return True
    return difflib.SequenceMatcher(None, g, p).ratio() >= 0.6


def _section(question_id: str) -> str:
    if question_id.startswith("COMO_C"):
        return "count"
    if question_id.startswith("COMO_B"):
        return "boolean"
    if question_id.startswith("COMO_D"):
        return "duration"
    if question_id.startswith("COMO_X"):
        return "contrast"
    return "other"


def main():
    # Guard: fail loudly if anything reaches the LLM path during this run.
    t2p.groq_client.chat.completions.create = _forbidden_groq_call

    df = pd.read_csv(os.path.join(_BENCH, "qa_pairs_comorbidity.csv"))

    rows = []
    for _, r in df.iterrows():
        pred = t2p.text_to_pandas_answer(r["question"])
        gold = str(r["answer"])
        rows.append({
            "question_id": r["question_id"],
            "section":     _section(r["question_id"]),
            "question":    r["question"],
            "gold":        gold,
            "pred":        pred,
            "pass":        fuzzy_grade(gold, pred),
        })

    out = pd.DataFrame(rows)
    out_path = os.path.join(_BENCH, "comorbidity_eval_results.csv")
    out.to_csv(out_path, index=False)

    print(f"OVERALL: {out['pass'].sum()}/{len(out)}")
    for sec in ["count", "boolean", "duration", "contrast"]:
        sub = out[out["section"] == sec]
        print(f"  {sec:10s}: {sub['pass'].sum()}/{len(sub)}")
    print(f"\nSaved -> {out_path}")
    print("No exception raised -> Groq client was never invoked (fully offline).")


if __name__ == "__main__":
    main()
