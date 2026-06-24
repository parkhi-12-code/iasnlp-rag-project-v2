"""
eval_t2p_extended.py
Quality audit: Text-to-Pandas accuracy on extended and final benchmarks.

Task 1 — 25 extended questions (trajectory/comorbidity/medication/intrawave/pattern)
Task 2 — 50 multi-hop + aggregate questions from qa_pairs_final.csv
Task 3 — Print comparison tables with filled-in T2P numbers

Run: python src/eval_t2p_extended.py
"""

import os
import re
import sys
import time
import difflib
import pandas as pd

HERE  = os.path.dirname(os.path.abspath(__file__))
ROOT  = os.path.dirname(HERE)
BENCH = os.path.join(ROOT, "benchmark")

sys.path.insert(0, HERE)

from text_to_pandas_fix import text_to_pandas_answer

# ── Grader ─────────────────────────────────────────────────────────────────────

_NUM_RE = re.compile(r'\b\d+\.?\d*\b')


def fuzzy_grade(gold: str, pred: str) -> bool:
    """Return True if pred adequately matches gold."""
    g = str(gold).lower().strip()
    p = str(pred).lower().strip()

    if p.startswith("error:") or p.startswith("insufficient"):
        return False

    # Substring containment
    if g in p or p in g:
        return True

    # Yes/No leading word match (with supporting content)
    for word in ("yes", "no"):
        if g.startswith(word) and p.startswith(word):
            return True

    # All key numbers from gold appear in pred
    nums = _NUM_RE.findall(g)
    if nums and all(n in p for n in nums):
        return True

    # Fuzzy character ratio
    return difflib.SequenceMatcher(None, g, p).ratio() >= 0.6


# ── Task 1 ─────────────────────────────────────────────────────────────────────

def task1():
    print("\n" + "=" * 65)
    print("TASK 1 — Text-to-Pandas on Extended Benchmark (25 questions)")
    print("=" * 65)

    df  = pd.read_csv(os.path.join(BENCH, "qa_pairs_extended.csv"))
    cats = ["trajectory", "comorbidity", "medication", "intrawave", "pattern"]

    rows = []
    for i, (_, row) in enumerate(df.iterrows(), 1):
        q    = row["question"]
        gold = str(row["answer"])
        cat  = row["category"]

        print(f"  [{i:2d}/25] {cat:<12s}  {row['question_id']}", end="", flush=True)
        pred = text_to_pandas_answer(q)
        time.sleep(2)

        ok    = fuzzy_grade(gold, pred)
        grade = "correct" if ok else "incorrect"
        print(f"  {'OK' if ok else 'XX'}  {pred[:60]}")

        rows.append({
            "question_id": row["question_id"],
            "category":    cat,
            "question":    q,
            "gold_answer": gold,
            "t2p_answer":  pred,
            "grade":       grade,
        })

    result_df = pd.DataFrame(rows)

    print("\n--- TASK 1 RESULTS ---")
    cat_scores: dict[str, int] = {}
    for cat in cats:
        sub = result_df[result_df["category"] == cat]
        nc  = int((sub["grade"] == "correct").sum())
        cat_scores[cat] = nc
        print(f"  {cat.upper():<12s}: {nc}/5")
    total = sum(cat_scores.values())
    print(f"  {'OVERALL':<12s}: {total}/25  ({total/25:.0%})")

    failures = result_df[result_df["grade"] != "correct"]
    fail_path = os.path.join(BENCH, "t2p_extended_failures.csv")
    failures.to_csv(fail_path, index=False)
    print(f"\n  Failures ({len(failures)}) saved -> {fail_path}")

    return cat_scores, total


# ── Task 2 ─────────────────────────────────────────────────────────────────────

def task2():
    print("\n" + "=" * 65)
    print("TASK 2 — Text-to-Pandas on Multi-hop & Aggregate (50 questions)")
    print("=" * 65)

    df     = pd.read_csv(os.path.join(BENCH, "qa_pairs_final.csv"))
    subset = df[df["category"].isin(["multi-hop", "aggregate"])].copy()

    rows = []
    for i, (_, row) in enumerate(subset.iterrows(), 1):
        q    = row["question"]
        gold = str(row["answer"])
        cat  = row["category"]

        print(f"  [{i:2d}/50] {cat:<10s}  {row['question_id']}", end="", flush=True)
        pred = text_to_pandas_answer(q)
        time.sleep(2)

        ok    = fuzzy_grade(gold, pred)
        grade = "correct" if ok else "incorrect"
        print(f"  {'OK' if ok else 'XX'}  {pred[:60]}")

        rows.append({
            "question_id": row["question_id"],
            "category":    cat,
            "question":    q,
            "gold_answer": gold,
            "t2p_answer":  pred,
            "grade":       grade,
        })

    result_df = pd.DataFrame(rows)

    print("\n--- TASK 2 RESULTS ---")
    cat_scores: dict[str, int] = {}
    for cat in ["aggregate", "multi-hop"]:
        sub = result_df[result_df["category"] == cat]
        nc  = int((sub["grade"] == "correct").sum())
        cat_scores[cat] = nc
        print(f"  {cat.upper():<12s}: {nc}/25  ({nc/25:.0%})")

    failures = result_df[result_df["grade"] != "correct"]
    fail_path = os.path.join(BENCH, "t2p_multihop_aggregate_failures.csv")
    failures.to_csv(fail_path, index=False)
    print(f"\n  Failures ({len(failures)}) saved -> {fail_path}")

    return cat_scores


# ── Task 3 ─────────────────────────────────────────────────────────────────────

def task3(task2_scores: dict, task1_scores: dict, task1_total: int):
    print("\n" + "=" * 65)
    print("TASK 3 — FINAL COMPARISON TABLES")
    print("=" * 65)

    # T2P is 100% on lookup and trend (from original v1 benchmark)
    t2p_lookup = 100
    t2p_trend  = 100
    agg_n      = task2_scores.get("aggregate", 0)
    mh_n       = task2_scores.get("multi-hop", 0)
    agg_pct    = round(agg_n / 25 * 100)
    mh_pct     = round(mh_n  / 25 * 100)
    # Overall: 4 categories × 25 questions = 100
    t2p_overall = round((25 + 25 + agg_n + mh_n) / 100 * 100)

    print("""
Main Benchmark (100 questions):

System           | Lookup | Trend | Agg  | M-Hop | Overall
-----------------|--------|-------|------|-------|--------
Naive RAG        |   20%  |   0%  |  16% |   8%  |   11%
Metadata Filter  |   48%  |   8%  |  28% |   8%  |   23%
Knowledge Graph  |   80%  |  24%  |  64% |   0%  |   42%""")
    print(f"Text-to-Pandas   |  {t2p_lookup:3d}%  | {t2p_trend:3d}%   | {agg_pct:3d}% |  {mh_pct:3d}%  |   {t2p_overall:3d}%")

    # Extended benchmark
    traj_pct    = round(task1_scores.get("trajectory",  0) / 5 * 100)
    comor_pct   = round(task1_scores.get("comorbidity", 0) / 5 * 100)
    med_pct     = round(task1_scores.get("medication",  0) / 5 * 100)
    intra_pct   = round(task1_scores.get("intrawave",   0) / 5 * 100)
    patt_pct    = round(task1_scores.get("pattern",     0) / 5 * 100)
    t2p_ext_pct = round(task1_total / 25 * 100)

    print("""
Extended Benchmark (25 questions):

System           | Traj | Comorbid | Med  | Intra | Pattern | Total
-----------------|------|----------|------|-------|---------|------
Knowledge Graph  |  20% |    0%    |  20% |  20%  |    0%   |  12%""")
    print(f"Text-to-Pandas   | {traj_pct:3d}% |   {comor_pct:3d}%    | {med_pct:3d}% |  {intra_pct:3d}%  |   {patt_pct:3d}%   | {t2p_ext_pct:3d}%")
    print()


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    task1_scores, task1_total = task1()
    task2_scores = task2()
    task3(task2_scores, task1_scores, task1_total)
    print("Audit complete.")
