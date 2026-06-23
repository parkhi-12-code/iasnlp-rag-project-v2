"""
eval_extended.py
Evaluates the 25-question extended benchmark against the graph-RAG pipeline.

Input  : benchmark/qa_pairs_extended.csv
Outputs: benchmark/extended_results.csv
         benchmark/extended_failure_log.csv
"""

import sys
import time
import difflib
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from graph_rag_pipeline import run_pipeline

BENCHMARK_PATH = Path(__file__).parent.parent / "benchmark" / "qa_pairs_extended.csv"
RESULTS_PATH   = Path(__file__).parent.parent / "benchmark" / "extended_results.csv"
FAILURES_PATH  = Path(__file__).parent.parent / "benchmark" / "extended_failure_log.csv"

FUZZY_THRESHOLD = 0.6
SLEEP_SECONDS   = 2

DISPLAY_ORDER = [
    ("trajectory",  "TRAJECTORY"),
    ("comorbidity", "COMORBIDITY"),
    ("medication",  "MEDICATION"),
    ("intrawave",   "INTRAWAVE"),
    ("pattern",     "PATTERN"),
]


def _normalize(text: str) -> str:
    return " ".join(str(text).lower().strip().split())


def fuzzy_match(pred: str, gold: str) -> bool:
    pn = _normalize(pred)
    gn = _normalize(gold)

    if pn == gn:
        return True
    if gn in pn:
        return True
    if difflib.SequenceMatcher(None, pn, gn).ratio() >= FUZZY_THRESHOLD:
        return True

    # Semicolon-list: ≥70 % of gold items found in pred
    if ";" in gn:
        gold_items = {x.strip() for x in gn.split(";")}
        pred_flat  = pn.replace(";", " ").replace(",", " ")
        found = sum(1 for item in gold_items if item in pred_flat)
        if found >= len(gold_items) * 0.7:
            return True

    return False


def run_evaluation():
    df = pd.read_csv(BENCHMARK_PATH)
    total = len(df)
    print(f"Loaded {total} questions from {BENCHMARK_PATH.name}")
    print(f"Categories: {df['category'].value_counts().to_dict()}")
    print(f"\nRunning evaluation (sleep={SLEEP_SECONDS}s between calls)...\n")

    results  = []
    failures = []
    cat_stats: dict[str, dict] = {}

    for i, row in df.iterrows():
        qid      = str(row["question_id"])
        category = str(row["category"])
        pid      = str(row["patient_id"])
        wave_ref = str(row.get("wave_reference", ""))
        question = str(row["question"])
        gold     = str(row["answer"])

        print(f"[{i+1:02d}/{total}] {qid} | {category} | pid={pid} ...", end=" ", flush=True)

        try:
            result = run_pipeline(
                question=question,
                patient_id=pid,
                wave_reference=wave_ref,
                category=category,
            )
            pred = result["answer"]
            ok   = fuzzy_match(pred, gold)
            tag  = "PASS" if ok else "FAIL"
        except Exception as e:
            pred = f"ERROR: {e}"
            ok   = False
            tag  = "ERROR"

        print(tag, flush=True)

        record = {
            "question_id":      qid,
            "category":         category,
            "patient_id":       pid,
            "wave_reference":   wave_ref,
            "question":         question,
            "gold_answer":      gold,
            "predicted_answer": pred,
            "correct":          ok,
            "status":           tag,
        }
        results.append(record)
        if not ok:
            failures.append(record)

        if category not in cat_stats:
            cat_stats[category] = {"correct": 0, "total": 0}
        cat_stats[category]["total"] += 1
        if ok:
            cat_stats[category]["correct"] += 1

        time.sleep(SLEEP_SECONDS)

    # Save outputs
    pd.DataFrame(results).to_csv(RESULTS_PATH, index=False)
    print(f"\nResults  -> {RESULTS_PATH}")
    if failures:
        pd.DataFrame(failures).to_csv(FAILURES_PATH, index=False)
        print(f"Failures -> {FAILURES_PATH}")
    else:
        print("No failures -- extended_failure_log.csv not written")

    # Print accuracy by category
    total_correct = sum(v["correct"] for v in cat_stats.values())
    total_qs      = sum(v["total"]   for v in cat_stats.values())

    print()
    for key, label in DISPLAY_ORDER:
        if key in cat_stats:
            c   = cat_stats[key]["correct"]
            t   = cat_stats[key]["total"]
            pct = round(c / t * 100) if t else 0
            print(f"{label + ':':13s} {c}/{t}  ({pct}%)")

    pct_total = round(total_correct / total_qs * 100) if total_qs else 0
    print(f"{'OVERALL:':13s} {total_correct}/{total_qs} ({pct_total}%)")

    return pd.DataFrame(results)


if __name__ == "__main__":
    run_evaluation()
