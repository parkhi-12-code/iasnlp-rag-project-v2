"""
src/eval_t2p_v2.py
Evaluate Text-to-Pandas accuracy on the 50 new v2 benchmark questions.

Loads benchmark/qa_pairs_v2.csv, runs each question through
text_to_pandas_answer(), grades with fuzzy match, and reports:
  AGGREGATE (V2): X/25 (XX%)
  MULTI-HOP (V2): X/25 (XX%)
  OVERALL (V2):   X/50 (XX%)

Saves:
  benchmark/t2p_v2_results.csv  — all 50 rows with grade
  benchmark/t2p_v2_failures.csv — failed rows with failure reason
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

# ── T2P known handler categories ───────────────────────────────────────────────
_T2P_HANDLERS = {
    "lookup_condition",
    "lookup_observation",
    "lookup_demographics",
    "trend",
    "aggregate_condition_count",
    "aggregate_observation_count",
    "aggregate_distinct_waves",
    "multihop_hypertension",
    "multihop_bmi_obesity",
}

# Keywords that hint which T2P handler Groq is likely to pick
_HANDLER_HINTS = [
    (r"condition.*wave|wave.*condition",         "lookup_condition"),
    (r"gender|birth year",                       "lookup_demographics"),
    (r"average.*wave|wave.*average",             "lookup_observation"),
    (r"change from wave \d+ to wave \d+",        "trend"),
    (r"how many unique conditions",              "aggregate_condition_count"),
    (r"how many observations.*wave",             "aggregate_observation_count"),
    (r"how many.*waves.*conditions|across.*waves", "aggregate_distinct_waves"),
    (r"hypertension.*before.*after|sbp.*higher after", "multihop_hypertension"),
    (r"bmi.*increase.*obesity|obesity.*bmi",     "multihop_bmi_obesity"),
]

def _infer_handler(question: str) -> str:
    """Best-guess at which T2P handler the LLM will pick, or 'unknown'."""
    q = question.lower()
    for pat, handler in _HANDLER_HINTS:
        if re.search(pat, q):
            return handler
    return "unknown"

def _failure_reason(question: str, pred: str) -> str:
    """Classify why T2P failed on a v2 question."""
    q = question.lower()
    p = pred.lower()

    if p.startswith("error:") or p.startswith("no ") or "not found" in p:
        return "data-not-found / handler-error"

    if "insufficient" in p:
        return "handler returned insufficient-data"

    # Check if the question type is simply outside T2P's handler set
    novel_patterns = [
        (r"highest average body mass index",          "no handler: peak-BMI-wave"),
        (r"most unique medications",                  "no handler: medication-heavy-wave"),
        (r"no new conditions",                        "no handler: condition-free-waves"),
        (r"chronic.*no recorded stop",                "no handler: chronic-vs-acute"),
        (r"average number of observations per wave",  "no handler: obs-density"),
        (r"recorded stop date.*time.limited",         "no handler: medication-duration"),
        (r"health record span",                       "no handler: record-span"),
        (r"overweight range",                         "no handler: BMI-category"),
        (r"new conditions.*diagnosed.*per wave",      "no handler: condition-rate"),
        (r"most total health events",                 "no handler: most-active-wave"),
        (r"blood pressure readings.*total",           "no handler: BP-count-total"),
        (r"first condition ever",                     "no handler: first-condition"),
        (r"no medications prescribed",                "no handler: medication-gap"),
        (r"recorded height remain stable",            "no handler: height-stability"),
        (r"distinct types of clinical",               "no handler: obs-type-variety"),
        (r"fraction of waves.*condition",             "no handler: wave-coverage"),
        (r"active simultaneously",                    "no handler: peak-comorbidity"),
        (r"with no medications at all",               "no handler: med-free-waves"),
        (r"total number of distinct.*medications.*obs", "no handler: health-burden"),
        (r"range of average heart rates",             "no handler: HR-range"),
        (r"overall increase.*decrease.*stable",       "no handler: BMI-trend-dir"),
        (r"active.*most number of waves",             "no handler: longest-condition"),
        (r"entirely new diagnosis",                   "no handler: new-diagnosis-waves"),
        (r"most complete clinical picture",           "no handler: obs-completeness"),
        (r"total count.*prescription",                "no handler: total-rx-burden"),
        (r"bmi.*first.*diagnosed",                    "no handler: BMI-at-first-diag"),
        (r"first medication.*before.*after.*condition", "no handler: med-timing"),
        (r"same wave.*first diagnosed",               "no handler: obs-at-diag"),
        (r"average body weight.*first prescribed",    "no handler: weight-at-first-med"),
        (r"immediately after.*hypertension",          "close: multihop-HTN-next-wave"),
        (r"new conditions.*after.*first medication",  "no handler: conditions-post-med"),
        (r"obese range.*hypertension",                "close: multihop-BMI-at-HTN"),
        (r"glucose.*diabetes|diabetes.*glucose",      "no handler: glucose-at-diagnosis"),
        (r"before.*prescribed.*hypertension",         "close: multihop-HTN-med-order"),
        (r"heart rate.*chronic condition",            "no handler: HR-at-chronic-diag"),
        (r"medications.*peak bmi",                    "no handler: med-at-peak-BMI"),
        (r"how old.*chronic condition",               "no handler: age-at-first-chronic"),
        (r"observations.*change.*diagnosed",          "no handler: obs-change-at-diag"),
        (r"both.*first diagnosed.*same wave",         "no handler: dual-condition-wave"),
        (r"lower after.*started taking",              "no handler: SBP-before-after-med"),
        (r"body weight increase.*diagnosed",          "no handler: weight-at-diag"),
        (r"bmi above 25.*first prescribed",           "no handler: BMI-at-first-med"),
        (r"condition burden.*waves 1-2.*3-4",         "no handler: conditions-by-decade"),
        (r"observations.*before.*first condition",    "no handler: obs-before-condition"),
        (r"medications stopped.*condition.*active",   "no handler: med-stopped-active"),
        (r"heart rate.*above.*personal average",      "no handler: HR-vs-personal-avg"),
        (r"new medication.*every wave.*new condition","no handler: med-per-new-condition"),
        (r"longest gap.*waves.*condition diagnoses",  "no handler: longest-diag-gap"),
        (r"most recent wave.*clinically intense",     "no handler: recent-vs-peak-obs"),
        (r"cardiovascular.*metabolic|metabolic.*cardiovascular", "no handler: cardio-vs-metabolic"),
    ]

    for pat, reason in novel_patterns:
        if re.search(pat, q):
            return reason

    guessed_handler = _infer_handler(question)
    if guessed_handler in _T2P_HANDLERS:
        return f"handler '{guessed_handler}' mapped but wrong answer"
    return "unknown category — T2P returned wrong/unrelated answer"


# ── Fuzzy grader ───────────────────────────────────────────────────────────────
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


# ── Main ────────────────────────────────────────────────────────────────────────
def main():
    bench_path = os.path.join(BENCH, "qa_pairs_v2.csv")
    if not os.path.exists(bench_path):
        print(f"ERROR: {bench_path} not found — run generate_v2_benchmark.py first.")
        sys.exit(1)

    df = pd.read_csv(bench_path)
    print(f"Loaded {len(df)} questions from qa_pairs_v2.csv")
    print(f"Categories: {df['category'].value_counts().to_dict()}\n")
    print("=" * 70)
    print("Running Text-to-Pandas on all 50 v2 questions (4s sleep between calls)")
    print("=" * 70)

    rows = []
    for i, (_, row) in enumerate(df.iterrows(), 1):
        qid  = row["question_id"]
        cat  = row["category"]
        q    = row["question"]
        gold = str(row["answer"])

        print(f"[{i:2d}/50] {cat:<11s}  {qid}", end="", flush=True)
        pred  = text_to_pandas_answer(q)
        time.sleep(4)

        ok    = fuzzy_grade(gold, pred)
        grade = "correct" if ok else "incorrect"
        print(f"  {'OK' if ok else 'XX'}  {pred[:55]}")

        reason = "" if ok else _failure_reason(q, pred)

        rows.append({
            "question_id":   qid,
            "category":      cat,
            "patient_id":    row["patient_id"],
            "question":      q,
            "gold_answer":   gold,
            "t2p_answer":    pred,
            "grade":         grade,
            "failure_reason": reason,
        })

    result_df = pd.DataFrame(rows)

    # ── Save results ──────────────────────────────────────────────────────────
    res_path  = os.path.join(BENCH, "t2p_v2_results.csv")
    fail_path = os.path.join(BENCH, "t2p_v2_failures.csv")

    result_df.to_csv(res_path, index=False)

    failures = result_df[result_df["grade"] != "correct"].copy()
    failures.to_csv(fail_path, index=False)

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("RESULTS — Text-to-Pandas on V2 Benchmark")
    print("=" * 70)

    agg_df = result_df[result_df["category"] == "aggregate"]
    mh_df  = result_df[result_df["category"] == "multi-hop"]

    agg_ok = int((agg_df["grade"] == "correct").sum())
    mh_ok  = int((mh_df["grade"] == "correct").sum())
    total  = agg_ok + mh_ok

    print(f"\nAGGREGATE (V2): {agg_ok}/25 ({agg_ok/25:.0%})")
    print(f"MULTI-HOP (V2): {mh_ok}/25 ({mh_ok/25:.0%})")
    print(f"OVERALL (V2):   {total}/50 ({total/50:.0%})")

    print(f"\nResults saved -> {res_path}")
    print(f"Failures saved -> {fail_path}  ({len(failures)} failures)\n")

    # ── Failure detail ────────────────────────────────────────────────────────
    if not failures.empty:
        print("=" * 70)
        print(f"FAILURE DETAILS ({len(failures)} questions)")
        print("=" * 70)
        for _, f in failures.iterrows():
            print(f"\n{f['question_id']}  [{f['category']}]")
            print(f"  Q:      {f['question']}")
            print(f"  Gold:   {f['gold_answer']}")
            print(f"  Pred:   {f['t2p_answer']}")
            print(f"  Reason: {f['failure_reason']}")

    # ── Reason breakdown ──────────────────────────────────────────────────────
    if not failures.empty:
        print("\n" + "=" * 70)
        print("FAILURE REASON BREAKDOWN")
        print("=" * 70)
        reasons = failures["failure_reason"].value_counts()
        for reason, count in reasons.items():
            print(f"  {count:2d}x  {reason}")

    print()


if __name__ == "__main__":
    main()
