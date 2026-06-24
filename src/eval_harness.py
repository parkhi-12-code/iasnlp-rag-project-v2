"""
eval_harness.py
Reusable grading harness for any QA pipeline evaluated against qa_pairs_final.csv.
Import run_eval() from any pipeline script — it doesn't care how answers are produced.
"""
import os
import re
import pandas as pd

_NUM_RE = re.compile(r'\b\d+\.?\d*\b')
_STOP   = {
    "with", "were", "from", "that", "this", "have", "been",
    "year", "born", "and", "the", "for", "patient", "also",
}


def _nums(text: str) -> list:
    return _NUM_RE.findall(text)


# ── Grading ────────────────────────────────────────────────────────────────────

def grade(category: str, ground_truth: str, generated: str) -> str:
    """Heuristic grade: 'correct' | 'uncertain' | 'incorrect'."""
    gt  = ground_truth.lower()
    gen = generated.lower()

    if "insufficient context" in gen or gen.startswith("error:"):
        return "incorrect"

    if category == "aggregate":
        return "correct" if ground_truth.strip() in gen else "incorrect"

    if category == "trend":
        direction = next(
            (w for w in ("increased", "decreased", "remained stable") if w in gt), None
        )
        dir_ok = bool(direction and direction in gen)
        hits   = sum(1 for n in _nums(ground_truth) if n in generated)
        if dir_ok and hits >= 2:
            return "correct"
        return "uncertain" if (dir_ok or hits >= 1) else "incorrect"

    if category == "lookup":
        gt_nums = _nums(ground_truth)
        if gt_nums:
            hits = sum(1 for n in gt_nums if n in generated)
            if hits == len(gt_nums):
                return "correct"
            return "uncertain" if hits > 0 else "incorrect"
        gt_words = {w for w in re.findall(r'\b[a-z]{4,}\b', gt) if w not in _STOP}
        if not gt_words:
            return "uncertain"
        ratio = sum(1 for w in gt_words if w in gen) / len(gt_words)
        if ratio >= 0.7:
            return "correct"
        return "uncertain" if ratio >= 0.3 else "incorrect"

    if category == "multi-hop":
        direction = next(
            (w for w in ("higher", "lower", "increased", "decreased", "remained stable") if w in gt),
            None,
        )
        dir_ok = bool(direction and direction in gen)
        hits   = sum(1 for n in _nums(ground_truth) if n in generated)
        if dir_ok and hits >= 2:
            return "correct"
        return "uncertain" if (dir_ok or hits >= 1) else "incorrect"

    return "uncertain"


def classify_failure(
    category: str,
    ground_truth: str,
    generated: str,
    pid_short: str = None,
    retrieved_chunks: list = None,
) -> str:
    """
    Best-effort failure-type label.

    Pass retrieved_chunks for full RAG-specific classification (detects
    no_relevant_chunk_in_top5 / wrong_chunk_retrieved). Omit for pipeline-agnostic
    classification using only category and answer text.
    """
    if retrieved_chunks is not None:
        patient_chunks = [c for c in retrieved_chunks if pid_short and pid_short in c]
        if not patient_chunks:
            return "no_relevant_chunk_in_top5"
        if pid_short and all(pid_short not in c for c in retrieved_chunks):
            return "wrong_chunk_retrieved"
        chunk_nums   = set(_nums(" ".join(retrieved_chunks)))
        hallucinated = set(_nums(generated)) - chunk_nums
        if category in ("trend", "multi-hop"):
            return "correct_chunk_wrong_wave_mixed_in"
        if category == "aggregate":
            return "correct_data_wrong_synthesis"
        if len(hallucinated) > 1:
            return "hallucinated_value"
        return "other"

    # No chunk context: infer from category and answer text
    if category in ("trend", "multi-hop"):
        return "correct_chunk_wrong_wave_mixed_in"
    if category == "aggregate":
        return "correct_data_wrong_synthesis"
    if len(set(_nums(generated)) - set(_nums(ground_truth))) > 1:
        return "hallucinated_value"
    return "other"


def grader(
    category: str,
    ground_truth: str,
    generated: str,
    pid_short: str = None,
    retrieved_chunks: list = None,
) -> tuple:
    """Return (grade, failure_type). failure_type is '' when grade == 'correct'."""
    g = grade(category, ground_truth, generated)
    ft = "" if g == "correct" else classify_failure(
        category, ground_truth, generated, pid_short, retrieved_chunks
    )
    return g, ft


# ── Eval runner ────────────────────────────────────────────────────────────────

def run_eval(qa_df: pd.DataFrame, answer_fn, output_path: str, run_label: str) -> tuple:
    """
    Evaluate answer_fn against every row in qa_df.

    answer_fn(question: str) -> str | (str, dict)
        Returns the generated answer as a plain string, or a (answer, metadata) tuple
        where metadata may contain 'retrieved_chunks': list[str]. When chunks are
        provided they are passed to grader for full failure classification and stored
        in the output CSV; otherwise classification falls back to category heuristics.

    Returns (results_df, summary_dict).
    """
    total   = len(qa_df)
    results = []

    print(f"\n[{run_label}] Evaluating {total} questions...\n")

    for seq, (_, row) in enumerate(qa_df.iterrows(), start=1):
        question  = row["question"]
        gt_answer = row["answer"]
        category  = row["category"]
        pid_short = str(row["patient_id"])

        # Unpack (answer, metadata) tuple if the pipeline returns one
        raw = answer_fn(question)
        if isinstance(raw, tuple) and len(raw) == 2:
            gen_answer, meta = raw
        else:
            gen_answer, meta = raw, {}
        retrieved_chunks = meta.get("retrieved_chunks") if isinstance(meta, dict) else None

        g, ft = grader(
            category, gt_answer, gen_answer,
            pid_short=pid_short,
            retrieved_chunks=retrieved_chunks,
        )

        row_data = {
            "question_id":         row["question_id"],
            "category":            category,
            "question":            question,
            "ground_truth_answer": gt_answer,
            "retrieved_chunks":    " ||| ".join(retrieved_chunks) if retrieved_chunks else "",
            "generated_answer":    gen_answer,
            "grade":               g,
            "failure_type":        ft,
        }
        results.append(row_data)

        icon = "OK" if g == "correct" else ("??" if g == "uncertain" else "XX")
        print(f"[{seq:3d}/{total}] {icon}  {category:10s}  {row['question_id']}")

    out_df = pd.DataFrame(results)
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    out_df.to_csv(output_path, index=False)
    print(f"\nSaved -> {output_path}")

    # ── Summary ────────────────────────────────────────────────────────────────
    n_correct   = int((out_df["grade"] == "correct").sum())
    n_uncertain = int((out_df["grade"] == "uncertain").sum())
    n_incorrect = int((out_df["grade"] == "incorrect").sum())

    print(f"\n{'='*55}")
    print(f"  [{run_label}]  {n_correct}/{total} correct ({n_correct/total:.1%})"
          f"   uncertain={n_uncertain}  incorrect={n_incorrect}")
    print(f"{'='*55}")

    summary = {
        "run_label":   run_label,
        "total":       total,
        "correct":     n_correct,
        "uncertain":   n_uncertain,
        "incorrect":   n_incorrect,
        "accuracy":    n_correct / total,
        "by_category": {},
    }

    print("\nAccuracy by category:")
    for cat in ["lookup", "trend", "aggregate", "multi-hop"]:
        sub = out_df[out_df["category"] == cat]
        nc  = int((sub["grade"] == "correct").sum())
        nu  = int((sub["grade"] == "uncertain").sum())
        print(f"  {cat:12s}: {nc:2d}/{len(sub)} correct ({nc/len(sub):.1%})"
              f"   uncertain={nu}")
        summary["by_category"][cat] = {
            "correct":  nc,
            "total":    len(sub),
            "accuracy": nc / len(sub),
            "uncertain": nu,
        }

    failed = out_df[out_df["grade"] != "correct"]
    print(f"\nFailure-type breakdown ({len(failed)} non-correct rows):")
    if failed.empty:
        print("  (none)")
    else:
        ft_counts = failed["failure_type"].value_counts()
        for ft, cnt in ft_counts.items():
            print(f"  {ft:45s}: {cnt}")
        summary["failure_types"] = ft_counts.to_dict()

    print()
    return out_df, summary
