"""
benchmark/generate_comorbidity_benchmark.py

Generates benchmark/qa_pairs_comorbidity.csv — 20 comorbidity benchmark questions.
All gold answers computed by running interval-overlap handlers (H35/H36/H37).
Re-runnable and deterministic. No hand-typed answers.

Schema matches qa_pairs_final.csv exactly (9 columns):
  question_id, category, patient_id, wave_reference, question,
  answer, verified, notes, temporal
"""

import os, sys, csv
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_ROOT, "src"))

import comorbidity_handlers as ch

DATA = os.path.join(_ROOT, "data", "raw", "conditions.csv")
OUT  = os.path.join(_HERE, "qa_pairs_comorbidity.csv")

KELLY   = "b118adba-1330-4ba0-8d96-fe610dfbd41e"
SANFORD = "3f336702-bf73-4fc8-bd59-3ba77fd65d0d"
K8, S8  = "b118adba", "3f336702"

COLS = [
    "question_id", "category", "patient_id", "wave_reference",
    "question", "answer", "verified", "notes", "temporal",
]

# ── Load data ─────────────────────────────────────────────────────────────────

# Raw CSV — what comorbidity_handlers expects (no WAVE column)
raw = pd.read_csv(DATA)

# Conditions with event-based WAVE column for contrast note computation
# Replicates data_utils._assign_wave without loading the full table set
cond_w = pd.read_csv(DATA)
cond_w["START"] = pd.to_datetime(cond_w["START"], utc=True).dt.tz_convert(None)
cond_w["WAVE"]  = cond_w["START"].apply(
    lambda dt: max(1, (dt.year - 1990) // 5 + 1) if pd.notna(dt) else None
)


def _old_event_count(pid: str, wave: int) -> int:
    """Replica of handle_lookup_condition → count. Event-based (WAVE == wave)."""
    sub = cond_w[(cond_w["PATIENT"] == pid) & (cond_w["WAVE"] == wave)]
    return int(sub["DESCRIPTION"].nunique())


def _row(qid, pid8, wave_ref, question, answer, notes):
    return {
        "question_id":    qid,
        "category":       "comorbidity",
        "patient_id":     pid8,
        "wave_reference": wave_ref,
        "question":       question,
        "answer":         str(answer),
        "verified":       "correct",
        "notes":          notes,
        "temporal":       True,
    }


rows = []

# ── SECTION A — COUNT (H37: handler_comorbidity_count) ───────────────────────

# Reconciliation notes for counts that were cross-checked against an earlier
# manual figure and confirmed correct (see COMO_C02 / COMO_C06 below).
COUNT_NOTES = {
    "COMO_C02": (
        "13 = 12 chronic (STOP null) + 1 resolved acute (Sinusitis "
        "2013-09-21 to 2014-01-04) fully contained in Wave 5; counted "
        "over distinct conditions via nunique"
    ),
    "COMO_C06": (
        "19 distinct conditions; two Viral sinusitis episodes (2010, 2011) "
        "collapse to 1 via nunique — count is over distinct conditions, "
        "not condition-events"
    ),
}

for qid, pid8, full_pid, wave in [
    ("COMO_C01", K8, KELLY,   3),
    ("COMO_C02", K8, KELLY,   5),
    ("COMO_C03", K8, KELLY,   6),
    ("COMO_C04", S8, SANFORD, 2),
    ("COMO_C05", S8, SANFORD, 4),
    ("COMO_C06", S8, SANFORD, 5),
]:
    answer = ch.handler_comorbidity_count(
        raw, {"patient_id": full_pid, "wave": wave}
    )
    rows.append(_row(
        qid, pid8, f"Wave {wave}",
        f"How many conditions were simultaneously active for patient {pid8} in Wave {wave}?",
        answer, COUNT_NOTES.get(qid, "deterministic"),
    ))

# ── SECTION B — BOOLEAN CO-OCCURRENCE (H35: handler_conditions_cooccur) ──────

bool_specs = [
    ("COMO_B01", S8, SANFORD,
        "Coronary Heart Disease", "Atrial Fibrillation"),
    ("COMO_B02", S8, SANFORD,
        "Diabetes",               "Stroke"),
    ("COMO_B03", S8, SANFORD,
        "Alzheimer's disease (disorder)", "Diabetic renal disease (disorder)"),
    ("COMO_B04", K8, KELLY,
        "Hypertension",           "Anemia (disorder)"),
    ("COMO_B05", K8, KELLY,
        "Hypertension",           "Metabolic syndrome X (disorder)"),
]

for qid, pid8, full_pid, cond_a, cond_b in bool_specs:
    raw_ans = ch.handler_conditions_cooccur(
        raw, {"patient_id": full_pid, "condition_a": cond_a, "condition_b": cond_b}
    )
    # Store full handler string; starts with Yes/No for grader prefix rule
    rows.append(_row(
        qid, pid8, "All waves",
        f"Were {cond_a} and {cond_b} ever simultaneously active for patient {pid8}?",
        raw_ans, "deterministic",
    ))

# ── SECTION C — DURATION (H36: handler_condition_duration_waves) ─────────────

dur_specs = [
    ("COMO_D01", K8, KELLY,   "Hypertension"),
    ("COMO_D02", K8, KELLY,   "Diabetes"),
    ("COMO_D03", S8, SANFORD, "Coronary Heart Disease"),
    ("COMO_D04", S8, SANFORD, "Atrial Fibrillation"),
    ("COMO_D05", S8, SANFORD, "Alzheimer's disease (disorder)"),
]

for qid, pid8, full_pid, condition in dur_specs:
    answer = ch.handler_condition_duration_waves(
        raw, {"patient_id": full_pid, "condition": condition}
    )
    rows.append(_row(
        qid, pid8, "All waves",
        f"Across how many waves was {condition} active for patient {pid8}?",
        answer, "deterministic",
    ))

# ── SECTION D — CONTRAST (computed old vs new) ───────────────────────────────

# X01, X02, X03: count — old event-based vs interval
for qid, pid8, full_pid, wave in [
    ("COMO_X01", S8, SANFORD, 6),
    ("COMO_X02", K8, KELLY,   4),
    ("COMO_X03", S8, SANFORD, 3),
]:
    old_n  = _old_event_count(full_pid, wave)
    new_ans = ch.handler_comorbidity_count(
        raw, {"patient_id": full_pid, "wave": wave}
    )
    notes = (
        f"CONTRAST: event-based={old_n} via handle_lookup_condition WAVE=={wave}; "
        f"interval={new_ans}"
    )
    rows.append(_row(
        qid, pid8, f"Wave {wave}",
        f"How many conditions were simultaneously active for patient {pid8} in Wave {wave}?",
        new_ans, notes,
    ))

# X04: capability-gap boolean — HTN + Diabetes for Kelly
x04_raw = ch.handler_conditions_cooccur(
    raw, {"patient_id": KELLY, "condition_a": "Hypertension", "condition_b": "Diabetes"}
)
# Extract wave number from handler output for the note
if "active during wave" in x04_raw:
    _wave_tok = x04_raw.split("wave")[-1].strip().rstrip(".")
else:
    _wave_tok = "?"
x04_notes = (
    "CONTRAST: old system lacks named-pair co-occurrence handler — "
    f"no handler accepts condition_a/condition_b args; "
    f"interval={x04_raw}"
)
rows.append(_row(
    "COMO_X04", K8, "All waves",
    "Were Hypertension and Diabetes ever simultaneously active for patient b118adba?",
    x04_raw, x04_notes,
))

# ── Print all 20 rows for review ──────────────────────────────────────────────

W = 92
print("=" * W)
print(f"{'COMORBIDITY BENCHMARK — 20 QUESTIONS':^{W}}")
print("=" * W)
print(f"  {'ID':<12}  {'pid8':<10}  {'wave_ref':<10}  {'answer':<42}  notes[:30]")
print(f"  {'-'*12}  {'-'*10}  {'-'*10}  {'-'*42}  {'-'*30}")
for r in rows:
    note_abbr = r["notes"] if r["notes"] == "deterministic" else "CONTRAST: " + r["notes"][10:36] + "..."
    print(
        f"  {r['question_id']:<12}  {r['patient_id']:<10}  "
        f"{r['wave_reference']:<10}  {str(r['answer']):<42}  {note_abbr[:30]}"
    )

print()
print("FULL answers and notes (all 20):")
print("-" * W)
for r in rows:
    print(f"  {r['question_id']:<12}  answer={r['answer']!r}")
    if r["notes"] != "deterministic":
        print(f"  {'':12}  notes ={r['notes']!r}")

# ── Write CSV ─────────────────────────────────────────────────────────────────

print()
print(f"Writing -> {OUT}")
with open(OUT, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=COLS)
    writer.writeheader()
    writer.writerows(rows)
print(f"Done — {len(rows)} rows written to qa_pairs_comorbidity.csv")
