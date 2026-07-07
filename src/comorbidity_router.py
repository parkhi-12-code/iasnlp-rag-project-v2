"""
comorbidity_router.py
----------------------
Deterministic (regex + dataset-grounded) spec extraction for the 3 comorbidity
categories, sitting IN FRONT of the LLM extraction path in text_to_pandas_fix.py.

Why deterministic, not LLM: the comorbidity questions are template-generated
with fixed phrasing, drawing condition names verbatim from
conditions["DESCRIPTION"].unique(). The condition vocabulary is known and finite,
so a generative extractor adds a failure mode (a plausible-but-wrong condition
name) for no benefit. Matching is exact-match against the real vocabulary, not
substring/fuzzy — this is what prevents collisions like "Diabetes" vs
"Prediabetes" vs the six "... type 2/II diabetes mellitus (disorder)" complication
names, all of which contain "diabetes" as a case-insensitive substring.

extract_comorbidity_spec() returns None (never a guess) when a question doesn't
match one of the 3 fixed templates, or when a matched condition span isn't an
exact member of the known vocabulary — the caller falls through to the existing
LLM path unchanged.
"""

import os
import re
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import data_utils
import comorbidity_handlers as ch

# ── Load data once (mirrors text_to_pandas_fix.py's own module-level load) ────
patients, conditions, _observations, _medications = data_utils.load_tables()

KNOWN_CONDITIONS = set(conditions["DESCRIPTION"].unique())


def _full_pid(pid_prefix: str) -> str | None:
    """Resolve an 8-char patient prefix to the full UUID, or None if not found."""
    matches = patients[patients["Id"].str[:8] == pid_prefix]
    return matches["Id"].iloc[0] if not matches.empty else None


# ── Fixed-template regexes (match generate_comorbidity_benchmark.py exactly) ──
_COUNT_RE = re.compile(
    r'^How many conditions were simultaneously active for patient '
    r'([0-9a-f]{8}) in Wave (\d+)\??$',
    re.IGNORECASE,
)
_COOCCUR_RE = re.compile(
    r'^Were (.+) ever simultaneously active for patient ([0-9a-f]{8})\??$',
    re.IGNORECASE,
)
_DURATION_RE = re.compile(
    r'^Across how many waves was (.+) active for patient ([0-9a-f]{8})\??$',
    re.IGNORECASE,
)


def _split_two_conditions(span: str) -> list[tuple[str, str]]:
    """
    Split "condition_a and condition_b" at every ' and ' occurrence, keeping
    only splits where BOTH halves are EXACT members of KNOWN_CONDITIONS.

    Exact-match validation (not longest-substring guessing) is what makes this
    safe even against a condition name that itself contains ' and ' inside it
    (e.g. "Macular edema and retinopathy due to type 2 diabetes mellitus
    (disorder)") — only the true split point produces two real condition names.
    """
    idxs = [m.start() for m in re.finditer(r' and ', span)]
    valid = []
    for i in idxs:
        left, right = span[:i].strip(), span[i + 5:].strip()
        if left in KNOWN_CONDITIONS and right in KNOWN_CONDITIONS:
            valid.append((left, right))
    return valid


def extract_comorbidity_spec(question: str) -> dict | None:
    """
    Pure regex + dataset lookup. No LLM call. Returns a spec dict shaped for
    the wrapper handlers below, with patient_id already resolved to the full
    UUID, or None if the question doesn't match one of the 3 fixed templates
    (or a matched condition span fails exact-match validation).
    """
    q = question.strip()

    m = _COUNT_RE.match(q)
    if m:
        pid8, wave = m.group(1), int(m.group(2))
        full_pid = _full_pid(pid8)
        if full_pid is None:
            return None
        return {"category": "comorbidity_count", "patient_id": full_pid, "wave": wave}

    m = _COOCCUR_RE.match(q)
    if m:
        span, pid8 = m.group(1), m.group(2)
        splits = _split_two_conditions(span)
        if len(splits) != 1:
            return None  # zero or ambiguous splits -- fail visibly, don't guess
        full_pid = _full_pid(pid8)
        if full_pid is None:
            return None
        cond_a, cond_b = splits[0]
        return {
            "category": "conditions_cooccur",
            "patient_id": full_pid,
            "condition_a": cond_a,
            "condition_b": cond_b,
        }

    m = _DURATION_RE.match(q)
    if m:
        cond, pid8 = m.group(1).strip(), m.group(2)
        if cond not in KNOWN_CONDITIONS:
            return None  # not an exact match -- fail visibly, don't guess
        full_pid = _full_pid(pid8)
        if full_pid is None:
            return None
        return {
            "category": "condition_duration_waves",
            "patient_id": full_pid,
            "condition": cond,
        }

    return None


# ── Thin wrappers delegating to H35/H36/H37 in comorbidity_handlers.py ───────

def handle_comorbidity_count(spec: dict) -> str:
    return ch.handler_comorbidity_count(
        conditions, {"patient_id": spec["patient_id"], "wave": spec["wave"]}
    )


def handle_conditions_cooccur(spec: dict) -> str:
    return ch.handler_conditions_cooccur(
        conditions,
        {
            "patient_id": spec["patient_id"],
            "condition_a": spec["condition_a"],
            "condition_b": spec["condition_b"],
        },
    )


def handle_condition_duration_waves(spec: dict) -> str:
    return ch.handler_condition_duration_waves(
        conditions, {"patient_id": spec["patient_id"], "condition": spec["condition"]}
    )
