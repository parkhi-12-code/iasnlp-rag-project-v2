"""
comorbidity_handlers.py
-----------------------
Interval-overlap logic + T2P handlers for temporally CO-ACTIVE conditions.
Solves the 0%/0% comorbidity gap (Finding F4).

Core idea (plain language):
  Your current system asks "was condition X DIAGNOSED in wave W?"
  (an event — a single point in time). The comorbidity questions ask
  "was condition X ACTIVE during wave W?" (an interval — a span of time).
  A condition diagnosed in W2 with no STOP date is still active in W6,
  even though no row 'happens' in W6. That is why both KG and T2P
  scored 0%: the representation was event-based, not interval-based.

The fix is one rule — the standard interval-overlap test:

  ACTIVE in [wave_start, wave_end]  <=>
      START <= wave_end  AND  (STOP is null OR STOP >= wave_start)

Everything below is built on that single rule.
Place this file in src/ on the future-work branch. Do NOT modify
text_to_pandas_fix.py yet — import from here first, integrate later.
"""

import pandas as pd

# ── Wave boundaries (must match your existing wave definitions) ──────────
WAVES = {
    1: ("1990-01-01", "1994-12-31"),
    2: ("1995-01-01", "1999-12-31"),
    3: ("2000-01-01", "2004-12-31"),
    4: ("2005-01-01", "2009-12-31"),
    5: ("2010-01-01", "2014-12-31"),
    6: ("2015-01-01", "2019-12-31"),
    7: ("2020-01-01", "2099-12-31"),   # open-ended final wave
}


def _prep_conditions(conditions_df: pd.DataFrame) -> pd.DataFrame:
    """Parse dates once. STOP=NaT means the condition is still active."""
    df = conditions_df.copy()
    df["START"] = pd.to_datetime(df["START"], errors="coerce")
    df["STOP"] = pd.to_datetime(df["STOP"], errors="coerce")
    return df


# ── THE CORE FUNCTION — everything else calls this ───────────────────────
def active_conditions_in_wave(conditions_df: pd.DataFrame,
                              patient_id: str,
                              wave: int) -> pd.DataFrame:
    """
    Return every condition row that was ACTIVE (not merely diagnosed)
    for `patient_id` at any point during `wave`.

    Deterministic: output is sorted by (DESCRIPTION, START).
    """
    wave_start = pd.Timestamp(WAVES[wave][0])
    wave_end = pd.Timestamp(WAVES[wave][1])

    df = _prep_conditions(conditions_df)
    df = df[df["PATIENT"] == patient_id]

    overlap = (df["START"] <= wave_end) & (
        df["STOP"].isna() | (df["STOP"] >= wave_start)
    )
    # sorted() lesson applied: deterministic ordering, always.
    return df[overlap].sort_values(["DESCRIPTION", "START"]).reset_index(drop=True)


# ── HANDLER 34: list co-active conditions in a wave ──────────────────────
def handler_coactive_conditions(conditions_df, spec: dict) -> str:
    """
    Q template: "Which conditions were active for patient P during wave W?"
    spec = {"patient_id": ..., "wave": ...}
    """
    rows = active_conditions_in_wave(conditions_df, spec["patient_id"], spec["wave"])
    names = sorted(rows["DESCRIPTION"].unique())
    if not names:
        return "No active conditions in this wave."
    return ", ".join(names)


# ── HANDLER 35: did two conditions co-occur (overlap in time)? ───────────
def handler_conditions_cooccur(conditions_df, spec: dict) -> str:
    """
    Q template: "Were condition A and condition B ever active at the same
    time for patient P?" (checks overlap in ANY wave, or a given wave)
    spec = {"patient_id": ..., "condition_a": ..., "condition_b": ...,
            "wave": optional}
    """
    waves = [spec["wave"]] if spec.get("wave") else list(WAVES.keys())
    for w in waves:
        active = handler_coactive_conditions(
            conditions_df, {"patient_id": spec["patient_id"], "wave": w}
        )
        a = spec["condition_a"].lower() in active.lower()
        b = spec["condition_b"].lower() in active.lower()
        if a and b:
            return f"Yes — both active during wave {w}."
    return "No — the two conditions were never active simultaneously."


# ── HANDLER 36: how many waves was a condition active? ───────────────────
def handler_condition_duration_waves(conditions_df, spec: dict) -> str:
    """
    Q template: "Across how many waves was condition C active for patient P?"
    spec = {"patient_id": ..., "condition": ...}
    """
    count = 0
    for w in WAVES:
        active = handler_coactive_conditions(
            conditions_df, {"patient_id": spec["patient_id"], "wave": w}
        )
        if spec["condition"].lower() in active.lower():
            count += 1
    return str(count)


# ── HANDLER 37: comorbidity burden per wave (count of co-active) ─────────
def handler_comorbidity_count(conditions_df, spec: dict) -> str:
    """
    Q template: "How many conditions were simultaneously active for
    patient P in wave W?"
    """
    rows = active_conditions_in_wave(conditions_df, spec["patient_id"], spec["wave"])
    return str(rows["DESCRIPTION"].nunique())


# ── SANITY CHECK — run this FIRST, on the STAR patient ───────────────────
if __name__ == "__main__":
    conditions = pd.read_csv("data/raw/conditions.csv")
    STAR = "3f336702-bf73-4fc8-bd59-3ba77fd65d0d"  # Sanford861 Fritsch593

    # Expectation from your cohort analysis: Coronary HD active in 6/7 waves.
    star_id = STAR
    result = handler_condition_duration_waves(
        conditions, {"patient_id": star_id, "condition": "Coronary Heart Disease"}
    )
    print(f"Coronary HD active waves for STAR patient: {result} (expected: 6)")

    for w in range(1, 8):
        n = handler_comorbidity_count(conditions, {"patient_id": star_id, "wave": w})
        print(f"Wave {w}: {n} co-active conditions")
    # Correct expectation for this patient (born 1923): 15 chronic conditions
    # already active at Wave 1, rising to 18 by Wave 6. Columns (a) all-active
    # and (b) chronic-only diverge only in W5/W6 by the 4 acute episodes.
    #
    # WHY the naive "1->7 growth" figure was wrong:
    #   The original cohort analysis counted NEW DIAGNOSES per wave — an event-
    #   based view where a condition "belongs" to the wave it was first recorded.
    #   Interval-overlap instead asks: is this condition ACTIVE (START <= wave_end
    #   AND STOP is null or STOP >= wave_start) during the wave?  A condition
    #   diagnosed in 1965 is still active — and must be counted — in every
    #   subsequent wave.  That cumulative carry-forward is exactly the
    #   representation gap that caused KG and T2P to score 0% on comorbidity
    #   questions: both systems stored conditions as point-in-time events, so
    #   they could not see a 1965 diagnosis as "co-active" in 2015.