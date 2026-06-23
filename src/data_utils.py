"""
data_utils.py
Single source of truth for loading Synthea tables and wave assignment.
Wave 1 = 1990-94, Wave 2 = 1995-99, ..., Wave 7 = 2020+
Formula: max(1, (year - 1990) // 5 + 1)

Also provides voice-path patient resolution (name + birth year -> patient_id).
"""
import os
import re
import difflib
import pandas as pd

_HERE     = os.path.dirname(os.path.abspath(__file__))
_DATA_DIR = os.path.join(os.path.dirname(_HERE), "data", "raw")


def _assign_wave(dt) -> int:
    return max(1, (dt.year - 1990) // 5 + 1)


def load_tables(data_dir: str = None):
    """
    Load patients, conditions, observations, medications from data_dir
    (defaults to data/raw/ relative to this file's project root).

    All date columns are returned as timezone-naive datetime64.
    WAVE (int) is added to conditions, observations, and medications.

    Returns:
        (patients, conditions, observations, medications)
    """
    d = data_dir or _DATA_DIR

    patients     = pd.read_csv(os.path.join(d, "patients.csv"))
    conditions   = pd.read_csv(os.path.join(d, "conditions.csv"))
    observations = pd.read_csv(os.path.join(d, "observations.csv"))
    medications  = pd.read_csv(os.path.join(d, "medications.csv"))

    # conditions has date-only START; observations/medications have ISO timestamps with Z
    conditions["START"]  = pd.to_datetime(conditions["START"])
    observations["DATE"] = pd.to_datetime(observations["DATE"], utc=True).dt.tz_convert(None)
    medications["START"] = pd.to_datetime(medications["START"], utc=True).dt.tz_convert(None)

    observations["WAVE"] = observations["DATE"].apply(_assign_wave)
    conditions["WAVE"]   = conditions["START"].apply(_assign_wave)
    medications["WAVE"]  = medications["START"].apply(_assign_wave)

    return patients, conditions, observations, medications


# ── Voice-path patient resolution ─────────────────────────────────────────────

def _clean_name(name: str) -> str:
    """Strip Synthea's trailing numeric suffix and lowercase ('Cythia210' -> 'cythia')."""
    return re.sub(r"\d+$", "", str(name)).strip().lower()


def build_patient_lookup(patients_df: pd.DataFrame) -> list:
    """
    Build a (clean_first_name, birth_year, patient_id_prefix) list from a patients
    dataframe, suitable for passing to resolve_patient().

    Returns list of (str, int, str) tuples — one per patient row.
    """
    result = []
    for _, row in patients_df.iterrows():
        clean = _clean_name(str(row["FIRST"]))
        year  = int(pd.to_datetime(str(row["BIRTHDATE"])).year)
        pid8  = str(row["Id"])[:8]
        result.append((clean, year, pid8))
    return result


def resolve_patient(spoken_name: str, spoken_birth_year,
                    known_patients: list) -> str | None:
    """
    Resolve a spoken first name + birth year to an 8-character patient_id prefix.

    known_patients: list of (clean_first_name, birth_year, patient_id_prefix) tuples,
                    as returned by build_patient_lookup().

    Resolution order:
      1. Exact match on (clean_name, birth_year) — handles perfect STT.
      2. Fuzzy name match (difflib, cutoff=0.6) restricted to patients sharing the
         exact birth year — handles minor STT noise on the name (e.g. 'Cynthia'
         for 'Cythia', 'Alta' for 'Altha') without allowing year mismatches.

    Returns the patient_id prefix (str) or None if no confident match.
    """
    try:
        year = int(spoken_birth_year)
    except (ValueError, TypeError):
        return None

    spoken_clean = _clean_name(str(spoken_name))

    # 1. Exact match
    for name, by, pid in known_patients:
        if name == spoken_clean and by == year:
            return pid

    # 2. Fuzzy match within same birth year
    same_year = [(name, pid) for name, by, pid in known_patients if by == year]
    if not same_year:
        return None
    candidates = [name for name, _ in same_year]
    close = difflib.get_close_matches(spoken_clean, candidates, n=1, cutoff=0.6)
    if close:
        for name, pid in same_year:
            if name == close[0]:
                return pid
    return None
