"""
text_to_pandas_fix.py
Wave-aware QA pipeline: Groq extracts a structured query spec from the question,
then a category-specific pandas handler computes the answer deterministically.
No retrieved chunks — the handler mirrors generate_qa_deterministic.py exactly.
"""

import os
import sys
import json
import time
import pandas as pd
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import data_utils
import eval_harness

_HERE = os.path.dirname(os.path.abspath(__file__))
BASE  = os.path.dirname(_HERE)
BENCH = os.path.join(BASE, "benchmark")

# ── Load tables once at module level ──────────────────────────────────────────
patients, conditions, observations, medications = data_utils.load_tables()

# Pre-built lookup for voice-path patient resolution (name, birth_year, pid8)
_known_patients = data_utils.build_patient_lookup(patients)


def _full_pid(pid_prefix: str) -> str | None:
    """Resolve an 8-char patient prefix to the full UUID, or None if not found."""
    matches = patients[patients["Id"].str[:8] == pid_prefix]
    return matches["Id"].iloc[0] if not matches.empty else None


def _wave_from_year(year) -> int:
    """Wave number from calendar year — mirrors data_utils.load_tables wave assignment."""
    return max(1, (int(year) - 1990) // 5 + 1)


def _med_waves(full_pid: str) -> "pd.Series":
    """Return a Series of wave numbers for a patient's medications (computed on demand)."""
    pm = medications[medications["PATIENT"] == full_pid].copy()
    if pm.empty:
        return pd.Series(dtype=float)
    if "WAVE" in pm.columns:
        return pm["WAVE"].dropna()
    return (
        pd.to_datetime(pm["START"], errors="coerce")
        .dt.year.dropna()
        .apply(_wave_from_year)
    )


# ── Groq extraction ────────────────────────────────────────────────────────────
groq_client = Groq()
GROQ_MODEL  = "llama-3.1-8b-instant"

_EXTRACTION_PROMPT = """You are a query parser for a clinical database. Given a patient question, output ONLY a valid JSON object (no markdown, no explanation) with these fields:

- "category": one of exactly: lookup_condition, lookup_observation, lookup_demographics, trend, aggregate_condition_count, aggregate_observation_count, aggregate_distinct_waves, multihop_hypertension, multihop_bmi_obesity, peak_bmi_wave, most_medication_wave, condition_free_waves, chronic_vs_acute, observation_density, record_span, bmi_overweight_waves, peak_comorbidity, medication_free_waves, heart_rate_range, bmi_overall_trend, longest_persisting_condition, new_diagnosis_waves, bmi_at_first_diagnosis, medication_timing, conditions_after_medication, age_at_first_chronic, meds_at_peak_bmi
- "patient_id": the 8-character patient ID string found in the question
- category-specific fields (integers for wave numbers, string for observation_name):
  - lookup_condition: wave
  - lookup_observation: wave, observation_name
  - lookup_demographics: (no extra fields)
  - trend: wave_a, wave_b, observation_name
  - aggregate_condition_count: (no extra fields)
  - aggregate_observation_count: wave
  - aggregate_distinct_waves: (no extra fields)
  - multihop_hypertension: (no extra fields)
  - multihop_bmi_obesity: wave_a, wave_b
  - All new v2 categories below: patient_id only, NO extra fields needed:
    peak_bmi_wave, most_medication_wave, condition_free_waves, chronic_vs_acute,
    observation_density, record_span, bmi_overweight_waves, peak_comorbidity,
    medication_free_waves, heart_rate_range, bmi_overall_trend,
    longest_persisting_condition, new_diagnosis_waves, bmi_at_first_diagnosis,
    medication_timing, conditions_after_medication, age_at_first_chronic, meds_at_peak_bmi

IMPORTANT — lookup_demographics vs lookup_observation:
  "lookup_demographics" is ONLY for questions about gender and birth year (no wave).
  Any question that asks about height, weight, BMI, blood pressure, or heart rate —
  even when phrased as "how tall", "how heavy", "what's their BMI" — is
  "lookup_observation" when it mentions a wave number, or "trend" when comparing
  two waves. The presence of a wave number is the key signal.

For observation_name, use ONLY these exact strings — never abbreviations, informal terms, or partial names:
  "Systolic Blood Pressure"  (not "systolic BP", "blood pressure", "BP")
  "Diastolic Blood Pressure" (not "diastolic BP")
  "Body Mass Index"          (not "BMI", "body mass")
  "Body Weight"              (not "weight", "weight readings")
  "Body Height"              (not "height", "how tall", "Height" with capital H only)
  "Heart rate"               (not "heart rate", "HR", "pulse")

For trend: wave_a is ALWAYS the numerically smaller (earlier) wave, wave_b the larger
(later) wave — even when the question mentions wave_b first (e.g. "wave six compared
to wave five" -> wave_a=5, wave_b=6). Trend ONLY applies when TWO WAVE NUMBERS are
present in the question. If the comparison is "before vs after a diagnosis event"
with no wave numbers, use multihop_hypertension or multihop_bmi_obesity instead.

Examples:

Q: "What condition(s) were diagnosed in Wave 5 for patient 9358d6df?"
{"category": "lookup_condition", "patient_id": "9358d6df", "wave": 5}

Q: "What was the average recorded Body Mass Index in Wave 5 for patient 25d6cb75?"
{"category": "lookup_observation", "patient_id": "25d6cb75", "wave": 5, "observation_name": "Body Mass Index"}

Q: "What was the average recorded Systolic Blood Pressure in Wave 6 for patient da44a96f?"
{"category": "lookup_observation", "patient_id": "da44a96f", "wave": 6, "observation_name": "Systolic Blood Pressure"}

Q: "What was the average recorded Heart rate in Wave 5 for patient 10134dbf?"
{"category": "lookup_observation", "patient_id": "10134dbf", "wave": 5, "observation_name": "Heart rate"}

Q: "How tall was patient fc4aa89c on average in wave six, going by their measurements?"
{"category": "lookup_observation", "patient_id": "fc4aa89c", "wave": 6, "observation_name": "Body Height"}

Q: "What is the gender and birth year of patient 9ad2f1f3?"
{"category": "lookup_demographics", "patient_id": "9ad2f1f3"}

Q: "How did the average Body Mass Index change from Wave 6 to Wave 7 for patient 69b60335?"
{"category": "trend", "patient_id": "69b60335", "wave_a": 6, "wave_b": 7, "observation_name": "Body Mass Index"}

Q: "How did the average Systolic Blood Pressure change from Wave 2 to Wave 4 for patient defd47cc?"
{"category": "trend", "patient_id": "defd47cc", "wave_a": 2, "wave_b": 4, "observation_name": "Systolic Blood Pressure"}

Q: "Was there any difference in patient ba756dea's Body Height during wave six compared to wave five?"
{"category": "trend", "patient_id": "ba756dea", "wave_a": 5, "wave_b": 6, "observation_name": "Body Height"}

Q: "How many unique conditions were recorded in total for patient 199946d9?"
{"category": "aggregate_condition_count", "patient_id": "199946d9"}

Q: "How many observations were recorded in Wave 5 for patient f7d7b580?"
{"category": "aggregate_observation_count", "patient_id": "f7d7b580", "wave": 5}

Q: "Across how many different waves were conditions recorded for patient 034e9e3b?"
{"category": "aggregate_distinct_waves", "patient_id": "034e9e3b"}

Q: "Was the average Systolic Blood Pressure higher after patient affb8a9e was diagnosed with Hypertension compared to before?"
{"category": "multihop_hypertension", "patient_id": "affb8a9e"}

Q: "Patient c3d53676 has hypertension — when you compare their systolic BP before the diagnosis to after, which period had the higher readings?"
{"category": "multihop_hypertension", "patient_id": "c3d53676"}

Q: "Did Body Mass Index increase between Wave 4 and Wave 6 for patient 690e0ead, and were they diagnosed with Obesity during that period?"
{"category": "multihop_bmi_obesity", "patient_id": "690e0ead", "wave_a": 4, "wave_b": 6}

Q: "In which wave did patient 0047123f record their highest average Body Mass Index?"
{"category": "peak_bmi_wave", "patient_id": "0047123f"}

Q: "In which wave was patient 00185faa prescribed the most unique medications?"
{"category": "most_medication_wave", "patient_id": "00185faa"}

Q: "How many waves did patient 023a7d29 have NO new conditions diagnosed?"
{"category": "condition_free_waves", "patient_id": "023a7d29"}

Q: "How many of patient 0149d553's ever-diagnosed conditions were chronic (no recorded stop date)?"
{"category": "chronic_vs_acute", "patient_id": "0149d553"}

Q: "What was the average number of observations per wave for patient 0042862c?"
{"category": "observation_density", "patient_id": "0042862c"}

Q: "How many years does patient 6abdc54d's health record span from first to last recorded event?"
{"category": "record_span", "patient_id": "6abdc54d"}

Q: "In how many waves was patient 010d4a3a's average BMI in the overweight range (25-30 kg/m2)?"
{"category": "bmi_overweight_waves", "patient_id": "010d4a3a"}

Q: "What is the maximum number of conditions patient 04db6603 had active simultaneously in any single wave?"
{"category": "peak_comorbidity", "patient_id": "04db6603"}

Q: "How many waves did patient 04630e85 appear in with NO medications at all?"
{"category": "medication_free_waves", "patient_id": "04630e85"}

Q: "What was the range of average heart rates recorded for patient 064341a0 across waves (minimum to maximum wave average)?"
{"category": "heart_rate_range", "patient_id": "064341a0"}

Q: "Did patient 0707e55b's BMI overall increase, decrease, or remain stable across all waves?"
{"category": "bmi_overall_trend", "patient_id": "0707e55b"}

Q: "Which condition was active for patient 052c405b across the most number of waves?"
{"category": "longest_persisting_condition", "patient_id": "052c405b"}

Q: "In how many waves did patient 058590f8 receive at least one entirely new diagnosis not seen in prior waves?"
{"category": "new_diagnosis_waves", "patient_id": "058590f8"}

Q: "What was patient 03172f6e's BMI in the wave they were first ever diagnosed with any condition?"
{"category": "bmi_at_first_diagnosis", "patient_id": "03172f6e"}

Q: "Was patient 034e9e3b's first medication prescribed before or after their first condition diagnosis?"
{"category": "medication_timing", "patient_id": "034e9e3b"}

Q: "How many new conditions were diagnosed for patient 05bfc523 after they started their first medication?"
{"category": "conditions_after_medication", "patient_id": "05bfc523"}

Q: "How old was patient 0606d624 when they were first diagnosed with a chronic condition (no stop date)?"
{"category": "age_at_first_chronic", "patient_id": "0606d624"}

Q: "How many medications was patient 03963166 taking in the wave when their BMI was highest?"
{"category": "meds_at_peak_bmi", "patient_id": "03963166"}

Now parse this question:
Q: "{question}"
"""


def extract_spec(question: str, retries: int = 3) -> dict | None:
    """Call Groq to extract a structured query spec from the question."""
    prompt = _EXTRACTION_PROMPT.replace("{question}", question)
    for attempt in range(retries):
        try:
            resp = groq_client.chat.completions.create(
                model=GROQ_MODEL,
                max_tokens=200,
                temperature=0,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = resp.choices[0].message.content.strip()
            # Strip markdown code fences if present
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            return json.loads(raw.strip())
        except json.JSONDecodeError:
            return None
        except Exception as exc:
            if attempt < retries - 1:
                wait = 2 ** attempt * 4
                time.sleep(wait)
            else:
                return None
    return None


# ── Category handlers ──────────────────────────────────────────────────────────
# Each mirrors the corresponding answer logic in generate_qa_deterministic.py.

def handle_lookup_condition(spec: dict) -> str:
    full_pid = _full_pid(spec["patient_id"])
    if full_pid is None:
        return f"Patient {spec['patient_id']} not found."
    wave = int(spec["wave"])
    sub = conditions[(conditions["PATIENT"] == full_pid) & (conditions["WAVE"] == wave)]
    if sub.empty:
        return f"No conditions found in Wave {wave} for patient {spec['patient_id']}."
    return "; ".join(sorted(sub["DESCRIPTION"].unique()))


def handle_lookup_observation(spec: dict) -> str:
    full_pid = _full_pid(spec["patient_id"])
    if full_pid is None:
        return f"Patient {spec['patient_id']} not found."
    wave     = int(spec["wave"])
    obs_name = spec["observation_name"]
    sub = observations[
        (observations["PATIENT"] == full_pid) &
        (observations["WAVE"] == wave) &
        (observations["DESCRIPTION"] == obs_name)
    ]
    if sub.empty:
        return f"No {obs_name} readings in Wave {wave} for patient {spec['patient_id']}."
    val   = round(float(pd.to_numeric(sub["VALUE"], errors="coerce").mean()), 1)
    units = sub["UNITS"].iloc[0] if pd.notna(sub["UNITS"].iloc[0]) else ""
    return f"{val} {units}".strip()


def handle_lookup_demographics(spec: dict) -> str:
    pid_prefix = spec["patient_id"]
    row = patients[patients["Id"].str[:8] == pid_prefix]
    if row.empty:
        return f"Patient {pid_prefix} not found."
    p          = row.iloc[0]
    gender     = "Male" if p["GENDER"] == "M" else "Female"
    birth_year = pd.to_datetime(p["BIRTHDATE"]).year
    return f"{gender}, born {birth_year}"


def handle_trend(spec: dict) -> str:
    full_pid = _full_pid(spec["patient_id"])
    if full_pid is None:
        return f"Patient {spec['patient_id']} not found."
    wa, wb   = int(spec["wave_a"]), int(spec["wave_b"])
    obs_name = spec["observation_name"]
    sub = observations[
        (observations["PATIENT"] == full_pid) &
        (observations["DESCRIPTION"] == obs_name)
    ].copy()
    sub["VALUE"] = pd.to_numeric(sub["VALUE"], errors="coerce")
    sub = sub.dropna(subset=["VALUE"])
    sub_a = sub[sub["WAVE"] == wa]
    sub_b = sub[sub["WAVE"] == wb]
    if sub_a.empty or sub_b.empty:
        return f"Insufficient data for {obs_name} in Wave {wa} or Wave {wb}."
    val_a     = round(float(sub_a["VALUE"].mean()), 1)
    val_b     = round(float(sub_b["VALUE"].mean()), 1)
    units     = sub["UNITS"].iloc[0] if pd.notna(sub["UNITS"].iloc[0]) else ""
    direction = (
        "increased"       if val_b > val_a else
        "decreased"       if val_b < val_a else
        "remained stable"
    )
    return (
        f"Average {obs_name} {direction} from {val_a} {units} (Wave {wa}) "
        f"to {val_b} {units} (Wave {wb})."
    )


def handle_aggregate_condition_count(spec: dict) -> str:
    full_pid = _full_pid(spec["patient_id"])
    if full_pid is None:
        return f"Patient {spec['patient_id']} not found."
    count = conditions[conditions["PATIENT"] == full_pid]["DESCRIPTION"].nunique()
    return str(count)


def handle_aggregate_observation_count(spec: dict) -> str:
    full_pid = _full_pid(spec["patient_id"])
    if full_pid is None:
        return f"Patient {spec['patient_id']} not found."
    wave  = int(spec["wave"])
    count = len(observations[
        (observations["PATIENT"] == full_pid) &
        (observations["WAVE"] == wave)
    ])
    return str(count)


def handle_aggregate_distinct_waves(spec: dict) -> str:
    full_pid = _full_pid(spec["patient_id"])
    if full_pid is None:
        return f"Patient {spec['patient_id']} not found."
    n_waves = int(conditions[conditions["PATIENT"] == full_pid]["WAVE"].nunique())
    return str(n_waves)


def handle_multihop_hypertension(spec: dict) -> str:
    full_pid = _full_pid(spec["patient_id"])
    if full_pid is None:
        return f"Patient {spec['patient_id']} not found."
    # Earliest Hypertension diagnosis date (matches generate_qa_deterministic sort_values)
    hyp = conditions[
        (conditions["PATIENT"] == full_pid) &
        (conditions["DESCRIPTION"] == "Hypertension")
    ].sort_values("START")
    if hyp.empty:
        return "No Hypertension diagnosis found."
    diag_date = hyp.iloc[0]["START"]
    sbp = observations[
        (observations["PATIENT"] == full_pid) &
        (observations["DESCRIPTION"] == "Systolic Blood Pressure")
    ].copy()
    sbp["VALUE"] = pd.to_numeric(sbp["VALUE"], errors="coerce")
    before = sbp[sbp["DATE"] < diag_date]["VALUE"].dropna()
    after  = sbp[sbp["DATE"] >= diag_date]["VALUE"].dropna()
    if before.empty or after.empty:
        return "Insufficient SBP data before or after diagnosis."
    avg_b  = round(float(before.mean()), 1)
    avg_a  = round(float(after.mean()), 1)
    units  = sbp["UNITS"].iloc[0] if pd.notna(sbp["UNITS"].iloc[0]) else ""
    result = "higher" if avg_a > avg_b else "lower or equal"
    return (
        f"Before diagnosis: avg SBP = {avg_b} {units}; "
        f"after diagnosis: avg SBP = {avg_a} {units}. "
        f"Post-diagnosis SBP was {result}."
    )


def handle_multihop_bmi_obesity(spec: dict) -> str:
    full_pid = _full_pid(spec["patient_id"])
    if full_pid is None:
        return f"Patient {spec['patient_id']} not found."
    wa, wb = int(spec["wave_a"]), int(spec["wave_b"])
    bmi = observations[
        (observations["PATIENT"] == full_pid) &
        (observations["DESCRIPTION"] == "Body Mass Index")
    ].copy()
    bmi["VALUE"] = pd.to_numeric(bmi["VALUE"], errors="coerce")
    bmi = bmi.dropna(subset=["VALUE"])
    sub_a = bmi[bmi["WAVE"] == wa]
    sub_b = bmi[bmi["WAVE"] == wb]
    if sub_a.empty or sub_b.empty:
        return f"Insufficient BMI data for Wave {wa} or Wave {wb}."
    val_a     = round(float(sub_a["VALUE"].mean()), 1)
    val_b     = round(float(sub_b["VALUE"].mean()), 1)
    units     = bmi["UNITS"].iloc[0] if pd.notna(bmi["UNITS"].iloc[0]) else ""
    direction = (
        "increased"       if val_b > val_a else
        "decreased"       if val_b < val_a else
        "remained stable"
    )
    # Obesity diagnosis — first occurrence, check if its wave falls in [wa, wb]
    obesity = conditions[
        (conditions["PATIENT"] == full_pid) &
        (conditions["DESCRIPTION"].str.contains("obesity", case=False, na=False))
    ].sort_values("START")
    if not obesity.empty:
        diag_wave = int(obesity.iloc[0]["WAVE"])
        during    = wa <= diag_wave <= wb
    else:
        diag_wave, during = None, False
    during_str = "was" if during else "was not"
    diag_part  = f" (Wave {diag_wave})" if diag_wave is not None else ""
    return (
        f"BMI {direction} from {val_a} {units} (Wave {wa}) to {val_b} {units} (Wave {wb}). "
        f"Patient {during_str} diagnosed with Obesity during this period{diag_part}."
    )


# ── New v2 Aggregate Handlers ──────────────────────────────────────────────────

def handle_peak_bmi_wave(spec: dict) -> str:
    full_pid = _full_pid(spec["patient_id"])
    if full_pid is None:
        return f"Patient {spec['patient_id']} not found."
    bmi = observations[
        (observations["PATIENT"] == full_pid) &
        (observations["DESCRIPTION"] == "Body Mass Index")
    ].copy()
    bmi["VALUE"] = pd.to_numeric(bmi["VALUE"], errors="coerce")
    bmi = bmi.dropna(subset=["VALUE", "WAVE"])
    if bmi.empty:
        return f"No BMI data found for patient {spec['patient_id']}."
    wave_avg  = bmi.groupby("WAVE")["VALUE"].mean()
    best_wave = int(wave_avg.idxmax())
    best_val  = round(float(wave_avg.max()), 1)
    return f"Wave {best_wave} (avg BMI = {best_val} kg/m2)"


def handle_most_medication_wave(spec: dict) -> str:
    full_pid = _full_pid(spec["patient_id"])
    if full_pid is None:
        return f"Patient {spec['patient_id']} not found."
    pm = medications[medications["PATIENT"] == full_pid].copy()
    if pm.empty:
        return f"No medication data for patient {spec['patient_id']}."
    if "WAVE" not in pm.columns:
        pm["WAVE"] = pd.to_datetime(pm["START"], errors="coerce").dt.year.apply(
            lambda y: _wave_from_year(y) if pd.notna(y) else None
        )
    pm = pm.dropna(subset=["WAVE"])
    if pm.empty:
        return f"No medication wave data for patient {spec['patient_id']}."
    pm["WAVE"] = pm["WAVE"].astype(int)
    wave_ct = pm.groupby("WAVE")["DESCRIPTION"].nunique()
    best = int(wave_ct.idxmax())
    return f"Wave {best} ({wave_ct[best]} unique medications)"


def handle_condition_free_waves(spec: dict) -> str:
    full_pid = _full_pid(spec["patient_id"])
    if full_pid is None:
        return f"Patient {spec['patient_id']} not found."
    obs_waves  = set(
        observations[observations["PATIENT"] == full_pid]["WAVE"].dropna().astype(int).unique()
    )
    cond_waves = set(
        conditions[conditions["PATIENT"] == full_pid]["WAVE"].dropna().astype(int).unique()
    )
    free = sorted(obs_waves - cond_waves)
    if not free:
        return "No condition-free waves found."
    wave_str = ", ".join(f"Wave {w}" for w in free)
    return f"{len(free)} waves with no new diagnoses ({wave_str})"


def handle_chronic_vs_acute(spec: dict) -> str:
    full_pid = _full_pid(spec["patient_id"])
    if full_pid is None:
        return f"Patient {spec['patient_id']} not found."
    pc      = conditions[conditions["PATIENT"] == full_pid]
    total   = len(pc)
    chronic = int(pc["STOP"].isna().sum())
    return f"{chronic} chronic conditions out of {total} total"


def handle_observation_density(spec: dict) -> str:
    full_pid = _full_pid(spec["patient_id"])
    if full_pid is None:
        return f"Patient {spec['patient_id']} not found."
    po = observations[observations["PATIENT"] == full_pid].dropna(subset=["WAVE"])
    if po.empty:
        return f"No observation data for patient {spec['patient_id']}."
    wave_counts = po.groupby("WAVE").size()
    total  = int(wave_counts.sum())
    n_wave = int(len(wave_counts))
    avg    = round(float(wave_counts.mean()), 1)
    return f"{avg} observations per wave (total {total} across {n_wave} waves)"


def handle_record_span(spec: dict) -> str:
    full_pid = _full_pid(spec["patient_id"])
    if full_pid is None:
        return f"Patient {spec['patient_id']} not found."
    dates = []
    for col in (
        conditions[conditions["PATIENT"] == full_pid]["START"],
        observations[observations["PATIENT"] == full_pid]["DATE"],
        medications[medications["PATIENT"] == full_pid]["START"],
    ):
        parsed = pd.to_datetime(col, errors="coerce").dropna()
        dates.extend(parsed.tolist())
    if not dates:
        return f"No data found for patient {spec['patient_id']}."
    first = min(dates)
    last  = max(dates)
    span  = last.year - first.year
    return f"{span} years ({first.year} to {last.year})"


def handle_bmi_overweight_waves(spec: dict) -> str:
    full_pid = _full_pid(spec["patient_id"])
    if full_pid is None:
        return f"Patient {spec['patient_id']} not found."
    bmi = observations[
        (observations["PATIENT"] == full_pid) &
        (observations["DESCRIPTION"] == "Body Mass Index")
    ].copy()
    bmi["VALUE"] = pd.to_numeric(bmi["VALUE"], errors="coerce")
    bmi = bmi.dropna(subset=["VALUE", "WAVE"])
    if bmi.empty:
        return f"No BMI data for patient {spec['patient_id']}."
    wave_avg   = bmi.groupby("WAVE")["VALUE"].mean()
    overweight = wave_avg[(wave_avg >= 25) & (wave_avg < 30)]
    if overweight.empty:
        return "No waves with average BMI in overweight range (25-30 kg/m2)."
    waves_str = ", ".join(f"Wave {int(w)}" for w in overweight.index)
    return f"{len(overweight)} waves ({waves_str})"


def handle_peak_comorbidity(spec: dict) -> str:
    full_pid = _full_pid(spec["patient_id"])
    if full_pid is None:
        return f"Patient {spec['patient_id']} not found."
    pc = conditions[conditions["PATIENT"] == full_pid].dropna(subset=["WAVE"])
    if pc.empty:
        return f"No condition data for patient {spec['patient_id']}."
    wave_ct = pc.groupby("WAVE")["DESCRIPTION"].nunique()
    best    = int(wave_ct.idxmax())
    return f"{wave_ct[best]} conditions in Wave {best}"


def handle_medication_free_waves(spec: dict) -> str:
    full_pid = _full_pid(spec["patient_id"])
    if full_pid is None:
        return f"Patient {spec['patient_id']} not found."
    obs_waves = set(
        observations[observations["PATIENT"] == full_pid]["WAVE"].dropna().astype(int).unique()
    )
    med_w = _med_waves(full_pid)
    med_waves = set(med_w.dropna().astype(int).unique()) if not med_w.empty else set()
    free = sorted(obs_waves - med_waves)
    return f"{len(free)} waves with no medications"


def handle_heart_rate_range(spec: dict) -> str:
    full_pid = _full_pid(spec["patient_id"])
    if full_pid is None:
        return f"Patient {spec['patient_id']} not found."
    hr = observations[
        (observations["PATIENT"] == full_pid) &
        (observations["DESCRIPTION"] == "Heart rate")
    ].copy()
    hr["VALUE"] = pd.to_numeric(hr["VALUE"], errors="coerce")
    hr = hr.dropna(subset=["VALUE", "WAVE"])
    if hr.empty:
        return f"No heart rate data for patient {spec['patient_id']}."
    wave_avg = hr.groupby("WAVE")["VALUE"].mean()
    mn_wave  = int(wave_avg.idxmin()); mn_val = round(float(wave_avg.min()), 1)
    mx_wave  = int(wave_avg.idxmax()); mx_val = round(float(wave_avg.max()), 1)
    return f"{mn_val} /min (Wave {mn_wave}) to {mx_val} /min (Wave {mx_wave})"


def handle_bmi_overall_trend(spec: dict) -> str:
    full_pid = _full_pid(spec["patient_id"])
    if full_pid is None:
        return f"Patient {spec['patient_id']} not found."
    bmi = observations[
        (observations["PATIENT"] == full_pid) &
        (observations["DESCRIPTION"] == "Body Mass Index")
    ].copy()
    bmi["VALUE"] = pd.to_numeric(bmi["VALUE"], errors="coerce")
    bmi = bmi.dropna(subset=["VALUE", "WAVE"])
    if bmi.empty:
        return f"No BMI data for patient {spec['patient_id']}."
    wave_avg = bmi.groupby("WAVE")["VALUE"].mean().sort_index()
    if len(wave_avg) < 2:
        return "Insufficient BMI data across waves."
    first_w = int(wave_avg.index[0]);  first_v = round(float(wave_avg.iloc[0]), 1)
    last_w  = int(wave_avg.index[-1]); last_v  = round(float(wave_avg.iloc[-1]), 1)
    if last_v > first_v + 0.5:        direction = "increased"
    elif last_v < first_v - 0.5:      direction = "decreased"
    else:                               direction = "remained stable"
    return f"Overall {direction} from {first_v} kg/m2 (Wave {first_w}) to {last_v} kg/m2 (Wave {last_w})"


def handle_longest_persisting_condition(spec: dict) -> str:
    full_pid = _full_pid(spec["patient_id"])
    if full_pid is None:
        return f"Patient {spec['patient_id']} not found."
    pc = conditions[conditions["PATIENT"] == full_pid].dropna(subset=["WAVE"])
    if pc.empty:
        return f"No condition data for patient {spec['patient_id']}."
    wave_counts = pc.groupby("DESCRIPTION")["WAVE"].nunique()
    best = wave_counts.idxmax()
    return f"{best} -- active across {wave_counts[best]} waves"


def handle_new_diagnosis_waves(spec: dict) -> str:
    full_pid = _full_pid(spec["patient_id"])
    if full_pid is None:
        return f"Patient {spec['patient_id']} not found."
    pc = conditions[conditions["PATIENT"] == full_pid].dropna(subset=["WAVE"]).copy()
    if pc.empty:
        return f"No condition data for patient {spec['patient_id']}."
    pc["WAVE"] = pc["WAVE"].astype(int)
    seen = set(); new_waves = 0
    all_waves = sorted(pc["WAVE"].unique())
    for w in all_waves:
        wave_descs = set(pc[pc["WAVE"] == w]["DESCRIPTION"].unique())
        if wave_descs - seen:
            new_waves += 1
        seen |= wave_descs
    return f"{new_waves} of {len(all_waves)} waves had new diagnoses"


# ── New v2 Multi-hop Handlers ──────────────────────────────────────────────────

def handle_bmi_at_first_diagnosis(spec: dict) -> str:
    full_pid = _full_pid(spec["patient_id"])
    if full_pid is None:
        return f"Patient {spec['patient_id']} not found."
    pc = conditions[conditions["PATIENT"] == full_pid].dropna(subset=["WAVE"])
    bmi = observations[
        (observations["PATIENT"] == full_pid) &
        (observations["DESCRIPTION"] == "Body Mass Index")
    ].copy()
    bmi["VALUE"] = pd.to_numeric(bmi["VALUE"], errors="coerce")
    bmi = bmi.dropna(subset=["VALUE", "WAVE"])
    if pc.empty or bmi.empty:
        return "Insufficient data."
    first_wave = int(pc["WAVE"].min())
    first_cond = pc[pc["WAVE"] == first_wave]["DESCRIPTION"].iloc[0]
    bmi_in     = bmi[bmi["WAVE"] == first_wave]
    if bmi_in.empty:
        return f"No BMI data in Wave {first_wave} (first diagnosis wave)."
    avg_bmi = round(float(bmi_in["VALUE"].mean()), 1)
    return f"BMI was {avg_bmi} kg/m2 in Wave {first_wave} (when first diagnosis made: {first_cond})"


def handle_medication_timing(spec: dict) -> str:
    full_pid = _full_pid(spec["patient_id"])
    if full_pid is None:
        return f"Patient {spec['patient_id']} not found."
    pc = conditions[conditions["PATIENT"] == full_pid].dropna(subset=["WAVE"])
    if pc.empty:
        return "No condition data."
    med_w = _med_waves(full_pid)
    if med_w.empty:
        return "No medication data."
    first_cond_wave = int(pc["WAVE"].min())
    first_med_wave  = int(med_w.min())
    first_cond = pc[pc["WAVE"] == first_cond_wave]["DESCRIPTION"].iloc[0]
    pm = medications[medications["PATIENT"] == full_pid]
    if "WAVE" in pm.columns:
        pm_w = pm.dropna(subset=["WAVE"])
        pm_w = pm_w[pm_w["WAVE"].astype(int) == first_med_wave]
    else:
        pm_w = pm
    first_med = pm_w["DESCRIPTION"].iloc[0] if not pm_w.empty else "unknown"
    if first_med_wave < first_cond_wave:
        direction = "before"
    elif first_med_wave > first_cond_wave:
        direction = "after"
    else:
        direction = "same wave as"
    return (
        f"Medication came {direction} -- "
        f"first med Wave {first_med_wave} ({str(first_med)[:40]}), "
        f"first condition Wave {first_cond_wave} ({first_cond[:40]})"
    )


def handle_conditions_after_medication(spec: dict) -> str:
    full_pid = _full_pid(spec["patient_id"])
    if full_pid is None:
        return f"Patient {spec['patient_id']} not found."
    pc = conditions[conditions["PATIENT"] == full_pid].dropna(subset=["WAVE"])
    med_w = _med_waves(full_pid)
    if pc.empty or med_w.empty:
        return "Insufficient data."
    pc = pc.copy(); pc["WAVE"] = pc["WAVE"].astype(int)
    first_med_wave   = int(med_w.min())
    before_descs     = set(pc[pc["WAVE"] <= first_med_wave]["DESCRIPTION"].unique())
    after_descs      = set(pc[pc["WAVE"] >  first_med_wave]["DESCRIPTION"].unique())
    new_conds        = len(after_descs - before_descs)
    return f"{new_conds} new conditions after Wave {first_med_wave} (when first medication started)"


def handle_age_at_first_chronic(spec: dict) -> str:
    full_pid = _full_pid(spec["patient_id"])
    if full_pid is None:
        return f"Patient {spec['patient_id']} not found."
    pc      = conditions[conditions["PATIENT"] == full_pid]
    chronic = pc[pc["STOP"].isna()].dropna(subset=["START"])
    if chronic.empty:
        return "No chronic conditions found."
    pat_row = patients[patients["Id"] == full_pid]
    if pat_row.empty:
        return "Patient demographics not found."
    birth_year  = pd.to_datetime(pat_row.iloc[0]["BIRTHDATE"]).year
    first_chron = chronic.sort_values("START").iloc[0]
    diag_year   = pd.to_datetime(first_chron["START"]).year
    cond_name   = first_chron["DESCRIPTION"]
    try:
        cond_wave = int(first_chron["WAVE"])
    except (ValueError, TypeError, KeyError):
        cond_wave = "?"
    age = diag_year - birth_year
    return f"Age {age} (born {birth_year}, diagnosed {diag_year} in Wave {cond_wave}: {str(cond_name)[:40]})"


def handle_meds_at_peak_bmi(spec: dict) -> str:
    full_pid = _full_pid(spec["patient_id"])
    if full_pid is None:
        return f"Patient {spec['patient_id']} not found."
    bmi = observations[
        (observations["PATIENT"] == full_pid) &
        (observations["DESCRIPTION"] == "Body Mass Index")
    ].copy()
    bmi["VALUE"] = pd.to_numeric(bmi["VALUE"], errors="coerce")
    bmi = bmi.dropna(subset=["VALUE", "WAVE"])
    if bmi.empty:
        return f"No BMI data for patient {spec['patient_id']}."
    wave_avg  = bmi.groupby("WAVE")["VALUE"].mean()
    peak_wave = int(wave_avg.idxmax())
    peak_bmi  = round(float(wave_avg.max()), 1)
    pm        = medications[medications["PATIENT"] == full_pid].copy()
    if pm.empty:
        return f"0 medications in Wave {peak_wave} (peak BMI wave, avg BMI = {peak_bmi} kg/m2)"
    if "WAVE" not in pm.columns:
        pm["WAVE"] = pd.to_datetime(pm["START"], errors="coerce").dt.year.apply(
            lambda y: _wave_from_year(y) if pd.notna(y) else None
        )
    pm = pm.dropna(subset=["WAVE"])
    pm["WAVE"] = pm["WAVE"].astype(int)
    n_meds = int((pm["WAVE"] == peak_wave).sum())
    return f"{n_meds} medications in Wave {peak_wave} (peak BMI wave, avg BMI = {peak_bmi} kg/m2)"


# ── Dispatcher ─────────────────────────────────────────────────────────────────
_HANDLERS = {
    # Original handlers
    "lookup_condition":             handle_lookup_condition,
    "lookup_observation":           handle_lookup_observation,
    "lookup_demographics":          handle_lookup_demographics,
    "trend":                        handle_trend,
    "aggregate_condition_count":    handle_aggregate_condition_count,
    "aggregate_observation_count":  handle_aggregate_observation_count,
    "aggregate_distinct_waves":     handle_aggregate_distinct_waves,
    "multihop_hypertension":        handle_multihop_hypertension,
    "multihop_bmi_obesity":         handle_multihop_bmi_obesity,
    # v2 aggregate handlers
    "peak_bmi_wave":                handle_peak_bmi_wave,
    "most_medication_wave":         handle_most_medication_wave,
    "condition_free_waves":         handle_condition_free_waves,
    "chronic_vs_acute":             handle_chronic_vs_acute,
    "observation_density":          handle_observation_density,
    "record_span":                  handle_record_span,
    "bmi_overweight_waves":         handle_bmi_overweight_waves,
    "peak_comorbidity":             handle_peak_comorbidity,
    "medication_free_waves":        handle_medication_free_waves,
    "heart_rate_range":             handle_heart_rate_range,
    "bmi_overall_trend":            handle_bmi_overall_trend,
    "longest_persisting_condition": handle_longest_persisting_condition,
    "new_diagnosis_waves":          handle_new_diagnosis_waves,
    # v2 multi-hop handlers
    "bmi_at_first_diagnosis":       handle_bmi_at_first_diagnosis,
    "medication_timing":            handle_medication_timing,
    "conditions_after_medication":  handle_conditions_after_medication,
    "age_at_first_chronic":         handle_age_at_first_chronic,
    "meds_at_peak_bmi":             handle_meds_at_peak_bmi,
}


def text_to_pandas_answer(question: str) -> str:
    """Extract a query spec via Groq, then compute the answer with pandas."""
    time.sleep(0.5)    # stay inside Groq rate limits
    spec = extract_spec(question)
    if spec is None:
        return "ERROR: spec extraction failed"
    category = spec.get("category", "")
    handler  = _HANDLERS.get(category)
    if handler is None:
        return f"ERROR: unknown category '{category}'"
    try:
        return handler(spec)
    except Exception as exc:
        return f"ERROR: {exc}"


# ── Voice extraction path ─────────────────────────────────────────────────────
# Parallel to the text path but parses patient_name + birth_year instead of
# patient_id.  The existing handlers are reused unchanged after name resolution.

_VOICE_EXTRACTION_PROMPT = """You are a query parser for a clinical database.
Voice queries identify patients by FIRST NAME and BIRTH YEAR, not by an internal ID.
Output ONLY a valid JSON object (no markdown, no explanation) with these fields:

- "category": one of exactly: lookup_condition, lookup_observation, lookup_demographics, trend, aggregate_condition_count, aggregate_observation_count, aggregate_distinct_waves, multihop_hypertension, multihop_bmi_obesity
- "patient_name": the patient's first name exactly as spoken (preserve spelling/capitalisation from the question)
- "birth_year": the patient's birth year as an integer (e.g. 1989, not "born 1989"; if spoken as "nineteen eighty-nine" convert to 1989)
- category-specific fields (integers for wave numbers, string for observation_name):
  - lookup_condition: wave
  - lookup_observation: wave, observation_name
  - lookup_demographics: (no extra fields beyond patient_name and birth_year)
  - trend: wave_a, wave_b, observation_name
  - aggregate_condition_count: (no extra fields)
  - aggregate_observation_count: wave
  - aggregate_distinct_waves: (no extra fields)
  - multihop_hypertension: (no extra fields)
  - multihop_bmi_obesity: wave_a, wave_b

IMPORTANT — lookup_demographics vs lookup_observation:
  "lookup_demographics" is ONLY for questions about gender and birth year (no wave).
  Any question about height, weight, BMI, blood pressure, or heart rate with a wave
  number is "lookup_observation"; comparing two waves is "trend".

For observation_name, use ONLY these exact strings:
  "Systolic Blood Pressure"  (not "systolic BP", "blood pressure", "BP")
  "Diastolic Blood Pressure" (not "diastolic BP")
  "Body Mass Index"          (not "BMI", "body mass")
  "Body Weight"              (not "weight", "weight readings")
  "Body Height"              (not "height", "how tall")
  "Heart rate"               (not "heart rate", "HR", "pulse")

For trend: wave_a is ALWAYS the numerically smaller wave, wave_b the larger — even
when the question mentions the larger wave first. Trend ONLY applies when TWO WAVE
NUMBERS are present. "Before vs after a diagnosis event" with no wave numbers ->
use multihop_hypertension or multihop_bmi_obesity.

Examples:

Q: "What condition(s) were diagnosed in Wave 5 for Stewart, born 1993?"
{"category": "lookup_condition", "patient_name": "Stewart", "birth_year": 1993, "wave": 5}

Q: "What was the average recorded Body Mass Index in Wave 5 for Wilson, born 2009?"
{"category": "lookup_observation", "patient_name": "Wilson", "birth_year": 2009, "wave": 5, "observation_name": "Body Mass Index"}

Q: "What was the average recorded Systolic Blood Pressure in Wave 6 for Colin, born 1960?"
{"category": "lookup_observation", "patient_name": "Colin", "birth_year": 1960, "wave": 6, "observation_name": "Systolic Blood Pressure"}

Q: "What was the average recorded Heart rate in Wave 5 for Ali, born 1958?"
{"category": "lookup_observation", "patient_name": "Ali", "birth_year": 1958, "wave": 5, "observation_name": "Heart rate"}

Q: "How tall was Grady, born 1981, on average in wave six?"
{"category": "lookup_observation", "patient_name": "Grady", "birth_year": 1981, "wave": 6, "observation_name": "Body Height"}

Q: "What is the gender and birth year of Latrina, born 2002?"
{"category": "lookup_demographics", "patient_name": "Latrina", "birth_year": 2002}

Q: "How did the average Body Mass Index change from Wave 6 to Wave 7 for Izola, born 1956?"
{"category": "trend", "patient_name": "Izola", "birth_year": 1956, "wave_a": 6, "wave_b": 7, "observation_name": "Body Mass Index"}

Q: "How did the average Systolic Blood Pressure change from Wave 2 to Wave 4 for Morton, born 1939?"
{"category": "trend", "patient_name": "Morton", "birth_year": 1939, "wave_a": 2, "wave_b": 4, "observation_name": "Systolic Blood Pressure"}

Q: "Was there any difference in Bao, born 1943's Body Height during wave six compared to wave five?"
{"category": "trend", "patient_name": "Bao", "birth_year": 1943, "wave_a": 5, "wave_b": 6, "observation_name": "Body Height"}

Q: "How many unique conditions were recorded in total for Lasandra, born 2017?"
{"category": "aggregate_condition_count", "patient_name": "Lasandra", "birth_year": 2017}

Q: "How many observations were recorded in Wave 5 for Micki, born 1975?"
{"category": "aggregate_observation_count", "patient_name": "Micki", "birth_year": 1975, "wave": 5}

Q: "Across how many different waves were conditions recorded for Douglas, born 1962?"
{"category": "aggregate_distinct_waves", "patient_name": "Douglas", "birth_year": 1962}

Q: "Was the average Systolic Blood Pressure higher after Marta, born 1995, was diagnosed with Hypertension compared to before?"
{"category": "multihop_hypertension", "patient_name": "Marta", "birth_year": 1995}

Q: "Joi, born 1994, has hypertension — when you compare their systolic BP before the diagnosis to after, which period had the higher readings?"
{"category": "multihop_hypertension", "patient_name": "Joi", "birth_year": 1994}

Q: "Did Body Mass Index increase between Wave 4 and Wave 6 for Cleotilde, born 1966, and were they diagnosed with Obesity during that period?"
{"category": "multihop_bmi_obesity", "patient_name": "Cleotilde", "birth_year": 1966, "wave_a": 4, "wave_b": 6}

Now parse this question:
Q: "{question}"
"""


def extract_spec_voice(question: str, retries: int = 3) -> dict | None:
    """Like extract_spec but returns patient_name + birth_year instead of patient_id."""
    prompt = _VOICE_EXTRACTION_PROMPT.replace("{question}", question)
    for attempt in range(retries):
        try:
            resp = groq_client.chat.completions.create(
                model=GROQ_MODEL,
                max_tokens=200,
                temperature=0,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = resp.choices[0].message.content.strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            return json.loads(raw.strip())
        except json.JSONDecodeError:
            return None
        except Exception:
            if attempt < retries - 1:
                time.sleep(2 ** attempt * 4)
            else:
                return None
    return None


def text_to_pandas_answer_voice(question: str) -> str:
    """
    Voice path: extract patient_name + birth_year via Groq, resolve to patient_id
    via data_utils.resolve_patient, then dispatch to the existing category handlers.

    The text-path benchmark (text_to_pandas_answer) is completely unchanged.
    """
    time.sleep(0.5)
    spec = extract_spec_voice(question)
    if spec is None:
        return "ERROR: voice spec extraction failed"

    name = spec.get("patient_name", "")
    year = spec.get("birth_year")
    pid  = data_utils.resolve_patient(name, year, _known_patients)
    if pid is None:
        return f"ERROR: could not identify patient '{name}', born {year}"

    # Build a spec the existing handlers understand
    resolved = {k: v for k, v in spec.items() if k not in ("patient_name", "birth_year")}
    resolved["patient_id"] = pid

    category = resolved.get("category", "")
    handler  = _HANDLERS.get(category)
    if handler is None:
        return f"ERROR: unknown category '{category}'"
    try:
        return handler(resolved)
    except Exception as exc:
        return f"ERROR: {exc}"


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    bench    = pd.read_csv(os.path.join(BENCH, "qa_pairs_final.csv"))
    log_path = os.path.join(BENCH, "text_to_pandas_failure_log.csv")

    print(f"Benchmark: {len(bench)} questions  |  model: {GROQ_MODEL} (spec extraction only)")
    print("Tables loaded from data_utils.load_tables()")
    print()

    _, summary = eval_harness.run_eval(
        bench, text_to_pandas_answer, log_path, "text_to_pandas_fix"
    )
