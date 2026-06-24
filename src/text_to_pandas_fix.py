"""
text_to_pandas_fix.py
Wave-aware QA pipeline: Groq extracts a structured query spec from the question,
then a category-specific pandas handler computes the answer deterministically.
No retrieved chunks — the handler mirrors generate_qa_deterministic.py exactly.
"""

import os
import re
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

_EXTRACTION_PROMPT = """You are a query parser for a clinical database. Output ONLY a valid JSON object.

Fields:
- "category": one of: lookup_condition, lookup_observation, lookup_demographics, trend, aggregate_condition_count, aggregate_observation_count, aggregate_distinct_waves, multihop_hypertension, multihop_bmi_obesity, peak_bmi_wave, most_medication_wave, condition_free_waves, chronic_vs_acute, observation_density, record_span, bmi_overweight_waves, peak_comorbidity, medication_free_waves, heart_rate_range, bmi_overall_trend, longest_persisting_condition, new_diagnosis_waves, bmi_at_first_diagnosis, medication_timing, conditions_after_medication, age_at_first_chronic, meds_at_peak_bmi, total_observation_types, bp_reading_count, first_condition_ever, wave_coverage_fraction, most_active_wave, total_rx_burden, height_stability, glucose_at_diabetes, obs_count_at_diagnosis, sbp_after_htn_wave, bmi_obese_at_htn, two_conditions_same_wave, hr_around_diagnosis, medication_stopped, hr_vs_bmi_peak, new_med_per_new_condition, longest_condition_gap, clinical_intensity_last, cardiovascular_vs_metabolic, medication_duration, condition_rate, medication_gap_waves, weight_at_first_med, htn_before_medication, sbp_before_after_medication, weight_at_diagnosis, bmi_at_first_med, conditions_by_decade, obs_before_first_condition
- "patient_id": the 8-character ID from the question
- Extra fields by category:
  - lookup_condition: wave (int)
  - lookup_observation: wave (int), observation_name (exact string)
  - trend: wave_a (smaller int), wave_b (larger int), observation_name
  - aggregate_observation_count: wave (int)
  - multihop_bmi_obesity: wave_a, wave_b
  - All other categories: patient_id only, no extra fields

observation_name MUST be one of exactly: "Systolic Blood Pressure", "Diastolic Blood Pressure", "Body Mass Index", "Body Weight", "Body Height", "Heart rate"

lookup_demographics = ONLY gender/birth year questions (no wave). Height/weight/BMI/BP/HR questions with a wave → lookup_observation. Two waves → trend.

trend: wave_a is always the smaller number even if question mentions larger wave first.

KEYWORD MAP (pick the most specific match):
  peak_bmi_wave        → highest/peak BMI + which wave
  most_medication_wave → most unique medications + wave
  condition_free_waves → no new conditions + waves
  chronic_vs_acute     → chronic (no stop date) vs time-limited
  observation_density  → average observations per wave
  record_span          → years health record spans / first to last event
  bmi_overweight_waves → BMI in overweight range (25-30) + waves
  peak_comorbidity     → most conditions active simultaneously / single wave
  medication_free_waves→ waves with NO medications at all
  heart_rate_range     → range of heart rates across waves (min to max)
  bmi_overall_trend    → BMI overall increase/decrease/stable across all waves
  longest_persisting_condition → condition active most number of waves
  new_diagnosis_waves  → waves with entirely new diagnoses / new per wave
  total_observation_types → highest variety/distinct types of observations
  bp_reading_count     → total blood pressure readings (systolic or diastolic)
  first_condition_ever → very first condition ever diagnosed + which wave
  wave_coverage_fraction → fraction/proportion of waves with conditions
  most_active_wave     → most total health events (conditions+obs+meds)
  total_rx_burden      → total prescriptions including repeats
  bmi_at_first_diagnosis → BMI in the wave of first ever diagnosis
  medication_timing    → first medication before or after first condition
  conditions_after_medication → new conditions after starting first medication
  age_at_first_chronic → age when first chronic condition diagnosed
  meds_at_peak_bmi     → medications in the wave when BMI was highest
  height_stability     → height remain stable / did height change across waves
  glucose_at_diabetes  → glucose level + diabetes/prediabetes diagnosis wave
  obs_count_at_diagnosis → observations in same wave as first diagnosed / how many obs that wave
  sbp_after_htn_wave   → SBP in wave immediately after hypertension diagnosis
  bmi_obese_at_htn     → BMI obese (>=30) in same wave as hypertension diagnosis
  two_conditions_same_wave → both [condition A] and [condition B] first diagnosed same wave
  hr_around_diagnosis  → heart rate change between wave BEFORE and wave OF first chronic condition diagnosis (NO explicit wave numbers given; do NOT use trend)
  medication_stopped   → medication stopped / discontinued while condition still active
  hr_vs_bmi_peak       → heart rate in peak BMI wave vs personal average heart rate
  new_med_per_new_condition → new medication in every wave that had a new condition
  longest_condition_gap → longest gap (in waves) between condition diagnoses
  clinical_intensity_last → most recent wave also most clinically intense / most observations
  cardiovascular_vs_metabolic → cardiovascular condition vs metabolic condition which came first
  medication_duration  → medications with a stop date / time-limited medications / how many medications stopped (about MEDICATIONS not conditions)
  condition_rate       → average new conditions per wave / on average how many conditions per wave
  medication_gap_waves → waves where patient had active CONDITIONS but NO medications prescribed (conditions without prescriptions)
  weight_at_first_med  → average body weight in the wave first prescribed any medication
  htn_before_medication → was hypertension already diagnosed before a specific medication was prescribed (temporal ordering of HTN vs a named drug)
  sbp_before_after_medication → SBP lower/higher after starting taking [medication name] vs before starting it
  weight_at_diagnosis  → body weight increase/decrease in the same wave diagnosed with [named condition]
  bmi_at_first_med     → BMI above/below threshold (25 or 30) in the wave first prescribed any medication
  conditions_by_decade → condition burden change by wave group / Wave 1-2 vs Wave 3-4 vs Wave 5-6
  obs_before_first_condition → clinical observations recorded before the first condition was diagnosed (observations vs first diagnosis date)
  obs_count_at_diagnosis → observations change in the wave diagnosed with [named condition] vs prior wave (the named thing is a CONDITION, NOT an observation name; do NOT use trend)

RULES: Never set wave=1 when no wave number appears. All v2 categories need patient_id only.

Examples:
Q: "What condition(s) were diagnosed in Wave 5 for patient 9358d6df?"
{"category": "lookup_condition", "patient_id": "9358d6df", "wave": 5}

Q: "What was the average recorded Body Mass Index in Wave 5 for patient 25d6cb75?"
{"category": "lookup_observation", "patient_id": "25d6cb75", "wave": 5, "observation_name": "Body Mass Index"}

Q: "What is the gender and birth year of patient 9ad2f1f3?"
{"category": "lookup_demographics", "patient_id": "9ad2f1f3"}

Q: "How did the average Body Mass Index change from Wave 6 to Wave 7 for patient 69b60335?"
{"category": "trend", "patient_id": "69b60335", "wave_a": 6, "wave_b": 7, "observation_name": "Body Mass Index"}

Q: "How many unique conditions were recorded in total for patient 199946d9?"
{"category": "aggregate_condition_count", "patient_id": "199946d9"}

Q: "Was the average Systolic Blood Pressure higher after patient affb8a9e was diagnosed with Hypertension?"
{"category": "multihop_hypertension", "patient_id": "affb8a9e"}

Q: "Did BMI increase between Wave 4 and Wave 6 for patient 690e0ead, and were they diagnosed with Obesity?"
{"category": "multihop_bmi_obesity", "patient_id": "690e0ead", "wave_a": 4, "wave_b": 6}

Q: "In which wave did patient 0047123f record their highest average Body Mass Index?"
{"category": "peak_bmi_wave", "patient_id": "0047123f"}

Q: "How many of patient 0149d553's ever-diagnosed conditions were chronic (no recorded stop date)?"
{"category": "chronic_vs_acute", "patient_id": "0149d553"}

Q: "What was patient 03172f6e's BMI in the wave they were first ever diagnosed with any condition?"
{"category": "bmi_at_first_diagnosis", "patient_id": "03172f6e"}

Q: "Was patient 034e9e3b's first medication prescribed before or after their first condition diagnosis?"
{"category": "medication_timing", "patient_id": "034e9e3b"}

Q: "How many new conditions were diagnosed for patient 05bfc523 after they started their first medication?"
{"category": "conditions_after_medication", "patient_id": "05bfc523"}

Q: "Did patient 04a29a39's recorded height remain stable across all waves they appear in?"
{"category": "height_stability", "patient_id": "04a29a39"}

Q: "What was patient acb4817f's recorded glucose level in the wave they were first diagnosed with Prediabetes?"
{"category": "glucose_at_diabetes", "patient_id": "acb4817f"}

Q: "How many observations were recorded for patient 04dff6e5 in the same wave they were first diagnosed with any condition?"
{"category": "obs_count_at_diagnosis", "patient_id": "04dff6e5"}

Q: "Did patient 83719bd7's average Systolic Blood Pressure increase in the wave immediately after their first Hypertension diagnosis?"
{"category": "sbp_after_htn_wave", "patient_id": "83719bd7"}

Q: "Was patient 0f5646bc's BMI in the obese range (>=30 kg/m2) in the same wave they were diagnosed with Hypertension?"
{"category": "bmi_obese_at_htn", "patient_id": "0f5646bc"}

Q: "In which wave were both 'Hypertension' and 'Body mass index 30+ - obesity (finding)' first diagnosed for patient 066c0f3d?"
{"category": "two_conditions_same_wave", "patient_id": "066c0f3d"}

Q: "How did patient 0982ef39's average heart rate change between the wave before and the wave of their first chronic condition diagnosis?"
{"category": "hr_around_diagnosis", "patient_id": "0982ef39"}

Q: "Did patient 072f2f15 have any medications stopped while a condition was still active?"
{"category": "medication_stopped", "patient_id": "072f2f15"}

Q: "In the wave patient ba190ea7 had their highest BMI, was their heart rate also above their personal average?"
{"category": "hr_vs_bmi_peak", "patient_id": "ba190ea7"}

Q: "Did patient 0780f97f receive a new medication in every wave they received a new condition diagnosis?"
{"category": "new_med_per_new_condition", "patient_id": "0780f97f"}

Q: "What was the longest gap (in waves) between condition diagnoses for patient 07b30273?"
{"category": "longest_condition_gap", "patient_id": "07b30273"}

Q: "Was patient 08180c75's most recent wave also their most clinically intense (highest observation count)?"
{"category": "clinical_intensity_last", "patient_id": "08180c75"}

Q: "Which came first for patient 09616ead — their cardiovascular condition or their metabolic condition?"
{"category": "cardiovascular_vs_metabolic", "patient_id": "09616ead"}

Q: "How many of patient 0288abb6's medications had a recorded stop date (were time-limited)?"
{"category": "medication_duration", "patient_id": "0288abb6"}

Q: "On average, how many new conditions was patient 0325261f diagnosed with per wave?"
{"category": "condition_rate", "patient_id": "0325261f"}

Q: "Were there any waves where patient 03612a7e had active conditions but NO medications prescribed?"
{"category": "medication_gap_waves", "patient_id": "03612a7e"}

Q: "What was patient 05ad1c43's average body weight in the wave they were first prescribed any medication?"
{"category": "weight_at_first_med", "patient_id": "05ad1c43"}

Q: "Was patient 2ffe9369 already diagnosed with Hypertension before they were prescribed Hydrochlorothiazide 25 MG Oral Tablet?"
{"category": "htn_before_medication", "patient_id": "2ffe9369"}

Q: "How did patient 0982ef39's average heart rate change between the wave before and the wave of their first chronic condition diagnosis (Body mass index 30+ - obesity (finding))?"
{"category": "hr_around_diagnosis", "patient_id": "0982ef39"}

Q: "How did the number of observations recorded for patient 09fc03c4 change in the wave they were diagnosed with Acute bacterial sinusitis (disorder) compared to the prior wave?"
{"category": "obs_count_at_diagnosis", "patient_id": "09fc03c4"}

Q: "Was patient 278c5ad9's average Systolic Blood Pressure lower after they started taking Acetaminophen 325 MG Oral Tablet compared to before?"
{"category": "sbp_before_after_medication", "patient_id": "278c5ad9"}

Q: "Did patient 6d4312d1's body weight increase in the same wave they were diagnosed with Sprain of ankle?"
{"category": "weight_at_diagnosis", "patient_id": "6d4312d1"}

Q: "Was patient 21838b2e's BMI above 25 kg/m2 in the wave they were first prescribed any medication?"
{"category": "bmi_at_first_med", "patient_id": "21838b2e"}

Q: "How did patient 076688b0's condition burden change across decades (Waves 1-2 vs Waves 3-4 vs Waves 5-6)?"
{"category": "conditions_by_decade", "patient_id": "076688b0"}

Q: "Did patient 05cdf25a have any clinical observations recorded before their first condition was diagnosed?"
{"category": "obs_before_first_condition", "patient_id": "05cdf25a"}

Now parse this question:
Q: "{question}"
"""


_JSON_OBJ_RE = re.compile(r'\{[^{}]*\}', re.DOTALL)

def extract_spec(question: str, retries: int = 3) -> dict | None:
    """Call Groq to extract a structured query spec from the question."""
    prompt = _EXTRACTION_PROMPT.replace("{question}", question)
    for attempt in range(retries):
        try:
            resp = groq_client.chat.completions.create(
                model=GROQ_MODEL,
                max_tokens=300,
                temperature=0,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = resp.choices[0].message.content.strip()
            # Strip markdown code fences if present
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            raw = raw.strip()
            # Fast path: entire response is the JSON object
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                pass
            # Fallback: 70b models sometimes prepend/append prose — extract the JSON object
            m = _JSON_OBJ_RE.search(raw)
            if m:
                try:
                    return json.loads(m.group())
                except json.JSONDecodeError:
                    pass
            # Truly unparseable — retry with back-off
            if attempt < retries - 1:
                time.sleep(2 ** attempt * 2)
        except Exception as exc:
            err = str(exc).lower()
            if "rate" in err or "429" in err or "quota" in err:
                return None  # Fast-fail on rate limit — retrying burns more tokens
            if attempt < retries - 1:
                time.sleep(2 ** attempt * 4)
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
    # Waves where patient appears in observations but started NO new conditions
    free = sorted(obs_waves - cond_waves)
    if not free:
        return f"No condition-free waves found — conditions were diagnosed in all {len(obs_waves)} waves."
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
    pc = conditions[conditions["PATIENT"] == full_pid].dropna(subset=["WAVE"]).copy()
    if pc.empty:
        return f"No condition data for patient {spec['patient_id']}."
    pc["WAVE"] = pc["WAVE"].astype(int)
    max_wave = int(pc["WAVE"].max())
    best_count = 0; best_cond = ""
    for cond, cg in pc.groupby("DESCRIPTION"):
        # Chronic (no STOP): active from first wave to patient's last wave
        if cg["STOP"].isna().all():
            n_active = max_wave - int(cg["WAVE"].min()) + 1
        else:
            n_active = int(cg["WAVE"].nunique())
        if n_active > best_count:
            best_count = n_active; best_cond = cond
    return f"{best_cond} -- active across {best_count} waves"


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
    pc = conditions[conditions["PATIENT"] == full_pid].copy()
    if pc.empty:
        return "No condition data."
    pm = medications[medications["PATIENT"] == full_pid].copy()
    if pm.empty:
        return "No medication data."
    # Compare exact dates so same-wave comparisons are date-accurate
    pc["START_DT"] = pd.to_datetime(pc["START"], errors="coerce")
    pm["START_DT"] = pd.to_datetime(pm["START"], errors="coerce")
    pc = pc.dropna(subset=["START_DT"]).sort_values("START_DT")
    pm = pm.dropna(subset=["START_DT"]).sort_values("START_DT")
    first_cond_date = pc["START_DT"].min()
    first_med_date  = pm["START_DT"].min()
    first_cond_wave = _wave_from_year(first_cond_date.year)
    first_med_wave  = _wave_from_year(first_med_date.year)
    first_cond = pc[pc["START_DT"] == first_cond_date]["DESCRIPTION"].iloc[0]
    first_med  = pm[pm["START_DT"] == first_med_date]["DESCRIPTION"].iloc[0]
    if first_med_date < first_cond_date:
        direction = "before"
    elif first_med_date > first_cond_date:
        direction = "after"
    else:
        direction = "same day as"
    return (
        f"Medication came {direction} -- "
        f"first med Wave {first_med_wave} ({str(first_med_date.date())}), "
        f"first condition Wave {first_cond_wave} ({str(first_cond_date.date())})"
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


# ── New v2 Aggregate Handlers (batch 2) ───────────────────────────────────────

def handle_bp_reading_count(spec: dict) -> str:
    full_pid = _full_pid(spec["patient_id"])
    if full_pid is None:
        return f"Patient {spec['patient_id']} not found."
    bp = observations[
        (observations["PATIENT"] == full_pid) &
        (observations["DESCRIPTION"].isin(["Systolic Blood Pressure", "Diastolic Blood Pressure"]))
    ]
    return f"{len(bp)} blood pressure readings"


def handle_first_condition_ever(spec: dict) -> str:
    full_pid = _full_pid(spec["patient_id"])
    if full_pid is None:
        return f"Patient {spec['patient_id']} not found."
    pc = conditions[conditions["PATIENT"] == full_pid].dropna(subset=["START"])
    if pc.empty:
        return f"No conditions found for patient {spec['patient_id']}."
    first = pc.sort_values("START").iloc[0]
    try:
        wave = int(first["WAVE"])
    except (ValueError, TypeError, KeyError):
        wave = "?"
    date_str = str(first["START"])[:10]
    return f"{first['DESCRIPTION']} (Wave {wave}, {date_str})"


def handle_wave_coverage_fraction(spec: dict) -> str:
    full_pid = _full_pid(spec["patient_id"])
    if full_pid is None:
        return f"Patient {spec['patient_id']} not found."
    obs_waves  = set(
        observations[observations["PATIENT"] == full_pid]["WAVE"].dropna().astype(int).unique()
    )
    cond_waves = set(
        conditions[conditions["PATIENT"] == full_pid]["WAVE"].dropna().astype(int).unique()
    )
    n_total = len(obs_waves)
    n_with  = len(cond_waves & obs_waves)
    pct     = int(round(n_with / n_total * 100)) if n_total else 0
    return f"{n_with} of {n_total} waves had condition diagnoses ({pct}%)"


def handle_most_active_wave(spec: dict) -> str:
    full_pid = _full_pid(spec["patient_id"])
    if full_pid is None:
        return f"Patient {spec['patient_id']} not found."
    cond_cts = conditions[conditions["PATIENT"] == full_pid].dropna(subset=["WAVE"]).groupby("WAVE").size()
    obs_cts  = observations[observations["PATIENT"] == full_pid].dropna(subset=["WAVE"]).groupby("WAVE").size()
    pm = medications[medications["PATIENT"] == full_pid].copy()
    if "WAVE" not in pm.columns:
        pm["WAVE"] = pd.to_datetime(pm["START"], errors="coerce").dt.year.apply(
            lambda y: _wave_from_year(y) if pd.notna(y) else None
        )
    pm = pm.dropna(subset=["WAVE"])
    pm["WAVE"] = pm["WAVE"].astype(int)
    med_cts = pm.groupby("WAVE").size() if not pm.empty else pd.Series(dtype=int)
    all_waves = sorted(
        set(cond_cts.index.astype(int)) | set(obs_cts.index.astype(int)) | set(med_cts.index.astype(int))
    )
    if not all_waves:
        return f"No data for patient {spec['patient_id']}."
    totals = {
        w: int(cond_cts.get(w, 0)) + int(obs_cts.get(w, 0)) + int(med_cts.get(w, 0))
        for w in all_waves
    }
    best_wave = max(totals, key=totals.get)
    return f"Wave {best_wave} ({totals[best_wave]} total events)"


def handle_total_observation_types(spec: dict) -> str:
    full_pid = _full_pid(spec["patient_id"])
    if full_pid is None:
        return f"Patient {spec['patient_id']} not found."
    po = observations[observations["PATIENT"] == full_pid].dropna(subset=["WAVE"])
    if po.empty:
        return f"No observation data for patient {spec['patient_id']}."
    wave_types = po.groupby("WAVE")["DESCRIPTION"].nunique()
    best_wave  = int(wave_types.idxmax())
    return f"Wave {best_wave} ({wave_types[best_wave]} distinct types)"


def handle_total_rx_burden(spec: dict) -> str:
    full_pid = _full_pid(spec["patient_id"])
    if full_pid is None:
        return f"Patient {spec['patient_id']} not found."
    pm = medications[medications["PATIENT"] == full_pid]
    return f"{len(pm)} total prescription records"


# ── New handlers: aggregate fix + multi-hop batch 3 ───────────────────────────

def handle_height_stability(spec: dict) -> str:
    full_pid = _full_pid(spec["patient_id"])
    if full_pid is None:
        return f"Patient {spec['patient_id']} not found."
    ht = observations[
        (observations["PATIENT"] == full_pid) &
        (observations["DESCRIPTION"] == "Body Height")
    ].copy()
    ht["VALUE"] = pd.to_numeric(ht["VALUE"], errors="coerce")
    ht = ht.dropna(subset=["VALUE"])
    if ht.empty:
        return f"No height data for patient {spec['patient_id']}."
    mn = round(float(ht["VALUE"].min()), 1)
    mx = round(float(ht["VALUE"].max()), 1)
    if mx - mn < 1.0:
        return f"Yes -- stable at {mn} cm across all waves"
    return f"No -- changed from {mn} cm to {mx} cm"


def handle_glucose_at_diabetes(spec: dict) -> str:
    full_pid = _full_pid(spec["patient_id"])
    if full_pid is None:
        return f"Patient {spec['patient_id']} not found."
    pc = conditions[
        (conditions["PATIENT"] == full_pid) &
        (conditions["DESCRIPTION"].str.contains("iabetes|Glucose|glucose|Prediabetes", na=False, regex=True))
    ].dropna(subset=["WAVE"])
    po = observations[
        (observations["PATIENT"] == full_pid) &
        (observations["DESCRIPTION"].str.contains("Glucose|glucose|Blood sugar", na=False, regex=True))
    ].dropna(subset=["WAVE"])
    if pc.empty:
        return "No diabetes/prediabetes diagnosis found."
    diag_wave = int(pc["WAVE"].min())
    if po.empty:
        return f"No glucose reading in Wave {diag_wave} (diagnosis wave)."
    wave_data = po[po["WAVE"].astype(int) == diag_wave]
    if wave_data.empty:
        return f"No glucose reading in Wave {diag_wave} (diagnosis wave)."
    val  = round(float(pd.to_numeric(wave_data["VALUE"], errors="coerce").mean()), 1)
    unit = wave_data["UNITS"].iloc[0] if pd.notna(wave_data["UNITS"].iloc[0]) else ""
    return f"{val} {unit} in Wave {diag_wave}".strip()


def handle_obs_count_at_diagnosis(spec: dict) -> str:
    full_pid = _full_pid(spec["patient_id"])
    if full_pid is None:
        return f"Patient {spec['patient_id']} not found."
    pc = conditions[conditions["PATIENT"] == full_pid].dropna(subset=["WAVE"])
    po = observations[observations["PATIENT"] == full_pid].dropna(subset=["WAVE"])
    if pc.empty or po.empty:
        return "Insufficient data."
    diag_wave  = int(pc["WAVE"].min())
    prior_wave = diag_wave - 1
    cond_name  = pc[pc["WAVE"] == diag_wave]["DESCRIPTION"].iloc[0] if not pc[pc["WAVE"] == diag_wave].empty else ""
    n_diag  = int((po["WAVE"].astype(int) == diag_wave).sum())
    n_prior = int((po["WAVE"].astype(int) == prior_wave).sum()) if prior_wave >= 1 else 0
    direction = "increased" if n_diag > n_prior else ("decreased" if n_diag < n_prior else "stayed the same")
    if n_prior == 0 and prior_wave < 1:
        return f"{n_diag} observations in Wave {diag_wave} (same wave as first diagnosis: {cond_name})"
    return (
        f"Observations {direction} from {n_prior} (Wave {prior_wave}) to "
        f"{n_diag} (Wave {diag_wave})"
    )


def handle_sbp_after_htn_wave(spec: dict) -> str:
    full_pid = _full_pid(spec["patient_id"])
    if full_pid is None:
        return f"Patient {spec['patient_id']} not found."
    pc = conditions[
        (conditions["PATIENT"] == full_pid) &
        (conditions["DESCRIPTION"].str.contains("ypertension", na=False))
    ].dropna(subset=["WAVE"])
    po = observations[
        (observations["PATIENT"] == full_pid) &
        (observations["DESCRIPTION"] == "Systolic Blood Pressure")
    ].dropna(subset=["WAVE"]).copy()
    if pc.empty or po.empty:
        return "Insufficient data."
    po["VALUE"] = pd.to_numeric(po["VALUE"], errors="coerce")
    diag_wave = int(pc["WAVE"].min())
    next_wave = diag_wave + 1
    sbp_diag = po[po["WAVE"].astype(int) == diag_wave]["VALUE"].mean()
    sbp_next = po[po["WAVE"].astype(int) == next_wave]["VALUE"].mean()
    if pd.isna(sbp_next):
        return f"No SBP data in Wave {next_wave} (wave after diagnosis)."
    direction = "increased" if sbp_next > sbp_diag else "decreased" if sbp_next < sbp_diag else "stayed the same"
    return (
        f"{'Yes' if sbp_next > sbp_diag else 'No'} -- SBP {direction} -- "
        f"was {round(float(sbp_diag),1)} in Wave {diag_wave}, "
        f"{round(float(sbp_next),1)} in Wave {next_wave}"
    )


def handle_bmi_obese_at_htn(spec: dict) -> str:
    full_pid = _full_pid(spec["patient_id"])
    if full_pid is None:
        return f"Patient {spec['patient_id']} not found."
    pc = conditions[
        (conditions["PATIENT"] == full_pid) &
        (conditions["DESCRIPTION"].str.contains("ypertension", na=False))
    ].dropna(subset=["WAVE"])
    po = observations[
        (observations["PATIENT"] == full_pid) &
        (observations["DESCRIPTION"] == "Body Mass Index")
    ].dropna(subset=["WAVE"]).copy()
    if pc.empty:
        return "No hypertension diagnosis found."
    if po.empty:
        return "No BMI data found."
    po["VALUE"] = pd.to_numeric(po["VALUE"], errors="coerce")
    diag_wave = int(pc["WAVE"].min())
    bmi_wave  = po[po["WAVE"].astype(int) == diag_wave]
    if bmi_wave.empty:
        return f"No BMI data in Wave {diag_wave} (HTN diagnosis wave)."
    avg_bmi = round(float(bmi_wave["VALUE"].mean()), 1)
    if avg_bmi >= 30:
        return f"Yes -- BMI was {avg_bmi} kg/m2 in Wave {diag_wave} (obese, >=30)"
    return f"No -- BMI was {avg_bmi} kg/m2 in Wave {diag_wave} (below 30)"


def handle_two_conditions_same_wave(spec: dict) -> str:
    full_pid = _full_pid(spec["patient_id"])
    if full_pid is None:
        return f"Patient {spec['patient_id']} not found."
    pc = conditions[conditions["PATIENT"] == full_pid].dropna(subset=["WAVE"]).copy()
    if pc.empty:
        return "No condition data."
    pc["WAVE"] = pc["WAVE"].astype(int)
    # Find the earliest wave where the patient has 2+ distinct conditions first diagnosed
    wave_first = pc.sort_values("START").groupby("DESCRIPTION")["WAVE"].min()
    wave_groups = wave_first.reset_index().groupby("WAVE")["DESCRIPTION"].apply(list)
    multi = wave_groups[wave_groups.apply(len) >= 2]
    if multi.empty:
        return "No single wave where 2+ conditions were first diagnosed."
    best_wave = int(multi.index.min())
    cond_list = multi[best_wave]
    return f"Both first diagnosed in Wave {best_wave}: {'; '.join(str(c) for c in cond_list[:2])}"


def handle_hr_around_diagnosis(spec: dict) -> str:
    full_pid = _full_pid(spec["patient_id"])
    if full_pid is None:
        return f"Patient {spec['patient_id']} not found."
    pc = conditions[conditions["PATIENT"] == full_pid].dropna(subset=["WAVE"])
    po = observations[
        (observations["PATIENT"] == full_pid) &
        (observations["DESCRIPTION"] == "Heart rate")
    ].dropna(subset=["WAVE"]).copy()
    if pc.empty or po.empty:
        return "Insufficient data."
    po["VALUE"] = pd.to_numeric(po["VALUE"], errors="coerce")
    # Use first chronic condition wave if available, else earliest condition
    chronic = conditions[
        (conditions["PATIENT"] == full_pid) & conditions["STOP"].isna()
    ].dropna(subset=["WAVE"])
    if not chronic.empty:
        diag_wave = int(chronic["WAVE"].min())
    else:
        diag_wave = int(pc["WAVE"].min())
    prior = diag_wave - 1
    hr_diag  = po[po["WAVE"].astype(int) == diag_wave]["VALUE"].mean()
    hr_prior = po[po["WAVE"].astype(int) == prior]["VALUE"].mean() if prior >= 1 else float("nan")
    if pd.isna(hr_diag):
        return f"No heart rate data in Wave {diag_wave} (diagnosis wave)."
    if pd.isna(hr_prior):
        return f"HR was {round(float(hr_diag),1)} /min in Wave {diag_wave} (no prior wave data)."
    return (
        f"HR was {round(float(hr_prior),1)} /min before (Wave {prior}), "
        f"{round(float(hr_diag),1)} /min in diagnosis wave (Wave {diag_wave})"
    )


def handle_medication_stopped(spec: dict) -> str:
    full_pid = _full_pid(spec["patient_id"])
    if full_pid is None:
        return f"Patient {spec['patient_id']} not found."
    pm = medications[medications["PATIENT"] == full_pid].copy()
    if pm.empty:
        return "No medication data."
    stopped = pm[pm["STOP"].notna()]
    if stopped.empty:
        return "No time-limited medications found."
    pc = conditions[conditions["PATIENT"] == full_pid]
    active = pc[pc["STOP"].isna()]
    if not active.empty:
        ex_med = stopped.sort_values("STOP").iloc[0]
        ex_cond = active.iloc[0]
        return (
            f"Yes -- '{str(ex_med['DESCRIPTION'])[:40]}' stopped "
            f"{str(ex_med['STOP'])[:10]} while "
            f"'{str(ex_cond['DESCRIPTION'])[:40]}' "
            f"(diagnosed {str(ex_cond['START'])[:10]}) was still active"
        )
    return f"Yes -- {len(stopped)} medications had stop dates."


def handle_hr_vs_bmi_peak(spec: dict) -> str:
    full_pid = _full_pid(spec["patient_id"])
    if full_pid is None:
        return f"Patient {spec['patient_id']} not found."
    bmi = observations[
        (observations["PATIENT"] == full_pid) &
        (observations["DESCRIPTION"] == "Body Mass Index")
    ].dropna(subset=["WAVE"]).copy()
    hr = observations[
        (observations["PATIENT"] == full_pid) &
        (observations["DESCRIPTION"] == "Heart rate")
    ].dropna(subset=["WAVE"]).copy()
    if bmi.empty or hr.empty:
        return "Insufficient data."
    bmi["VALUE"] = pd.to_numeric(bmi["VALUE"], errors="coerce")
    hr["VALUE"]  = pd.to_numeric(hr["VALUE"],  errors="coerce")
    bmi = bmi.dropna(subset=["VALUE"])
    hr  = hr.dropna(subset=["VALUE"])
    bmi_avg  = bmi.groupby("WAVE")["VALUE"].mean()
    peak_wave = int(bmi_avg.idxmax())
    peak_bmi  = round(float(bmi_avg.max()), 1)
    hr_overall = round(float(hr["VALUE"].mean()), 1)
    hr_peak    = hr[hr["WAVE"].astype(int) == peak_wave]["VALUE"].mean()
    if pd.isna(hr_peak):
        return f"BMI peaked at {peak_bmi} kg/m2 in Wave {peak_wave}; no HR data that wave."
    hr_peak = round(float(hr_peak), 1)
    above = hr_peak > hr_overall
    return (
        f"{'Yes' if above else 'No'} -- "
        f"peak BMI {peak_bmi} kg/m2 (Wave {peak_wave}), "
        f"HR {hr_peak} /min (personal avg {hr_overall} /min)"
    )


def handle_new_med_per_new_condition(spec: dict) -> str:
    full_pid = _full_pid(spec["patient_id"])
    if full_pid is None:
        return f"Patient {spec['patient_id']} not found."
    pc = conditions[conditions["PATIENT"] == full_pid].dropna(subset=["WAVE"]).copy()
    pm = medications[medications["PATIENT"] == full_pid].copy()
    if pc.empty or pm.empty:
        return "Insufficient data."
    pc["WAVE"] = pc["WAVE"].astype(int)
    if "WAVE" not in pm.columns:
        pm["WAVE"] = pd.to_datetime(pm["START"], errors="coerce").dt.year.apply(
            lambda y: _wave_from_year(y) if pd.notna(y) else None
        )
    pm = pm.dropna(subset=["WAVE"])
    pm["WAVE"] = pm["WAVE"].astype(int)
    seen_conds = set(); seen_meds = set(); mismatched = []
    for wave in sorted(pc["WAVE"].unique()):
        wc = set(pc[pc["WAVE"] == wave]["DESCRIPTION"])
        wm = set(pm[pm["WAVE"] == wave]["DESCRIPTION"]) if not pm.empty else set()
        new_c = wc - seen_conds
        new_m = wm - seen_meds
        if new_c and not new_m:
            mismatched.append(wave)
        seen_conds |= wc; seen_meds |= wm
    if not mismatched:
        n_waves = len([w for w in pc["WAVE"].unique() if set(pc[pc["WAVE"]==w]["DESCRIPTION"]) - (seen_conds - set(pc[pc["WAVE"]==w]["DESCRIPTION"]))])
        return f"Yes -- new medications matched in all {len(pc['WAVE'].unique())} waves with new conditions"
    return f"No -- Wave {mismatched[0]} had new condition(s) but no new medication"


def handle_longest_condition_gap(spec: dict) -> str:
    full_pid = _full_pid(spec["patient_id"])
    if full_pid is None:
        return f"Patient {spec['patient_id']} not found."
    pc = conditions[conditions["PATIENT"] == full_pid].dropna(subset=["WAVE"]).copy()
    if pc.empty:
        return "No condition data."
    waves = sorted(pc["WAVE"].astype(int).unique())
    if len(waves) < 2:
        return "Only one condition wave — no gap to measure."
    max_gap = 0; gap_start = gap_end = 0
    for i in range(len(waves) - 1):
        gap = waves[i + 1] - waves[i]
        if gap > max_gap:
            max_gap = gap; gap_start = waves[i]; gap_end = waves[i + 1]
    return f"{max_gap} waves between Wave {gap_start} and Wave {gap_end}"


def handle_clinical_intensity_last(spec: dict) -> str:
    full_pid = _full_pid(spec["patient_id"])
    if full_pid is None:
        return f"Patient {spec['patient_id']} not found."
    po = observations[observations["PATIENT"] == full_pid].dropna(subset=["WAVE"]).copy()
    if po.empty:
        return "No observation data."
    po["WAVE"] = po["WAVE"].astype(int)
    wave_counts = po.groupby("WAVE").size()
    last_wave  = int(wave_counts.index.max())
    last_count = int(wave_counts[last_wave])
    peak_wave  = int(wave_counts.idxmax())
    peak_count = int(wave_counts[peak_wave])
    if last_wave == peak_wave:
        return f"Yes -- Wave {last_wave} (most recent) had {last_count} observations (highest)"
    return (
        f"No -- Wave {peak_wave} had more ({peak_count} obs) "
        f"vs Wave {last_wave} ({last_count} obs)"
    )


def handle_cardiovascular_vs_metabolic(spec: dict) -> str:
    full_pid = _full_pid(spec["patient_id"])
    if full_pid is None:
        return f"Patient {spec['patient_id']} not found."
    pc = conditions[conditions["PATIENT"] == full_pid].dropna(subset=["START"]).copy()
    if pc.empty:
        return "No condition data."
    cardio_pat   = "Hypertension|Coronary|Cardiac|Heart|Stroke|Atrial|Angina"
    metabolic_pat = "Diabetes|Obesity|BMI|Prediabetes|Metabolic|Lipid|Cholesterol"
    cardio   = pc[pc["DESCRIPTION"].str.contains(cardio_pat,   case=False, na=False)]
    metabolic = pc[pc["DESCRIPTION"].str.contains(metabolic_pat, case=False, na=False)]
    if cardio.empty and metabolic.empty:
        return "No cardiovascular or metabolic conditions found."
    if cardio.empty:
        first_m = metabolic.loc[pd.to_datetime(metabolic["START"]).idxmin()]
        return f"No cardiovascular condition. First metabolic: {str(first_m['DESCRIPTION'])[:50]}"
    if metabolic.empty:
        first_c = cardio.loc[pd.to_datetime(cardio["START"]).idxmin()]
        return f"No metabolic condition. First cardiovascular: {str(first_c['DESCRIPTION'])[:50]}"
    first_c_date = pd.to_datetime(cardio["START"]).min()
    first_m_date = pd.to_datetime(metabolic["START"]).min()
    first_c_name = cardio.loc[pd.to_datetime(cardio["START"]).idxmin(), "DESCRIPTION"]
    first_m_name = metabolic.loc[pd.to_datetime(metabolic["START"]).idxmin(), "DESCRIPTION"]
    c_wave = _wave_from_year(first_c_date.year)
    m_wave = _wave_from_year(first_m_date.year)
    if first_c_date <= first_m_date:
        return f"{str(first_c_name)[:50]} diagnosed first (Wave {c_wave}, {first_c_date.date()})"
    return f"{str(first_m_name)[:50]} diagnosed first (Wave {m_wave}, {first_m_date.date()})"


# ── New handlers: batch 4 (12 remaining failures) ────────────────────────────

def handle_medication_duration(spec: dict) -> str:
    full_pid = _full_pid(spec["patient_id"])
    if full_pid is None:
        return f"Patient {spec['patient_id']} not found."
    pm = medications[medications["PATIENT"] == full_pid]
    if pm.empty:
        return f"No medication data for patient {spec['patient_id']}."
    time_limited = int(pm["STOP"].notna().sum())
    total = len(pm)
    ongoing = total - time_limited
    return f"{time_limited} time-limited out of {total} total ({ongoing} ongoing)"


def handle_condition_rate(spec: dict) -> str:
    full_pid = _full_pid(spec["patient_id"])
    if full_pid is None:
        return f"Patient {spec['patient_id']} not found."
    pc = conditions[conditions["PATIENT"] == full_pid].dropna(subset=["WAVE"]).copy()
    if pc.empty:
        return f"No condition data for patient {spec['patient_id']}."
    pc["WAVE"] = pc["WAVE"].astype(int)
    first_wave = pc.sort_values("START").groupby("DESCRIPTION")["WAVE"].min()
    new_per_wave = first_wave.reset_index().groupby("WAVE").size()
    total_new = int(new_per_wave.sum())
    n_waves = int(len(new_per_wave))
    rate = round(float(total_new / n_waves), 1)
    return f"{rate} new conditions per wave ({total_new} unique across {n_waves} waves)"


def handle_medication_gap_waves(spec: dict) -> str:
    full_pid = _full_pid(spec["patient_id"])
    if full_pid is None:
        return f"Patient {spec['patient_id']} not found."
    pc = conditions[conditions["PATIENT"] == full_pid].dropna(subset=["WAVE"]).copy()
    if pc.empty:
        return f"No condition data for patient {spec['patient_id']}."
    pc["WAVE"] = pc["WAVE"].astype(int)
    pm = medications[medications["PATIENT"] == full_pid].copy()
    if "WAVE" not in pm.columns:
        pm["WAVE"] = pd.to_datetime(pm["START"], errors="coerce").dt.year.apply(
            lambda y: _wave_from_year(y) if pd.notna(y) else None
        )
    pm = pm.dropna(subset=["WAVE"])
    pm["WAVE"] = pm["WAVE"].astype(int)
    med_waves = set(pm["WAVE"].unique()) if not pm.empty else set()
    gaps = []
    for w in sorted(pc["WAVE"].unique()):
        if w not in med_waves:
            n_conds = int((pc["WAVE"] == w).sum())
            gaps.append(f"Wave {w} ({n_conds} conditions, 0 medications)")
    if not gaps:
        return "No — medications were prescribed in all waves with conditions."
    return f"Yes — {'; '.join(gaps)}"


def handle_weight_at_first_med(spec: dict) -> str:
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
    pm["WAVE"] = pm["WAVE"].astype(int)
    first_med_wave = int(pm["WAVE"].min())
    po = observations[
        (observations["PATIENT"] == full_pid) &
        (observations["DESCRIPTION"] == "Body Weight") &
        (observations["WAVE"] == first_med_wave)
    ].copy()
    if po.empty:
        return f"No Body Weight readings in Wave {first_med_wave} (first prescription wave)."
    po["VALUE"] = pd.to_numeric(po["VALUE"], errors="coerce")
    avg_weight = round(float(po["VALUE"].dropna().mean()), 1)
    unit = po["UNITS"].iloc[0] if pd.notna(po["UNITS"].iloc[0]) else "kg"
    return f"{avg_weight} {unit} in Wave {first_med_wave} (wave of first prescription)"


def handle_htn_before_medication(spec: dict) -> str:
    full_pid = _full_pid(spec["patient_id"])
    if full_pid is None:
        return f"Patient {spec['patient_id']} not found."
    hyp = conditions[
        (conditions["PATIENT"] == full_pid) &
        (conditions["DESCRIPTION"] == "Hypertension")
    ].dropna(subset=["START"])
    pm = medications[medications["PATIENT"] == full_pid].dropna(subset=["START"])
    if hyp.empty:
        return "No Hypertension diagnosis found."
    if pm.empty:
        return "No medication data."
    hyp_date = pd.to_datetime(hyp["START"]).min()
    med_date = pd.to_datetime(pm["START"]).min()
    hyp_wave = _wave_from_year(hyp_date.year)
    med_wave = _wave_from_year(med_date.year)
    if hyp_date <= med_date:
        return (
            f"Yes — Hypertension Wave {hyp_wave} ({str(hyp_date.date())}), "
            f"medication prescribed Wave {med_wave} ({str(med_date.date())})"
        )
    return (
        f"No — medication Wave {med_wave} ({str(med_date.date())}) came before "
        f"Hypertension Wave {hyp_wave} ({str(hyp_date.date())})"
    )


def handle_sbp_before_after_medication(spec: dict) -> str:
    full_pid = _full_pid(spec["patient_id"])
    if full_pid is None:
        return f"Patient {spec['patient_id']} not found."
    pm = medications[medications["PATIENT"] == full_pid].dropna(subset=["START"])
    if pm.empty:
        return "No medication data."
    first_med_date = pd.to_datetime(pm["START"]).min()
    sbp = observations[
        (observations["PATIENT"] == full_pid) &
        (observations["DESCRIPTION"] == "Systolic Blood Pressure")
    ].copy()
    sbp["VALUE"]   = pd.to_numeric(sbp["VALUE"], errors="coerce")
    sbp["DATE_DT"] = pd.to_datetime(sbp["DATE"], errors="coerce")
    sbp = sbp.dropna(subset=["VALUE", "DATE_DT"])
    if sbp.empty:
        return "No SBP data."
    before = sbp[sbp["DATE_DT"] <  first_med_date]["VALUE"]
    after  = sbp[sbp["DATE_DT"] >= first_med_date]["VALUE"]
    if before.empty or after.empty:
        return "Insufficient SBP data before or after first medication."
    avg_b = round(float(before.mean()), 1)
    avg_a = round(float(after.mean()), 1)
    unit  = sbp["UNITS"].iloc[0] if pd.notna(sbp["UNITS"].iloc[0]) else "mm[Hg]"
    lower = avg_a < avg_b
    return (
        f"{'Yes' if lower else 'No'} -- "
        f"avg SBP before: {avg_b} {unit}; after: {avg_a} {unit}"
    )


def handle_weight_at_diagnosis(spec: dict) -> str:
    full_pid = _full_pid(spec["patient_id"])
    if full_pid is None:
        return f"Patient {spec['patient_id']} not found."
    pc = conditions[conditions["PATIENT"] == full_pid].dropna(subset=["WAVE"]).copy()
    if pc.empty:
        return f"No condition data for patient {spec['patient_id']}."
    pc["WAVE"] = pc["WAVE"].astype(int)
    po = observations[
        (observations["PATIENT"] == full_pid) &
        (observations["DESCRIPTION"] == "Body Weight")
    ].dropna(subset=["WAVE"]).copy()
    if po.empty:
        return f"No Body Weight data for patient {spec['patient_id']}."
    po["VALUE"] = pd.to_numeric(po["VALUE"], errors="coerce")
    po = po.dropna(subset=["VALUE"])
    po["WAVE"] = po["WAVE"].astype(int)
    diag_wave  = int(pc["WAVE"].min())
    prior_wave = diag_wave - 1
    unit = po["UNITS"].iloc[0] if pd.notna(po["UNITS"].iloc[0]) else "kg"
    w_diag  = po[po["WAVE"] == diag_wave]["VALUE"].mean()
    w_prior = po[po["WAVE"] == prior_wave]["VALUE"].mean() if prior_wave >= 1 else float("nan")
    if pd.isna(w_diag):
        return f"No Body Weight readings in Wave {diag_wave} for patient {spec['patient_id']}."
    if pd.isna(w_prior):
        return f"Weight was {round(float(w_diag),1)} {unit} in Wave {diag_wave} (no prior wave data)."
    wd = round(float(w_diag), 1); wp = round(float(w_prior), 1)
    increased = wd > wp
    return (
        f"{'Yes' if increased else 'No'} -- "
        f"weight {wp} {unit} (Wave {prior_wave}), {wd} {unit} (Wave {diag_wave})"
    )


def handle_bmi_at_first_med(spec: dict) -> str:
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
    pm["WAVE"] = pm["WAVE"].astype(int)
    first_med_wave = int(pm["WAVE"].min())
    po = observations[
        (observations["PATIENT"] == full_pid) &
        (observations["DESCRIPTION"] == "Body Mass Index") &
        (observations["WAVE"] == first_med_wave)
    ].copy()
    if po.empty:
        return f"No BMI data in Wave {first_med_wave} (first medication wave)."
    po["VALUE"] = pd.to_numeric(po["VALUE"], errors="coerce")
    avg_bmi = round(float(po["VALUE"].dropna().mean()), 1)
    above_25 = avg_bmi > 25
    return (
        f"{'Yes' if above_25 else 'No'} -- "
        f"BMI {avg_bmi} kg/m2 in Wave {first_med_wave} (first medication wave)"
    )


def handle_conditions_by_decade(spec: dict) -> str:
    full_pid = _full_pid(spec["patient_id"])
    if full_pid is None:
        return f"Patient {spec['patient_id']} not found."
    pc = conditions[conditions["PATIENT"] == full_pid].dropna(subset=["WAVE"]).copy()
    if pc.empty:
        return f"No condition data for patient {spec['patient_id']}."
    pc["WAVE"] = pc["WAVE"].astype(int)
    early = int((pc["WAVE"].isin([1, 2])).sum())
    mid   = int((pc["WAVE"].isin([3, 4])).sum())
    late  = int((pc["WAVE"].isin([5, 6])).sum())
    return f"Early waves (1-2): {early}, Mid waves (3-4): {mid}, Late waves (5-6): {late} conditions"


def handle_obs_before_first_condition(spec: dict) -> str:
    full_pid = _full_pid(spec["patient_id"])
    if full_pid is None:
        return f"Patient {spec['patient_id']} not found."
    pc = conditions[conditions["PATIENT"] == full_pid].dropna(subset=["START"]).copy()
    po = observations[observations["PATIENT"] == full_pid].dropna(subset=["DATE"]).copy()
    if pc.empty:
        return "No condition data."
    if po.empty:
        return "No observation data."
    pc["START_DT"] = pd.to_datetime(pc["START"], errors="coerce")
    po["DATE_DT"]  = pd.to_datetime(po["DATE"],  errors="coerce")
    pc = pc.dropna(subset=["START_DT"])
    po = po.dropna(subset=["DATE_DT"])
    first_cond_date = pc["START_DT"].min()
    first_obs_date  = po["DATE_DT"].min()
    first_cond_wave = _wave_from_year(first_cond_date.year)
    first_obs_wave  = _wave_from_year(first_obs_date.year)
    if first_obs_date < first_cond_date:
        return (
            f"Yes -- first obs Wave {first_obs_wave} ({str(first_obs_date.date())}), "
            f"first diagnosis Wave {first_cond_wave} ({str(first_cond_date.date())})"
        )
    return (
        f"No -- first obs Wave {first_obs_wave} ({str(first_obs_date.date())}) "
        f"was not before first diagnosis Wave {first_cond_wave} ({str(first_cond_date.date())})"
    )


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
    # v2 aggregate batch 2
    "bp_reading_count":             handle_bp_reading_count,
    "first_condition_ever":         handle_first_condition_ever,
    "wave_coverage_fraction":       handle_wave_coverage_fraction,
    "most_active_wave":             handle_most_active_wave,
    "total_observation_types":      handle_total_observation_types,
    "total_rx_burden":              handle_total_rx_burden,
    # new aggregate + multi-hop batch 3
    "height_stability":             handle_height_stability,
    "glucose_at_diabetes":          handle_glucose_at_diabetes,
    "obs_count_at_diagnosis":       handle_obs_count_at_diagnosis,
    "sbp_after_htn_wave":           handle_sbp_after_htn_wave,
    "bmi_obese_at_htn":             handle_bmi_obese_at_htn,
    "two_conditions_same_wave":     handle_two_conditions_same_wave,
    "hr_around_diagnosis":          handle_hr_around_diagnosis,
    "medication_stopped":           handle_medication_stopped,
    "hr_vs_bmi_peak":               handle_hr_vs_bmi_peak,
    "new_med_per_new_condition":    handle_new_med_per_new_condition,
    "longest_condition_gap":        handle_longest_condition_gap,
    "clinical_intensity_last":      handle_clinical_intensity_last,
    "cardiovascular_vs_metabolic":  handle_cardiovascular_vs_metabolic,
    # batch 4: 12 remaining failure fixes
    "medication_duration":          handle_medication_duration,
    "condition_rate":               handle_condition_rate,
    "medication_gap_waves":         handle_medication_gap_waves,
    "weight_at_first_med":          handle_weight_at_first_med,
    "htn_before_medication":        handle_htn_before_medication,
    "sbp_before_after_medication":  handle_sbp_before_after_medication,
    "weight_at_diagnosis":          handle_weight_at_diagnosis,
    "bmi_at_first_med":             handle_bmi_at_first_med,
    "conditions_by_decade":         handle_conditions_by_decade,
    "obs_before_first_condition":   handle_obs_before_first_condition,
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
