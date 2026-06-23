"""
src/patient_deep_dive.py

Cohort deep-dive: selects 3 patients and runs
  * 15-question analysis on Patient 1 (Star)
  * 5-question analysis on Patients 2 & 3 (Contrast, Comorbid)

Data source: raw CSVs loaded with pandas (no graph pickle — memory-efficient).
No Groq API calls anywhere.
"""

import sys
import json
from pathlib import Path
from collections import defaultdict

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ─────────────────────────────────────────────────────────────────────────────
# Paths & constants
# ─────────────────────────────────────────────────────────────────────────────
DATA_DIR     = Path(__file__).parent.parent / "data" / "raw"
CHART_DIR    = Path(__file__).parent.parent / "benchmark" / "charts"
RESULTS_PATH = Path(__file__).parent.parent / "benchmark" / "patient_deep_dive_results.json"
SUMMARY_PATH = Path(__file__).parent.parent / "benchmark" / "cohort_summary.txt"

WAVE_RANGES = {
    1: (1990, 1994),
    2: (1995, 1999),
    3: (2000, 2004),
    4: (2005, 2009),
    5: (2010, 2014),
    6: (2015, 2019),
    7: (2020, 2026),
}
WAVE_LABEL = {w: f"W{w} ({s}-{e})" for w, (s, e) in WAVE_RANGES.items()}

BMI_DESC = "Body Mass Index"
SBP_DESC = "Systolic Blood Pressure"

PATIENT_COLOURS = ["#4e79a7", "#f28e2b", "#e15759"]


# ─────────────────────────────────────────────────────────────────────────────
# Wave helpers
# ─────────────────────────────────────────────────────────────────────────────
def _assign_wave(date_str) -> int | None:
    try:
        y = int(str(date_str).strip()[:4])
        for w, (s, e) in WAVE_RANGES.items():
            if s <= y <= e:
                return w
        return None
    except (ValueError, TypeError):
        return None


def _year(date_str) -> int | None:
    try:
        return int(str(date_str).strip()[:4])
    except (ValueError, TypeError):
        return None


def _active_waves(start_str, stop_str) -> list[int]:
    """Waves in which a condition is active (interval overlap check)."""
    sy = _year(start_str)
    if sy is None:
        return []
    stop_raw = str(stop_str).strip()
    ey = _year(stop_str) if stop_raw not in ("nan", "", "None", "NaT") else 2026
    if ey is None:
        ey = 2026
    return [w for w, (ws, we) in WAVE_RANGES.items() if sy <= we and ey >= ws]


def _safe_float(s) -> float | None:
    try:
        return float(s)
    except (ValueError, TypeError):
        return None


def _avg(vals: list) -> float | None:
    return round(sum(vals) / len(vals), 1) if vals else None


# ─────────────────────────────────────────────────────────────────────────────
# Condition matchers
# ─────────────────────────────────────────────────────────────────────────────
def _is_htn(desc: str) -> bool:
    return desc.strip() == "Hypertension"


def _is_diabetes(desc: str) -> bool:
    d = desc.lower()
    return "diabetes" in d and "prediabetes" not in d


def _is_obesity(desc: str) -> bool:
    d = desc.lower()
    return "obes" in d or "body mass index 30+" in d or "body mass index 40+" in d


# ─────────────────────────────────────────────────────────────────────────────
# Load CSVs and assign wave numbers
# ─────────────────────────────────────────────────────────────────────────────
def load_dataframes() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    print("Loading CSVs ...")
    patients_df  = pd.read_csv(DATA_DIR / "patients.csv",     dtype=str)
    conds_df     = pd.read_csv(DATA_DIR / "conditions.csv",   dtype=str)
    obs_df       = pd.read_csv(DATA_DIR / "observations.csv", dtype=str)
    meds_df      = pd.read_csv(DATA_DIR / "medications.csv",  dtype=str)

    conds_df["wave"] = conds_df["START"].apply(_assign_wave)
    obs_df["wave"]   = obs_df["DATE"].apply(_assign_wave)
    meds_df["wave"]  = meds_df["START"].apply(_assign_wave)

    print(f"  patients={len(patients_df):,}  conditions={len(conds_df):,}  "
          f"observations={len(obs_df):,}  medications={len(meds_df):,}")
    return patients_df, conds_df, obs_df, meds_df


# ─────────────────────────────────────────────────────────────────────────────
# Patient selection (pure pandas, no graph)
# ─────────────────────────────────────────────────────────────────────────────
def select_cohort(patients_df, conds_df, obs_df, meds_df) -> dict:
    print("Scanning patients for cohort selection ...")

    stats: dict[str, dict] = {}

    for _, prow in patients_df.iterrows():
        full_id  = str(prow["Id"])
        short_id = full_id[:8]
        name     = f"{prow.get('FIRST', '')} {prow.get('LAST', '')}".strip()

        pc = conds_df[conds_df["PATIENT"] == full_id]
        po = obs_df[obs_df["PATIENT"] == full_id]
        pm = meds_df[meds_df["PATIENT"] == full_id]

        # Wave count = union of waves from all tables (non-null)
        c_waves = set(pc["wave"].dropna().astype(int))
        o_waves = set(po["wave"].dropna().astype(int))
        m_waves = set(pm["wave"].dropna().astype(int))
        data_waves = sorted(c_waves | o_waves | m_waves)
        wave_count = len(data_waves)

        unique_conds = set(pc["DESCRIPTION"].dropna())
        has_htn  = any(_is_htn(d)     for d in unique_conds)
        has_dm   = any(_is_diabetes(d) for d in unique_conds)

        score = (wave_count * 3) + len(unique_conds) + (2 if has_htn else 0) + (2 if has_dm else 0)

        stats[short_id] = {
            "full_id":       full_id,
            "name":          name,
            "wave_count":    wave_count,
            "data_waves":    data_waves,
            "unique_conds":  len(unique_conds),
            "has_htn":       has_htn,
            "has_dm":        has_dm,
            "score":         score,
        }

    # Patient 1 — Star: highest score
    star_id, star_meta = max(stats.items(), key=lambda x: x[1]["score"])

    # Patient 2 — Contrast: most waves, lowest condition count
    max_waves = max(v["wave_count"] for v in stats.values())
    contrast_id, contrast_meta = min(
        ((k, v) for k, v in stats.items() if v["wave_count"] == max_waves),
        key=lambda x: x[1]["unique_conds"],
    )

    # Patient 3 — Comorbid: both HTN + DM, highest wave count
    comorbid_pool = [(k, v) for k, v in stats.items() if v["has_htn"] and v["has_dm"]]
    if not comorbid_pool:
        raise RuntimeError("No patient with both Hypertension and Diabetes found.")
    comorbid_id, comorbid_meta = max(comorbid_pool, key=lambda x: x[1]["wave_count"])
    # Avoid duplicating the Star patient
    if comorbid_id == star_id:
        remaining = [(k, v) for k, v in comorbid_pool if k != star_id]
        if remaining:
            comorbid_id, comorbid_meta = max(remaining, key=lambda x: x[1]["wave_count"])

    return {
        "star":     (star_id,     star_meta),
        "contrast": (contrast_id, contrast_meta),
        "comorbid": (comorbid_id, comorbid_meta),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Per-patient data collection
# ─────────────────────────────────────────────────────────────────────────────
def collect_patient_data(full_id: str, conds_df, obs_df, meds_df) -> dict:
    """Build structured data dict for one patient from filtered DataFrames."""
    pc = conds_df[conds_df["PATIENT"] == full_id]
    po = obs_df[obs_df["PATIENT"] == full_id]
    pm = meds_df[meds_df["PATIENT"] == full_id]

    # Convert rows to attribute dicts (matching graph node format used by analysis fns)
    conds_raw = [
        {
            "description": str(r.get("DESCRIPTION", "")),
            "start":       str(r.get("START", "")),
            "stop":        str(r.get("STOP", "")),
            "code":        str(r.get("CODE", "")),
            "wave":        int(r["wave"]) if pd.notna(r.get("wave")) else None,
        }
        for _, r in pc.iterrows()
        if pd.notna(r.get("wave"))
    ]

    obs_raw = [
        {
            "description": str(r.get("DESCRIPTION", "")),
            "value":       str(r.get("VALUE", "")),
            "units":       str(r.get("UNITS", "")),
            "date":        str(r.get("DATE", "")),
            "wave":        int(r["wave"]) if pd.notna(r.get("wave")) else None,
        }
        for _, r in po.iterrows()
        if pd.notna(r.get("wave"))
    ]

    meds_raw = [
        {
            "description": str(r.get("DESCRIPTION", "")),
            "start":       str(r.get("START", "")),
            "stop":        str(r.get("STOP", "")),
            "wave":        int(r["wave"]) if pd.notna(r.get("wave")) else None,
        }
        for _, r in pm.iterrows()
        if pd.notna(r.get("wave"))
    ]

    # Wave-bucketed structures
    conds_by_wave: dict[int, list] = defaultdict(list)
    meds_by_wave:  dict[int, set]  = defaultdict(set)
    bmi_by_wave:   dict[int, list] = defaultdict(list)
    sbp_by_wave:   dict[int, list] = defaultdict(list)

    for c in conds_raw:
        conds_by_wave[c["wave"]].append(c)

    for m in meds_raw:
        meds_by_wave[m["wave"]].add(m["description"])

    for o in obs_raw:
        v = _safe_float(o["value"])
        if v is None:
            continue
        if o["description"] == BMI_DESC:
            bmi_by_wave[o["wave"]].append(v)
        elif o["description"] == SBP_DESC:
            sbp_by_wave[o["wave"]].append(v)

    data_waves = sorted(
        set(conds_by_wave) | set(meds_by_wave) | set(bmi_by_wave) | set(sbp_by_wave)
    )
    return {
        "conds_raw":     conds_raw,
        "obs_raw":       obs_raw,
        "meds_raw":      meds_raw,
        "conds_by_wave": dict(conds_by_wave),
        "meds_by_wave":  dict(meds_by_wave),
        "bmi_by_wave":   dict(bmi_by_wave),
        "sbp_by_wave":   dict(sbp_by_wave),
        "data_waves":    data_waves,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Analysis helpers (all operate on the data dict, never on graph/DataFrames)
# ─────────────────────────────────────────────────────────────────────────────
def _unique_conditions(data: dict) -> set[str]:
    return {c["description"] for c in data["conds_raw"]}


def _peak_bmi(data: dict) -> tuple[float | None, int | None]:
    best_v, best_w = None, None
    for w, vals in data["bmi_by_wave"].items():
        avg = _avg(vals)
        if avg is not None and (best_v is None or avg > best_v):
            best_v, best_w = avg, w
    return best_v, best_w


def _new_conditions_per_wave(data: dict) -> dict[int, list[str]]:
    seen: set[str] = set()
    new_pw: dict[int, list[str]] = {}
    for w in sorted(data["conds_by_wave"]):
        descs = {c["description"] for c in data["conds_by_wave"][w]}
        first = sorted(descs - seen)
        new_pw[w] = first
        seen |= descs
    return new_pw


def _cumulative_conditions_per_wave(data: dict) -> dict[int, int]:
    seen: set[str] = set()
    cumul: dict[int, int] = {}
    for w in sorted(data["conds_by_wave"]):
        seen |= {c["description"] for c in data["conds_by_wave"][w]}
        cumul[w] = len(seen)
    return cumul


def _first_wave_for(data: dict, pred) -> int | None:
    for w in sorted(data["conds_by_wave"]):
        for c in data["conds_by_wave"][w]:
            if pred(c["description"]):
                return w
    return None


def _most_persistent(data: dict) -> tuple[str, list[int]]:
    persistence: dict[str, set[int]] = defaultdict(set)
    for c in data["conds_raw"]:
        persistence[c["description"]] |= set(_active_waves(c["start"], c["stop"]))
    if not persistence:
        return ("(none)", [])
    best = max(persistence, key=lambda d: len(persistence[d]))
    return best, sorted(persistence[best])


# ─────────────────────────────────────────────────────────────────────────────
# Full 15-question analysis — Patient 1 (Star)
# ─────────────────────────────────────────────────────────────────────────────
def analyze_star(data: dict, short_id: str, name: str) -> dict:
    result: dict = {"patient_id": short_id, "name": name}

    def _p(label: str, value):
        print(f"  {label}: {value}")
        result[label] = value

    print(f"\n{'='*64}")
    print(f"PATIENT 1 -- STAR  ({short_id} / {name})")
    print(f"{'='*64}")

    # ── FACTS ─────────────────────────────────────────────────────────────
    print("\n-- FACTS --")

    unique_conds = _unique_conditions(data)
    _p("1. Total unique conditions", len(unique_conds))
    _p("2. Total observations",      len(data["obs_raw"]))
    unique_meds = {m["description"] for m in data["meds_raw"]}
    _p("3. Total unique medications", len(unique_meds))
    _p("4. Waves with data", f"{len(data['data_waves'])}  {data['data_waves']}")

    peak_bmi, peak_wave = _peak_bmi(data)
    _p("5. Peak BMI",
       f"{peak_bmi} in Wave {peak_wave} ({WAVE_LABEL.get(peak_wave, '?')})"
       if peak_bmi else "No BMI data")

    # ── TRENDS ────────────────────────────────────────────────────────────
    print("\n-- TRENDS --")

    bmi_trend = {w: _avg(v) for w, v in sorted(data["bmi_by_wave"].items())}
    result["bmi_per_wave"] = bmi_trend
    _p("6. BMI per wave",
       ", ".join(f"Wave {w}: {v}" for w, v in bmi_trend.items()) or "No BMI data")

    cumul = _cumulative_conditions_per_wave(data)
    result["cumul_conditions"] = cumul
    is_mono = all(
        cumul[w2] > cumul[w1]
        for w1, w2 in zip(sorted(cumul)[:-1], sorted(cumul)[1:])
    ) if len(cumul) > 1 else True
    cumul_str = ", ".join(f"Wave {w}: {n}" for w, n in sorted(cumul.items()))
    _p("7. Condition count monotonically increasing?",
       f"{'YES' if is_mono else 'NO'}  --  cumulative: {cumul_str}")

    new_pw = _new_conditions_per_wave(data)
    result["new_per_wave"] = {w: v for w, v in new_pw.items()}
    peak_n = max((len(v) for v in new_pw.values()), default=0)
    peak_wl = [w for w, v in new_pw.items() if len(v) == peak_n]
    if len(peak_wl) == 1:
        peak_new_str = f"Wave {peak_wl[0]}"
    else:
        peak_new_str = f"Wave {peak_wl[0]} and Wave {peak_wl[1]} (tied)"
    _p("8. Wave with most NEW conditions first diagnosed",
       f"{peak_new_str} ({peak_n} new)")

    # ── PATTERNS ──────────────────────────────────────────────────────────
    print("\n-- PATTERNS --")

    persist_desc, persist_waves = _most_persistent(data)
    _p("9. Most persistent condition",
       f'"{persist_desc}" -- active in {len(persist_waves)} waves: {persist_waves}')

    htn_wave   = _first_wave_for(data, _is_htn)
    obese_wave = _first_wave_for(data, _is_obesity)
    if htn_wave and obese_wave:
        if htn_wave < obese_wave:
            rel = f"Hypertension (Wave {htn_wave}) BEFORE obesity (Wave {obese_wave})"
        elif htn_wave > obese_wave:
            rel = f"Obesity (Wave {obese_wave}) BEFORE hypertension (Wave {htn_wave})"
        else:
            # Compare exact start dates within the same wave
            htn_date = min(
                (c["start"] for c in data["conds_by_wave"].get(htn_wave, [])
                 if _is_htn(c["description"])),
                default="unknown"
            )
            obs_date = min(
                (c["start"] for c in data["conds_by_wave"].get(obese_wave, [])
                 if _is_obesity(c["description"])),
                default="unknown"
            )
            if htn_date < obs_date:
                rel = f"Hypertension ({htn_date}) before obesity ({obs_date}) within Wave {htn_wave}"
            elif obs_date < htn_date:
                rel = f"Obesity ({obs_date}) before hypertension ({htn_date}) within Wave {htn_wave}"
            else:
                rel = f"Hypertension and obesity on same date in Wave {htn_wave}"
    elif htn_wave:
        rel = f"Only Hypertension (Wave {htn_wave}); no obesity on record"
    elif obese_wave:
        rel = f"Only Obesity (Wave {obese_wave}); no hypertension on record"
    else:
        rel = "Neither hypertension nor obesity found"
    _p("10. Hypertension vs obesity timing", rel)

    # Q11: wave where BOTH med count increased AND new conditions appeared
    med_seq = [(w, len(data["meds_by_wave"].get(w, set())))
               for w in sorted(data["data_waves"])]
    spike_waves = []
    for i in range(1, len(med_seq)):
        pw, pm_ = med_seq[i - 1]
        cw, cm  = med_seq[i]
        nc = len(new_pw.get(cw, []))
        if cm > pm_ and nc > 0:
            spike_waves.append((cw, cm - pm_, nc))
    if spike_waves:
        spike_str = "; ".join(
            f"Wave {w} (+{dm} meds, {nc} new conds)" for w, dm, nc in spike_waves
        )
    else:
        spike_str = "No wave showed simultaneous medication and condition increase"
    _p("11. Med spike coincides with new conditions", spike_str)

    # ── HYPOTHESES ────────────────────────────────────────────────────────
    print("\n-- HYPOTHESES --")

    sbp_trend = {w: _avg(v) for w, v in sorted(data["sbp_by_wave"].items())}
    overlap   = sorted(set(bmi_trend) & set(sbp_trend))
    if overlap:
        ref_w = peak_wave if peak_wave in overlap else overlap[-1]
        bv, sv = bmi_trend[ref_w], sbp_trend[ref_w]
        both_high = (bv is not None and bv >= 25) and (sv is not None and sv >= 120)
        assessment = "consistent with" if both_high else "inconsistent with"
        h12 = (f"In Wave {ref_w}, BMI was {bv} and SBP was {sv} -- "
               f"{assessment} concurrent metabolic-cardiovascular deterioration")
    else:
        h12 = "Insufficient overlapping BMI + SBP data to assess co-movement"
    _p("12. BMI-SBP co-movement", h12)

    if cumul:
        fw, lw = min(cumul), max(cumul)
        c0, cN = cumul[fw], cumul[lw]
        fold = round(cN / c0, 1) if c0 else "N/A"
        if isinstance(fold, float) and fold >= 3:
            finding = "high cumulative disease burden accumulation"
        elif isinstance(fold, float) and fold >= 2:
            finding = "moderate cumulative burden growth"
        else:
            finding = "relatively stable condition burden"
        h13 = (f"Condition burden grew from {c0} in Wave {fw} to {cN} in Wave {lw} "
               f"-- a {fold}-fold increase suggesting {finding}")
    else:
        h13 = "No condition data for burden assessment"
    _p("13. Cumulative burden", h13)

    fcw = min(data["conds_by_wave"]) if data["conds_by_wave"] else None
    fmw = min(data["meds_by_wave"])  if data["meds_by_wave"]  else None
    if fmw and fcw:
        diff = fmw - fcw
        lag  = ("same wave as" if diff == 0 else
                f"{abs(diff)} wave(s) {'after' if diff > 0 else 'before'}")
        h14 = (f"First medication in Wave {fmw}, "
               f"{lag} first condition diagnosis (Wave {fcw})")
    elif fmw:
        h14 = f"First medication in Wave {fmw}; no condition data"
    else:
        h14 = "No medication data"
    _p("14. Medication lag", h14)

    dm_wave = _first_wave_for(data, _is_diabetes)
    narrative = _build_narrative(
        short_id, name, unique_conds, data["data_waves"],
        peak_bmi, peak_wave, bmi_trend, sbp_trend,
        cumul, new_pw, htn_wave, obese_wave, dm_wave,
        persist_desc, persist_waves, fcw, fmw, spike_waves,
    )
    _p("15. Clinical narrative", narrative)

    return result


def _build_narrative(
    pid, name, unique_conds, data_waves,
    peak_bmi, peak_wave, bmi_trend, sbp_trend,
    cumul, new_pw, htn_wave, obese_wave, dm_wave,
    persist_desc, persist_waves, fcw, fmw, spike_waves,
) -> str:
    parts = []
    wspan = (f"waves {min(data_waves)} through {max(data_waves)}"
             if data_waves else "the observed period")
    parts.append(
        f"Patient {pid} ({name}) presents a longitudinal profile spanning "
        f"{len(data_waves)} wave(s) ({wspan}), with {len(unique_conds)} unique conditions."
    )
    flags = []
    if htn_wave:  flags.append(f"hypertension (first Wave {htn_wave})")
    if dm_wave:   flags.append(f"diabetes (Wave {dm_wave})")
    if obese_wave: flags.append(f"obesity (Wave {obese_wave})")
    if flags:
        parts.append("Record shows " + ", ".join(flags) + ", indicating metabolic comorbidity.")
    if bmi_trend:
        vals = list(bmi_trend.values())
        direction = ("declining" if vals[-1] < vals[0] else
                     "increasing" if vals[-1] > vals[0] else "stable")
        parts.append(
            f"BMI peaked at {peak_bmi} in Wave {peak_wave} and trended {direction}."
        )
    if cumul:
        wl = sorted(cumul)
        flat = [wl[i] for i in range(1, len(wl)) if cumul[wl[i]] == cumul[wl[i-1]]]
        if flat:
            parts.append(f"Condition accumulation plateaued in Wave(s) {flat}, "
                         "suggesting periods of stability.")
        else:
            parts.append("Condition burden grew in every observed wave without plateau.")
    if persist_desc and len(persist_waves) >= 2:
        parts.append(
            f'"{persist_desc}" was most persistent, active across {len(persist_waves)} waves.'
        )
    if spike_waves:
        sw = ", ".join(f"Wave {w}" for w, _, _ in spike_waves)
        parts.append(f"Medication count increased alongside new diagnoses in {sw}, "
                     "consistent with reactive prescribing.")
    return " ".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# Light 5-question analysis — Patients 2 & 3
# ─────────────────────────────────────────────────────────────────────────────
def analyze_light(data: dict, short_id: str, name: str, role: str) -> dict:
    result: dict = {"patient_id": short_id, "name": name, "role": role}

    def _p(label: str, value):
        print(f"  {label}: {value}")
        result[label] = value

    print(f"\n{'='*64}")
    print(f"{role.upper()}  ({short_id} / {name})")
    print(f"{'='*64}")

    _p("1. Total unique conditions", len(_unique_conditions(data)))
    _p("2. Waves with data", f"{len(data['data_waves'])}  {data['data_waves']}")

    peak_bmi, peak_wave = _peak_bmi(data)
    _p("3. Peak BMI",
       f"{peak_bmi} in Wave {peak_wave} ({WAVE_LABEL.get(peak_wave, '?')})"
       if peak_bmi else "No BMI data")

    htn_wave = _first_wave_for(data, _is_htn)
    dm_wave  = _first_wave_for(data, _is_diabetes)
    htn_str  = f"Hypertension first Wave {htn_wave}" if htn_wave else "Hypertension: not diagnosed"
    dm_str   = f"Diabetes first Wave {dm_wave}"      if dm_wave  else "Diabetes: not diagnosed"
    _p("4. HTN / Diabetes timing", f"{htn_str}  |  {dm_str}")

    cumul = _cumulative_conditions_per_wave(data)
    result["cumul_conditions"] = cumul
    result["bmi_per_wave"]     = {w: _avg(v) for w, v in data["bmi_by_wave"].items()}
    if len(cumul) > 1:
        vals = [cumul[w] for w in sorted(cumul)]
        direction = ("monotonically increasing"
                     if all(b > a for a, b in zip(vals, vals[1:]))
                     else "non-monotonic (flat or declining in some waves)")
    else:
        direction = "single wave only"
    _p("5. Condition burden trajectory",
       f"{direction}  --  " +
       ", ".join(f"Wave {w}: {n}" for w, n in sorted(cumul.items())))

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Charts
# ─────────────────────────────────────────────────────────────────────────────
def _wx(waves: list[int]) -> list[str]:
    return [f"W{w}" for w in waves]


def chart_bmi_trajectory(data: dict, short_id: str, name: str, peak_wave: int | None):
    bmi = {w: _avg(v) for w, v in sorted(data["bmi_by_wave"].items())}
    if not bmi:
        return
    ws  = sorted(bmi)
    vs  = [bmi[w] for w in ws]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(_wx(ws), vs, marker="o", color=PATIENT_COLOURS[0], linewidth=2)
    if peak_wave in bmi:
        pi = ws.index(peak_wave)
        ax.plot(_wx(ws)[pi], vs[pi], marker="*", color="red", markersize=14,
                zorder=5, label=f"Peak (W{peak_wave})")
        ax.annotate(f"Peak\n{vs[pi]}", (_wx(ws)[pi], vs[pi]),
                    textcoords="offset points", xytext=(8, 6), fontsize=8)
    ax.axhline(25, color="orange", linestyle="--", linewidth=1, alpha=0.6, label="BMI 25")
    ax.axhline(30, color="red",    linestyle="--", linewidth=1, alpha=0.6, label="BMI 30")
    ax.set_title(f"BMI Trajectory -- {short_id} ({name})", fontsize=11)
    ax.set_xlabel("Wave"); ax.set_ylabel("Body Mass Index (kg/m^2)")
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(CHART_DIR / "bmi_trajectory.png", dpi=130)
    plt.close(fig)
    print("  Saved bmi_trajectory.png")


def chart_condition_burden(data: dict, short_id: str, name: str):
    new_pw = _new_conditions_per_wave(data)
    if not new_pw:
        return
    ws  = sorted(new_pw)
    cnts = [len(new_pw[w]) for w in ws]
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(_wx(ws), cnts, color=PATIENT_COLOURS[0], alpha=0.85, edgecolor="white")
    for bar, c in zip(bars, cnts):
        if c:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                    str(c), ha="center", va="bottom", fontsize=9)
    ax.set_title(f"New Conditions per Wave -- {short_id} ({name})", fontsize=11)
    ax.set_xlabel("Wave"); ax.set_ylabel("New conditions first diagnosed")
    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(CHART_DIR / "condition_burden.png", dpi=130)
    plt.close(fig)
    print("  Saved condition_burden.png")


def chart_medication_burden(data: dict, short_id: str, name: str):
    med_counts = {w: len(s) for w, s in sorted(data["meds_by_wave"].items())}
    if not med_counts:
        return
    ws  = sorted(med_counts)
    cnts = [med_counts[w] for w in ws]
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(_wx(ws), cnts, color="#59a14f", alpha=0.85, edgecolor="white")
    for bar, c in zip(bars, cnts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                str(c), ha="center", va="bottom", fontsize=9)
    ax.set_title(f"Medication Count per Wave -- {short_id} ({name})", fontsize=11)
    ax.set_xlabel("Wave"); ax.set_ylabel("Unique medications")
    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(CHART_DIR / "medication_burden.png", dpi=130)
    plt.close(fig)
    print("  Saved medication_burden.png")


def chart_cohort_bmi(entries: list[tuple[dict, str, str]]):
    """Three lines: (data, short_id, label)."""
    fig, ax = plt.subplots(figsize=(9, 5))
    has_data = False
    for i, (data, short_id, label) in enumerate(entries):
        bmi = {w: _avg(v) for w, v in sorted(data["bmi_by_wave"].items())}
        if not bmi:
            continue
        has_data = True
        ws = sorted(bmi)
        vs = [bmi[w] for w in ws]
        ax.plot(_wx(ws), vs, marker="o", color=PATIENT_COLOURS[i],
                linewidth=2, label=f"{short_id} ({label})", alpha=0.9)
    if not has_data:
        plt.close(fig)
        return
    ax.axhline(25, color="orange", linestyle="--", linewidth=1, alpha=0.5, label="BMI 25")
    ax.axhline(30, color="red",    linestyle="--", linewidth=1, alpha=0.5, label="BMI 30")
    ax.set_title("Cohort BMI Comparison", fontsize=12)
    ax.set_xlabel("Wave"); ax.set_ylabel("Body Mass Index (kg/m^2)")
    ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(CHART_DIR / "cohort_bmi_comparison.png", dpi=130)
    plt.close(fig)
    print("  Saved cohort_bmi_comparison.png")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    CHART_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    patients_df, conds_df, obs_df, meds_df = load_dataframes()

    # Select cohort
    cohort = select_cohort(patients_df, conds_df, obs_df, meds_df)
    star_id,     star_meta     = cohort["star"]
    contrast_id, contrast_meta = cohort["contrast"]
    comorbid_id, comorbid_meta = cohort["comorbid"]

    print("\n" + "-" * 64)
    print("SELECTED COHORT")
    print("-" * 64)
    print(f"  P1 STAR     : {star_id} ({star_meta['name']})")
    print(f"    score={star_meta['score']}  "
          f"(waves={star_meta['wave_count']}x3 + conds={star_meta['unique_conds']} + "
          f"htn={2 if star_meta['has_htn'] else 0} + dm={2 if star_meta['has_dm'] else 0})")
    print(f"  P2 CONTRAST : {contrast_id} ({contrast_meta['name']})")
    print(f"    {contrast_meta['wave_count']} waves (max), "
          f"only {contrast_meta['unique_conds']} condition(s) -- most stable")
    print(f"  P3 COMORBID : {comorbid_id} ({comorbid_meta['name']})")
    print(f"    HTN + Diabetes; {comorbid_meta['wave_count']} waves, "
          f"{comorbid_meta['unique_conds']} conditions")
    print("-" * 64)

    # Collect per-patient data
    star_data     = collect_patient_data(star_meta["full_id"],     conds_df, obs_df, meds_df)
    contrast_data = collect_patient_data(contrast_meta["full_id"], conds_df, obs_df, meds_df)
    comorbid_data = collect_patient_data(comorbid_meta["full_id"], conds_df, obs_df, meds_df)

    # Run analyses
    star_res     = analyze_star(star_data, star_id, star_meta["name"])
    contrast_res = analyze_light(contrast_data, contrast_id, contrast_meta["name"],
                                 "Patient 2 -- Contrast")
    comorbid_res = analyze_light(comorbid_data, comorbid_id, comorbid_meta["name"],
                                 "Patient 3 -- Comorbid")

    # Charts
    print(f"\n-- Generating charts -> {CHART_DIR} --")
    _, peak_wave = _peak_bmi(star_data)
    chart_bmi_trajectory(star_data,     star_id, star_meta["name"], peak_wave)
    chart_condition_burden(star_data,   star_id, star_meta["name"])
    chart_medication_burden(star_data,  star_id, star_meta["name"])
    chart_cohort_bmi([
        (star_data,     star_id,     "Star"),
        (contrast_data, contrast_id, "Contrast"),
        (comorbid_data, comorbid_id, "Comorbid"),
    ])

    # Save JSON
    output = {
        "selected_patients": {
            "star": {
                "short_id": star_id, "name": star_meta["name"],
                "selection_reason": (
                    f"Highest composite score ({star_meta['score']}): "
                    f"waves={star_meta['wave_count']}, conds={star_meta['unique_conds']}, "
                    f"htn={star_meta['has_htn']}, dm={star_meta['has_dm']}"
                ),
            },
            "contrast": {
                "short_id": contrast_id, "name": contrast_meta["name"],
                "selection_reason": (
                    f"Most waves ({contrast_meta['wave_count']}) with fewest conditions "
                    f"({contrast_meta['unique_conds']}) -- most stable"
                ),
            },
            "comorbid": {
                "short_id": comorbid_id, "name": comorbid_meta["name"],
                "selection_reason": (
                    f"Both Hypertension + Diabetes; highest wave count "
                    f"({comorbid_meta['wave_count']}) among comorbid patients"
                ),
            },
        },
        "patient_1_star":     star_res,
        "patient_2_contrast": contrast_res,
        "patient_3_comorbid": comorbid_res,
    }
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved -> {RESULTS_PATH}")

    # Save cohort narrative text
    star_narr = star_res.get("15. Clinical narrative", "N/A")
    contrast_narr = (
        f"Patient {contrast_id} ({contrast_meta['name']}) -- Contrast. "
        f"Spans {contrast_meta['wave_count']} waves with only {contrast_meta['unique_conds']} "
        f"condition(s), making them the most clinically stable patient in the cohort. "
        f"{'BMI data available.' if contrast_data['bmi_by_wave'] else 'No BMI data.'} "
        "Serves as the control archetype for longitudinal stability."
    )
    comorbid_narr = (
        f"Patient {comorbid_id} ({comorbid_meta['name']}) -- Comorbid. "
        f"Carries both Hypertension and Diabetes across {comorbid_meta['wave_count']} waves "
        f"with {comorbid_meta['unique_conds']} unique conditions. "
        "Exemplifies the dual-comorbidity burden pattern central to metabolic syndrome research."
    )
    summary = "\n\n".join([
        "COHORT CLINICAL NARRATIVE SUMMARY",
        "=" * 60,
        f"[P1 STAR: {star_id}]", star_narr,
        "=" * 60,
        f"[P2 CONTRAST: {contrast_id}]", contrast_narr,
        "=" * 60,
        f"[P3 COMORBID: {comorbid_id}]", comorbid_narr,
    ])
    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        f.write(summary)
    print(f"Summary saved -> {SUMMARY_PATH}")
    print("\nDone.")
