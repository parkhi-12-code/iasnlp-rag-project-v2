"""
src/generate_v2_benchmark.py
Generates benchmark/qa_pairs_v2.csv — 50 unique benchmark questions.
  AGG_V2_001..025 — Aggregate (25 templates)
  MH_V2_001..025  — Multi-hop (25 templates)
DO NOT modify qa_pairs_final.csv.
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

ROOT  = Path(__file__).parent.parent
DATA  = ROOT / "data" / "raw"
BENCH = ROOT / "benchmark"
BENCH.mkdir(exist_ok=True)

# ── Wave assignment ─────────────────────────────────────────────────────────────
def assign_wave(year) -> int:
    y = int(year)
    if   y <= 1994: return 1
    elif y <= 1999: return 2
    elif y <= 2004: return 3
    elif y <= 2009: return 4
    elif y <= 2014: return 5
    elif y <= 2019: return 6
    else:           return 7

WAVE_BOUNDS = {
    1: (datetime(1990,1,1), datetime(1994,12,31,23,59,59)),
    2: (datetime(1995,1,1), datetime(1999,12,31,23,59,59)),
    3: (datetime(2000,1,1), datetime(2004,12,31,23,59,59)),
    4: (datetime(2005,1,1), datetime(2009,12,31,23,59,59)),
    5: (datetime(2010,1,1), datetime(2014,12,31,23,59,59)),
    6: (datetime(2015,1,1), datetime(2019,12,31,23,59,59)),
    7: (datetime(2020,1,1), datetime(2099,12,31,23,59,59)),
}

# ── Load tables ─────────────────────────────────────────────────────────────────
print("Loading tables...")
patients     = pd.read_csv(DATA / "patients.csv")
conditions   = pd.read_csv(DATA / "conditions.csv")
observations = pd.read_csv(DATA / "observations.csv")
medications  = pd.read_csv(DATA / "medications.csv")

conditions["START"]   = pd.to_datetime(conditions["START"],   errors="coerce")
conditions["STOP"]    = pd.to_datetime(conditions["STOP"],    errors="coerce")
observations["DATE"]  = pd.to_datetime(observations["DATE"],  errors="coerce", utc=True).dt.tz_convert(None)
medications["START"]  = pd.to_datetime(medications["START"],  errors="coerce", utc=True).dt.tz_convert(None)
medications["STOP"]   = pd.to_datetime(medications["STOP"],   errors="coerce", utc=True).dt.tz_convert(None)
patients["BIRTHDATE"] = pd.to_datetime(patients["BIRTHDATE"], errors="coerce")

def _wave(d):
    return assign_wave(d.year) if pd.notna(d) else np.nan

conditions["WAVE"]   = conditions["START"].apply(_wave)
observations["WAVE"] = observations["DATE"].apply(_wave)
medications["WAVE"]  = medications["START"].apply(_wave)

patients["pid8"]     = patients["Id"].str[:8]
conditions["pid8"]   = conditions["PATIENT"].str[:8]
observations["pid8"] = observations["PATIENT"].str[:8]
medications["pid8"]  = medications["PATIENT"].str[:8]

# Precomputed subsets
bmi_obs = observations[observations["DESCRIPTION"] == "Body Mass Index"].copy()
bmi_obs["VALUE"] = pd.to_numeric(bmi_obs["VALUE"], errors="coerce")
bmi_obs = bmi_obs.dropna(subset=["VALUE"])

hr_obs = observations[observations["DESCRIPTION"] == "Heart rate"].copy()
hr_obs["VALUE"] = pd.to_numeric(hr_obs["VALUE"], errors="coerce")
hr_obs = hr_obs.dropna(subset=["VALUE"])

sbp_obs = observations[observations["DESCRIPTION"] == "Systolic Blood Pressure"].copy()
sbp_obs["VALUE"] = pd.to_numeric(sbp_obs["VALUE"], errors="coerce")
sbp_obs = sbp_obs.dropna(subset=["VALUE"])

weight_obs = observations[observations["DESCRIPTION"] == "Body Weight"].copy()
weight_obs["VALUE"] = pd.to_numeric(weight_obs["VALUE"], errors="coerce")
weight_obs = weight_obs.dropna(subset=["VALUE"])

print(f"  {len(patients)} patients | {len(conditions)} conditions | "
      f"{len(observations)} obs | {len(medications)} meds\n")

# ── Helpers ─────────────────────────────────────────────────────────────────────
records: list  = []
skipped: list  = []
used_pids: set = set()

def add_q(qid, category, pid8, wave_ref, question, answer, temporal=True):
    print(f"\n{qid}  [{pid8}]")
    print(f"  Q: {question[:100]}")
    print(f"  A: {answer[:100]}")
    records.append({
        "question_id":    qid,
        "category":       category,
        "patient_id":     pid8,
        "wave_reference": wave_ref,
        "question":       question,
        "answer":         answer,
        "verified":       "correct",
        "notes":          "deterministic",
        "temporal":       temporal,
    })
    used_pids.add(pid8)

def skip(qid, reason):
    print(f"\nSKIP {qid}: {reason}")
    skipped.append((qid, reason))

def _pids(series):
    """Return candidate patient list, preferring unused ones."""
    cands = list(series)
    return [p for p in cands if p not in used_pids] + \
           [p for p in cands if p in used_pids]

def _patient_all_waves(pid8):
    """Set of all waves a patient appears in across all tables."""
    waves = set()
    for tbl in (conditions, observations, medications):
        w = tbl[tbl["pid8"] == pid8]["WAVE"].dropna()
        waves.update(w.astype(int).unique())
    return waves

def _active_conds_in_wave(pid8, wave):
    """Count unique condition descriptions active during wave (not just start)."""
    ws, we = WAVE_BOUNDS[wave]
    sub = conditions[conditions["pid8"] == pid8]
    act = sub[
        (sub["START"] <= we) &
        (sub["STOP"].isna() | (sub["STOP"] >= ws))
    ]
    return act["DESCRIPTION"].nunique(), act

# ══════════════════════════════════════════════════════════════════════════════
# SECTION A — AGGREGATE (AGG_V2_001 to AGG_V2_025)
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 65)
print("SECTION A — AGGREGATE (25 questions)")
print("=" * 65)

# ── A1: Peak BMI wave ──────────────────────────────────────────────────────────
bmi_by_wave_all = bmi_obs.groupby(["pid8", "WAVE"])["VALUE"].mean().reset_index()
bmi_n_waves     = bmi_by_wave_all.groupby("pid8")["WAVE"].count()
cands = _pids(bmi_n_waves[bmi_n_waves >= 3].index)
gen = False
for pid8 in cands:
    sub = bmi_by_wave_all[bmi_by_wave_all["pid8"] == pid8]
    best = sub.loc[sub["VALUE"].idxmax()]
    w, v = int(best["WAVE"]), round(float(best["VALUE"]), 1)
    if 15 < v < 70:
        add_q("AGG_V2_001", "aggregate", pid8, f"Wave {w}",
              f"In which wave did patient {pid8} record their highest average Body Mass Index?",
              f"Wave {w} (avg BMI = {v} kg/m2)")
        gen = True; break
if not gen: skip("AGG_V2_001", "No qualifying patient")

# ── A2: Most medication-heavy wave ────────────────────────────────────────────
med_by_wave = medications.groupby(["pid8", "WAVE"])["DESCRIPTION"].nunique().reset_index()
med_by_wave.columns = ["pid8", "WAVE", "n_meds"]
med_wave_count = med_by_wave.groupby("pid8")["WAVE"].count()
cands = _pids(med_wave_count[med_wave_count >= 3].index)
gen = False
for pid8 in cands:
    sub = med_by_wave[med_by_wave["pid8"] == pid8]
    best = sub.loc[sub["n_meds"].idxmax()]
    w, n = int(best["WAVE"]), int(best["n_meds"])
    if n >= 2:
        add_q("AGG_V2_002", "aggregate", pid8, f"Wave {w}",
              f"In which wave was patient {pid8} prescribed the most unique medications?",
              f"Wave {w} ({n} unique medications)")
        gen = True; break
if not gen: skip("AGG_V2_002", "No qualifying patient")

# ── A3: Condition-free waves ──────────────────────────────────────────────────
cond_wave_count = conditions.groupby("pid8")["WAVE"].nunique()
cands = _pids(cond_wave_count[cond_wave_count >= 2].index)
gen = False
for pid8 in cands:
    all_w  = _patient_all_waves(pid8)
    cond_w = set(conditions[conditions["pid8"] == pid8]["WAVE"].dropna().astype(int).unique())
    free_w = all_w - cond_w
    n_total = len(all_w)
    if n_total >= 5 and len(free_w) >= 1:
        wave_list = sorted(free_w)
        wl_str = ", ".join(f"Wave {w}" for w in wave_list)
        add_q("AGG_V2_003", "aggregate", pid8, "All waves",
              f"How many waves did patient {pid8} have NO new conditions diagnosed?",
              f"{len(free_w)} waves with no new diagnoses ({wl_str})")
        gen = True; break
if not gen: skip("AGG_V2_003", "No qualifying patient")

# ── A4: Chronic vs acute conditions ──────────────────────────────────────────
cond_total = conditions.groupby("pid8").size()
cands = _pids(cond_total[cond_total >= 5].index)
gen = False
for pid8 in cands:
    sub     = conditions[conditions["pid8"] == pid8]
    total   = len(sub)
    chronic = int(sub["STOP"].isna().sum())
    if chronic >= 2 and chronic < total:
        add_q("AGG_V2_004", "aggregate", pid8, "All waves",
              f"How many of patient {pid8}'s ever-diagnosed conditions were chronic (no recorded stop date)?",
              f"{chronic} chronic conditions out of {total} total",
              temporal=False)
        gen = True; break
if not gen: skip("AGG_V2_004", "No qualifying patient")

# ── A5: Observation density ────────────────────────────────────────────────────
obs_by_wave = observations.groupby(["pid8", "WAVE"]).size().reset_index(name="n")
obs_wave_ct = obs_by_wave.groupby("pid8")["WAVE"].count()
cands = _pids(obs_wave_ct[obs_wave_ct >= 2].index)
gen = False
for pid8 in cands:
    sub   = obs_by_wave[obs_by_wave["pid8"] == pid8]
    total = int(sub["n"].sum())
    n_w   = int(len(sub))
    avg   = round(total / n_w, 1)
    if n_w >= 2 and avg >= 2:
        add_q("AGG_V2_005", "aggregate", pid8, "All waves",
              f"What was the average number of observations per wave for patient {pid8}?",
              f"{avg} observations per wave (total {total} across {n_w} waves)")
        gen = True; break
if not gen: skip("AGG_V2_005", "No qualifying patient")

# ── A6: Medication duration ────────────────────────────────────────────────────
med_total = medications.groupby("pid8").size()
cands = _pids(med_total[med_total >= 5].index)
gen = False
for pid8 in cands:
    sub        = medications[medications["pid8"] == pid8]
    total      = len(sub)
    time_lim   = int(sub["STOP"].notna().sum())
    if time_lim >= 2 and time_lim < total:
        add_q("AGG_V2_006", "aggregate", pid8, "All waves",
              f"How many of patient {pid8}'s medications had a recorded stop date (were time-limited)?",
              f"{time_lim} time-limited out of {total} total",
              temporal=False)
        gen = True; break
if not gen: skip("AGG_V2_006", "No qualifying patient")

# ── A7: Wave span ─────────────────────────────────────────────────────────────
all_dates = pd.concat([
    conditions[["pid8","START"]].rename(columns={"START":"dt"}),
    observations[["pid8","DATE"]].rename(columns={"DATE":"dt"}),
    medications[["pid8","START"]].rename(columns={"START":"dt"}),
])
all_dates = all_dates.dropna()
span_df = all_dates.groupby("pid8")["dt"].agg(["min","max"]).reset_index()
span_df["span_years"] = span_df["max"].dt.year - span_df["min"].dt.year
cands = _pids(span_df.sort_values("span_years", ascending=False)["pid8"])
gen = False
for pid8 in cands:
    row = span_df[span_df["pid8"] == pid8].iloc[0]
    yy  = int(row["span_years"])
    y0  = int(row["min"].year)
    y1  = int(row["max"].year)
    if yy >= 15:
        add_q("AGG_V2_007", "aggregate", pid8, "All waves",
              f"How many years does patient {pid8}'s health record span from first to last recorded event?",
              f"{yy} years ({y0} to {y1})")
        gen = True; break
if not gen: skip("AGG_V2_007", "No qualifying patient")

# ── A8: BMI overweight waves (25–30 kg/m2) ────────────────────────────────────
bmi_wave_ct = bmi_by_wave_all.groupby("pid8")["WAVE"].count()
cands = _pids(bmi_wave_ct[bmi_wave_ct >= 2].index)
gen = False
for pid8 in cands:
    sub     = bmi_by_wave_all[bmi_by_wave_all["pid8"] == pid8]
    ow_rows = sub[(sub["VALUE"] >= 25) & (sub["VALUE"] < 30)]
    if len(ow_rows) >= 1:
        waves_list = sorted(ow_rows["WAVE"].astype(int).tolist())
        wl_str = ", ".join(f"Wave {w}" for w in waves_list)
        add_q("AGG_V2_008", "aggregate", pid8, "All waves",
              f"In how many waves was patient {pid8}'s average BMI in the overweight range (25-30 kg/m2)?",
              f"{len(waves_list)} waves ({wl_str})")
        gen = True; break
if not gen: skip("AGG_V2_008", "No qualifying patient")

# ── A9: Condition accumulation rate ───────────────────────────────────────────
cond_wave_ct2 = conditions.groupby("pid8")["WAVE"].nunique()
cands = _pids(cond_wave_ct2[cond_wave_ct2 >= 4].index)
gen = False
for pid8 in cands:
    sub   = conditions[conditions["pid8"] == pid8]
    total = sub["DESCRIPTION"].nunique()
    n_w   = sub["WAVE"].nunique()
    rate  = round(total / n_w, 1)
    if rate >= 1.0 and n_w >= 4:
        add_q("AGG_V2_009", "aggregate", pid8, "All waves",
              f"On average, how many new conditions was patient {pid8} diagnosed with per wave?",
              f"{rate} new conditions per wave ({total} unique across {n_w} waves)")
        gen = True; break
if not gen: skip("AGG_V2_009", "No qualifying patient")

# ── A10: Most active clinical wave ────────────────────────────────────────────
def _events_by_wave(pid8):
    c = conditions[conditions["pid8"] == pid8].groupby("WAVE").size().rename("c")
    o = observations[observations["pid8"] == pid8].groupby("WAVE").size().rename("o")
    m = medications[medications["pid8"] == pid8].groupby("WAVE").size().rename("m")
    df = pd.concat([c, o, m], axis=1).fillna(0)
    df["total"] = df.sum(axis=1)
    return df

wave_count_per_pid = (
    pd.concat([
        conditions[["pid8","WAVE"]],
        observations[["pid8","WAVE"]],
        medications[["pid8","WAVE"]],
    ]).dropna().groupby("pid8")["WAVE"].nunique()
)
cands = _pids(wave_count_per_pid[wave_count_per_pid >= 4].index)
gen = False
for pid8 in cands:
    edf = _events_by_wave(pid8)
    if edf.empty: continue
    best_w   = int(edf["total"].idxmax())
    best_tot = int(edf["total"].max())
    if best_tot >= 10:
        add_q("AGG_V2_010", "aggregate", pid8, f"Wave {best_w}",
              f"In which wave did patient {pid8} have the most total health events "
              f"(conditions + observations + medications combined)?",
              f"Wave {best_w} ({best_tot} total events)")
        gen = True; break
if not gen: skip("AGG_V2_010", "No qualifying patient")

# ── A11: Blood pressure readings total ────────────────────────────────────────
bp_obs = observations[
    observations["DESCRIPTION"].isin(["Systolic Blood Pressure","Diastolic Blood Pressure"])
]
bp_counts = bp_obs.groupby("pid8").size()
cands = _pids(bp_counts[bp_counts >= 10].index)
gen = False
for pid8 in cands:
    n = int(bp_counts[pid8])
    add_q("AGG_V2_011", "aggregate", pid8, "All waves",
          f"How many total blood pressure readings (systolic or diastolic) were recorded "
          f"for patient {pid8} across all waves?",
          f"{n} blood pressure readings",
          temporal=False)
    gen = True; break
if not gen: skip("AGG_V2_011", "No qualifying patient")

# ── A12: First condition ever ─────────────────────────────────────────────────
cond_wave_ct3 = conditions.groupby("pid8")["WAVE"].nunique()
cands = _pids(cond_wave_ct3[cond_wave_ct3 >= 2].index)
gen = False
for pid8 in cands:
    sub  = conditions[conditions["pid8"] == pid8].sort_values("START")
    first = sub.iloc[0]
    cname = first["DESCRIPTION"]
    w     = int(first["WAVE"]) if pd.notna(first["WAVE"]) else None
    dstr  = first["START"].strftime("%Y-%m-%d") if pd.notna(first["START"]) else "unknown"
    if w is not None:
        add_q("AGG_V2_012", "aggregate", pid8, f"Wave {w}",
              f"What was the first condition ever diagnosed for patient {pid8} and in which wave?",
              f"{cname} (Wave {w}, {dstr})")
        gen = True; break
if not gen: skip("AGG_V2_012", "No qualifying patient")

# ── A13: Medication gap (waves with conditions but no medications) ──────────────
conds_with_waves = cond_wave_ct3
cands = _pids(conds_with_waves[conds_with_waves >= 3].index)
gen = False
for pid8 in cands:
    c_waves = set(conditions[conditions["pid8"] == pid8]["WAVE"].dropna().astype(int).unique())
    m_waves = set(medications[medications["pid8"] == pid8]["WAVE"].dropna().astype(int).unique())
    gaps    = sorted(c_waves - m_waves)
    if gaps:
        gap_strs = []
        for gw in gaps:
            nc = conditions[(conditions["pid8"] == pid8) & (conditions["WAVE"] == gw)]["DESCRIPTION"].nunique()
            gap_strs.append(f"Wave {gw} ({nc} conditions, 0 medications)")
        answer = "Yes — " + "; ".join(gap_strs)
        add_q("AGG_V2_013", "aggregate", pid8, "All waves",
              f"Were there any waves where patient {pid8} had active conditions but NO medications prescribed?",
              answer)
        gen = True; break
if not gen: skip("AGG_V2_013", "No qualifying patient")

# ── A14: Height stability ─────────────────────────────────────────────────────
ht_obs = observations[observations["DESCRIPTION"] == "Body Height"].copy()
ht_obs["VALUE"] = pd.to_numeric(ht_obs["VALUE"], errors="coerce")
ht_obs = ht_obs.dropna(subset=["VALUE"])
ht_wave_ct = ht_obs.groupby("pid8")["WAVE"].nunique()
cands = _pids(ht_wave_ct[ht_wave_ct >= 3].index)
gen = False
for pid8 in cands:
    sub = ht_obs[ht_obs["pid8"] == pid8].groupby("WAVE")["VALUE"].mean()
    mn  = round(float(sub.min()), 1)
    mx  = round(float(sub.max()), 1)
    if mx - mn < 1.0:
        answer = f"Yes — {mn} cm across all {len(sub)} waves"
    else:
        first_w = int(sub.index.min()); last_w = int(sub.index.max())
        answer  = f"No — changed from {round(float(sub[first_w]),1)} cm (Wave {first_w}) to {round(float(sub[last_w]),1)} cm (Wave {last_w})"
    add_q("AGG_V2_014", "aggregate", pid8, "All waves",
          f"Did patient {pid8}'s recorded height remain stable across all waves they appear in?",
          answer)
    gen = True; break
if not gen: skip("AGG_V2_014", "No qualifying patient")

# ── A15: Observation type variety ─────────────────────────────────────────────
obs_types = observations.groupby("pid8")["DESCRIPTION"].nunique()
obs_total = observations.groupby("pid8").size()
cands = _pids(obs_total[obs_total >= 50].index)
gen = False
for pid8 in cands:
    n_types = int(obs_types[pid8])
    n_total = int(obs_total[pid8])
    if n_types >= 5:
        add_q("AGG_V2_015", "aggregate", pid8, "All waves",
              f"How many distinct types of clinical measurements were ever recorded for patient {pid8}?",
              f"{n_types} distinct observation types",
              temporal=False)
        gen = True; break
if not gen: skip("AGG_V2_015", "No qualifying patient")

# ── A16: Condition wave coverage ──────────────────────────────────────────────
cands = _pids(wave_count_per_pid[wave_count_per_pid >= 4].index)
gen = False
for pid8 in cands:
    all_w  = _patient_all_waves(pid8)
    cond_w = set(conditions[conditions["pid8"] == pid8]["WAVE"].dropna().astype(int).unique())
    if len(all_w) >= 4 and len(cond_w) >= 2:
        pct = round(len(cond_w) / len(all_w) * 100)
        add_q("AGG_V2_016", "aggregate", pid8, "All waves",
              f"What fraction of waves patient {pid8} appears in had at least one condition diagnosed?",
              f"{len(cond_w)} of {len(all_w)} waves had condition diagnoses ({pct}%)")
        gen = True; break
if not gen: skip("AGG_V2_016", "No qualifying patient")

# ── A17: Peak comorbidity count ────────────────────────────────────────────────
cands = _pids(cond_wave_ct3[cond_wave_ct3 >= 3].index)
gen = False
for pid8 in cands:
    all_w = _patient_all_waves(pid8)
    best_cnt = 0; best_wave = None
    for w in all_w:
        cnt, _ = _active_conds_in_wave(pid8, w)
        if cnt > best_cnt:
            best_cnt = cnt; best_wave = w
    if best_cnt >= 3:
        add_q("AGG_V2_017", "aggregate", pid8, f"Wave {best_wave}",
              f"What is the maximum number of conditions patient {pid8} had active simultaneously in any single wave?",
              f"{best_cnt} conditions in Wave {best_wave}")
        gen = True; break
if not gen: skip("AGG_V2_017", "No qualifying patient")

# ── A18: Medication-free waves ────────────────────────────────────────────────
cands = _pids(wave_count_per_pid[wave_count_per_pid >= 4].index)
gen = False
for pid8 in cands:
    all_w = _patient_all_waves(pid8)
    m_w   = set(medications[medications["pid8"] == pid8]["WAVE"].dropna().astype(int).unique())
    free  = sorted(all_w - m_w)
    if len(free) >= 1 and len(all_w) >= 4:
        add_q("AGG_V2_018", "aggregate", pid8, "All waves",
              f"How many waves did patient {pid8} appear in with NO medications at all?",
              f"{len(free)} waves with no medications (Wave {', Wave '.join(map(str, free))})")
        gen = True; break
if not gen: skip("AGG_V2_018", "No qualifying patient")

# ── A19: Total health burden score ────────────────────────────────────────────
cands = _pids(wave_count_per_pid[wave_count_per_pid >= 4].index)
gen = False
for pid8 in cands:
    nc  = int(conditions[conditions["pid8"] == pid8]["DESCRIPTION"].nunique())
    nm  = int(medications[medications["pid8"] == pid8]["DESCRIPTION"].nunique())
    not_ = int(observations[observations["pid8"] == pid8]["DESCRIPTION"].nunique())
    tot = nc + nm + not_
    if tot >= 15:
        add_q("AGG_V2_019", "aggregate", pid8, "All waves",
              f"What was the total number of distinct conditions, medications, and observation types "
              f"ever recorded for patient {pid8}?",
              f"{tot} total ({nc} conditions, {nm} medications, {not_} obs types)",
              temporal=False)
        gen = True; break
if not gen: skip("AGG_V2_019", "No qualifying patient")

# ── A20: Heart rate range across waves ────────────────────────────────────────
hr_by_wave = hr_obs.groupby(["pid8","WAVE"])["VALUE"].mean().reset_index()
hr_wave_ct = hr_by_wave.groupby("pid8")["WAVE"].count()
cands = _pids(hr_wave_ct[hr_wave_ct >= 3].index)
gen = False
for pid8 in cands:
    sub  = hr_by_wave[hr_by_wave["pid8"] == pid8].sort_values("VALUE")
    mn_w = int(sub.iloc[0]["WAVE"]);  mn_v = round(float(sub.iloc[0]["VALUE"]), 1)
    mx_w = int(sub.iloc[-1]["WAVE"]); mx_v = round(float(sub.iloc[-1]["VALUE"]), 1)
    if mx_v - mn_v >= 5:
        add_q("AGG_V2_020", "aggregate", pid8, "All waves",
              f"What was the range of average heart rates recorded for patient {pid8} "
              f"across waves (minimum to maximum wave average)?",
              f"{mn_v} /min (Wave {mn_w}) to {mx_v} /min (Wave {mx_w})")
        gen = True; break
if not gen: skip("AGG_V2_020", "No qualifying patient")

# ── A21: BMI trend direction overall ─────────────────────────────────────────
cands = _pids(bmi_wave_ct[bmi_wave_ct >= 3].index)
gen = False
for pid8 in cands:
    sub = bmi_by_wave_all[bmi_by_wave_all["pid8"] == pid8].sort_values("WAVE")
    if len(sub) < 3: continue
    fw  = int(sub.iloc[0]["WAVE"]);  fv = round(float(sub.iloc[0]["VALUE"]), 1)
    lw  = int(sub.iloc[-1]["WAVE"]); lv = round(float(sub.iloc[-1]["VALUE"]), 1)
    if   lv > fv + 0.5:  direction = "increased"
    elif lv < fv - 0.5:  direction = "decreased"
    else:                  direction = "remained stable"
    add_q("AGG_V2_021", "aggregate", pid8, f"Wave {fw} to Wave {lw}",
          f"Did patient {pid8}'s BMI overall increase, decrease, or remain stable across all waves?",
          f"Overall {direction} from {fv} kg/m2 (Wave {fw}) to {lv} kg/m2 (Wave {lw})")
    gen = True; break
if not gen: skip("AGG_V2_021", "No qualifying patient")

# ── A22: Condition that persisted longest (most active waves) ─────────────────
cands = _pids(cond_wave_ct3[cond_wave_ct3 >= 3].index)
gen = False
for pid8 in cands:
    sub   = conditions[conditions["pid8"] == pid8]
    descs = sub["DESCRIPTION"].unique()
    best_cond = None; best_cnt = 0
    for desc in descs:
        crecs = sub[sub["DESCRIPTION"] == desc]
        cnt = 0
        for w in range(1, 8):
            ws, we = WAVE_BOUNDS[w]
            if not crecs[(crecs["START"] <= we) &
                         (crecs["STOP"].isna() | (crecs["STOP"] >= ws))].empty:
                cnt += 1
        if cnt > best_cnt:
            best_cnt = cnt; best_cond = desc
    if best_cond and best_cnt >= 3:
        add_q("AGG_V2_022", "aggregate", pid8, "All waves",
              f"Which condition was active for patient {pid8} across the most number of waves?",
              f"{best_cond} — active across {best_cnt} waves")
        gen = True; break
if not gen: skip("AGG_V2_022", "No qualifying patient")

# ── A23: Waves with at least one NEW diagnosis not seen before ─────────────────
cands = _pids(wave_count_per_pid[wave_count_per_pid >= 4].index)
gen = False
for pid8 in cands:
    sub = conditions[conditions["pid8"] == pid8].dropna(subset=["WAVE"]).copy()
    if sub.empty: continue
    sub["WAVE"] = sub["WAVE"].astype(int)
    seen = set(); new_waves = 0; total_waves = len(_patient_all_waves(pid8))
    for w in sorted(sub["WAVE"].unique()):
        wave_descs = set(sub[sub["WAVE"] == w]["DESCRIPTION"].unique())
        if wave_descs - seen:
            new_waves += 1
        seen |= wave_descs
    if total_waves >= 4 and new_waves >= 2:
        add_q("AGG_V2_023", "aggregate", pid8, "All waves",
              f"In how many waves did patient {pid8} receive at least one entirely new diagnosis "
              f"not seen in prior waves?",
              f"{new_waves} of {total_waves} waves had new diagnoses")
        gen = True; break
if not gen: skip("AGG_V2_023", "No qualifying patient")

# ── A24: Wave with most distinct observation types ────────────────────────────
obs_type_by_wave = (observations.groupby(["pid8","WAVE"])["DESCRIPTION"].nunique()
                    .reset_index(name="n_types"))
obs_w_ct = obs_type_by_wave.groupby("pid8")["WAVE"].count()
cands = _pids(obs_w_ct[obs_w_ct >= 3].index)
gen = False
for pid8 in cands:
    sub  = obs_type_by_wave[obs_type_by_wave["pid8"] == pid8]
    best = sub.loc[sub["n_types"].idxmax()]
    w, nt = int(best["WAVE"]), int(best["n_types"])
    if nt >= 5:
        add_q("AGG_V2_024", "aggregate", pid8, f"Wave {w}",
              f"Which wave had the most complete clinical picture for patient {pid8} "
              f"(highest variety of observation types)?",
              f"Wave {w} ({nt} distinct types)")
        gen = True; break
if not gen: skip("AGG_V2_024", "No qualifying patient")

# ── A25: Total medication prescription records (including repeats) ─────────────
med_total_ct = medications.groupby("pid8").size()
med_w_ct2    = medications.groupby("pid8")["WAVE"].nunique()
cands = _pids(med_w_ct2[med_w_ct2 >= 3].index)
gen = False
for pid8 in cands:
    n = int(med_total_ct[pid8])
    if n >= 5:
        add_q("AGG_V2_025", "aggregate", pid8, "All waves",
              f"What is the total count of all medication prescriptions ever written for patient {pid8} "
              f"(including repeats)?",
              f"{n} total prescription records",
              temporal=False)
        gen = True; break
if not gen: skip("AGG_V2_025", "No qualifying patient")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION B — MULTI-HOP (MH_V2_001 to MH_V2_025)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("SECTION B — MULTI-HOP (25 questions)")
print("=" * 65)

# ── B1: BMI at first diagnosis ────────────────────────────────────────────────
cands = _pids(bmi_wave_ct.index)
gen = False
for pid8 in cands:
    cond_sub = conditions[conditions["pid8"] == pid8].dropna(subset=["START"]).sort_values("START")
    if cond_sub.empty: continue
    first     = cond_sub.iloc[0]
    first_w   = first["WAVE"]
    if pd.isna(first_w): continue
    first_w   = int(first_w)
    cname     = first["DESCRIPTION"]
    bmi_in_w  = bmi_obs[(bmi_obs["pid8"] == pid8) & (bmi_obs["WAVE"] == first_w)]
    if bmi_in_w.empty: continue
    bmi_val   = round(float(bmi_in_w["VALUE"].mean()), 1)
    add_q("MH_V2_001", "multi-hop", pid8, f"Wave {first_w}",
          f"What was patient {pid8}'s BMI in the wave they were first ever diagnosed with any condition?",
          f"BMI was {bmi_val} kg/m2 in Wave {first_w} (when first diagnosis made: {cname})")
    gen = True; break
if not gen: skip("MH_V2_001", "No qualifying patient")

# ── B2: First medication before or after first condition ───────────────────────
cands = _pids(wave_count_per_pid[wave_count_per_pid >= 2].index)
gen = False
for pid8 in cands:
    cond_sub = conditions[conditions["pid8"] == pid8].dropna(subset=["START"])
    meds_sub = medications[medications["pid8"] == pid8].dropna(subset=["START"])
    if cond_sub.empty or meds_sub.empty: continue
    fc_date = cond_sub["START"].min()
    fm_date = meds_sub["START"].min()
    fc_wave = int(_wave(fc_date))
    fm_wave = int(_wave(fm_date))
    if pd.isna(fc_date) or pd.isna(fm_date): continue
    timing  = "before" if fm_date < fc_date else "after"
    fc_str  = fc_date.strftime("%Y-%m-%d"); fm_str = fm_date.strftime("%Y-%m-%d")
    add_q("MH_V2_002", "multi-hop", pid8, f"Wave {fm_wave} vs Wave {fc_wave}",
          f"Was patient {pid8}'s first medication prescribed before or after their first condition diagnosis?",
          f"Medication came {timing} — first med Wave {fm_wave} ({fm_str}), first condition Wave {fc_wave} ({fc_str})")
    gen = True; break
if not gen: skip("MH_V2_002", "No qualifying patient")

# ── B3: Observation count at first condition's wave ───────────────────────────
cands = _pids(obs_total[obs_total >= 10].index)
gen = False
for pid8 in cands:
    cond_sub = conditions[conditions["pid8"] == pid8].dropna(subset=["START"]).sort_values("START")
    if cond_sub.empty: continue
    first   = cond_sub.iloc[0]
    cname   = first["DESCRIPTION"]
    c_wave  = first["WAVE"]
    if pd.isna(c_wave): continue
    c_wave  = int(c_wave)
    n_obs   = int(len(observations[(observations["pid8"] == pid8) & (observations["WAVE"] == c_wave)]))
    if n_obs >= 3:
        add_q("MH_V2_003", "multi-hop", pid8, f"Wave {c_wave}",
              f"How many observations were recorded for patient {pid8} in the same wave "
              f"they were first diagnosed with any condition?",
              f"{n_obs} observations in Wave {c_wave} (same wave as first diagnosis: {cname})")
        gen = True; break
if not gen: skip("MH_V2_003", "No qualifying patient")

# ── B4: Body weight at first medication wave ───────────────────────────────────
cands = _pids(med_total_ct[med_total_ct >= 1].index)
gen = False
for pid8 in cands:
    meds_sub = medications[medications["pid8"] == pid8].dropna(subset=["START"]).sort_values("START")
    if meds_sub.empty: continue
    first_med_wave = meds_sub.iloc[0]["WAVE"]
    if pd.isna(first_med_wave): continue
    first_med_wave = int(first_med_wave)
    w_sub = weight_obs[(weight_obs["pid8"] == pid8) & (weight_obs["WAVE"] == first_med_wave)]
    if w_sub.empty: continue
    w_val = round(float(w_sub["VALUE"].mean()), 1)
    add_q("MH_V2_004", "multi-hop", pid8, f"Wave {first_med_wave}",
          f"What was patient {pid8}'s average body weight in the wave they were first prescribed any medication?",
          f"{w_val} kg in Wave {first_med_wave} (wave of first prescription)")
    gen = True; break
if not gen: skip("MH_V2_004", "No qualifying patient")

# ── B5: SBP in wave immediately after hypertension diagnosis ──────────────────
hyp_pids = conditions[conditions["DESCRIPTION"] == "Hypertension"]["pid8"].unique()
sbp_pids = sbp_obs["pid8"].unique()
cands = _pids([p for p in hyp_pids if p in sbp_pids])
gen = False
for pid8 in cands:
    hyp = conditions[
        (conditions["pid8"] == pid8) &
        (conditions["DESCRIPTION"] == "Hypertension")
    ].sort_values("START")
    hyp_wave = int(hyp.iloc[0]["WAVE"])
    sbp_sub  = sbp_obs[sbp_obs["pid8"] == pid8].groupby("WAVE")["VALUE"].mean()
    post_waves = sorted([int(w) for w in sbp_sub.index if int(w) > hyp_wave])
    if not post_waves: continue
    sbp_at_diag = round(float(sbp_sub.get(hyp_wave, sbp_sub.iloc[0])), 1)
    next_wave   = post_waves[0]
    sbp_next    = round(float(sbp_sub[next_wave]), 1)
    if pd.isna(sbp_at_diag): continue
    rose = sbp_next > sbp_at_diag
    add_q("MH_V2_005", "multi-hop", pid8, f"Wave {hyp_wave} to Wave {next_wave}",
          f"Did patient {pid8}'s average Systolic Blood Pressure increase in the wave "
          f"immediately after their first Hypertension diagnosis?",
          f"{'Yes' if rose else 'No'} — SBP was {sbp_at_diag} mm[Hg] in diagnosis wave (Wave {hyp_wave}), "
          f"{sbp_next} mm[Hg] in Wave {next_wave}")
    gen = True; break
if not gen: skip("MH_V2_005", "No qualifying patient")

# ── B6: New conditions after first medication ──────────────────────────────────
cands = _pids(wave_count_per_pid[wave_count_per_pid >= 2].index)
gen = False
for pid8 in cands:
    meds_sub = medications[medications["pid8"] == pid8].dropna(subset=["START"])
    if meds_sub.empty: continue
    cond_sub = conditions[conditions["pid8"] == pid8].dropna(subset=["START"])
    if cond_sub.empty: continue
    first_med_date = meds_sub["START"].min()
    first_med_wave = int(_wave(first_med_date))
    before_descs   = set(cond_sub[cond_sub["START"] < first_med_date]["DESCRIPTION"].unique())
    after_descs    = set(cond_sub[cond_sub["START"] >= first_med_date]["DESCRIPTION"].unique())
    new_conds      = after_descs - before_descs
    if len(new_conds) >= 1:
        add_q("MH_V2_006", "multi-hop", pid8, f"Wave {first_med_wave} onwards",
              f"How many new conditions were diagnosed for patient {pid8} "
              f"after they started their first medication?",
              f"{len(new_conds)} new conditions after Wave {first_med_wave} (when first medication started)")
        gen = True; break
if not gen: skip("MH_V2_006", "No qualifying patient")

# ── B7: BMI obese (>=30) in hypertension diagnosis wave ───────────────────────
cands = _pids([p for p in hyp_pids if p in bmi_obs["pid8"].unique()])
gen = False
for pid8 in cands:
    hyp = conditions[
        (conditions["pid8"] == pid8) &
        (conditions["DESCRIPTION"] == "Hypertension")
    ].sort_values("START")
    hyp_wave = int(hyp.iloc[0]["WAVE"])
    bmi_in_w = bmi_obs[(bmi_obs["pid8"] == pid8) & (bmi_obs["WAVE"] == hyp_wave)]
    if bmi_in_w.empty: continue
    bmi_val = round(float(bmi_in_w["VALUE"].mean()), 1)
    is_ob   = bmi_val >= 30
    add_q("MH_V2_007", "multi-hop", pid8, f"Wave {hyp_wave}",
          f"Was patient {pid8}'s BMI in the obese range (>=30 kg/m2) "
          f"in the same wave they were diagnosed with Hypertension?",
          f"{'Yes' if is_ob else 'No'} — BMI {bmi_val} kg/m2 in Wave {hyp_wave} (Hypertension diagnosis wave)")
    gen = True; break
if not gen: skip("MH_V2_007", "No qualifying patient")

# ── B8: Glucose level at diabetes diagnosis ────────────────────────────────────
glucose_obs = observations[observations["DESCRIPTION"] == "Glucose"].copy()
glucose_obs["VALUE"] = pd.to_numeric(glucose_obs["VALUE"], errors="coerce")
glucose_obs = glucose_obs.dropna(subset=["VALUE"])
diab_conds  = conditions[conditions["DESCRIPTION"].isin(["Diabetes","Prediabetes"])]
diab_pids   = diab_conds["pid8"].unique()
gluc_pids   = glucose_obs["pid8"].unique()
cands = _pids([p for p in diab_pids if p in gluc_pids])
gen = False
for pid8 in cands:
    dc  = diab_conds[diab_conds["pid8"] == pid8].sort_values("START")
    dw  = int(dc.iloc[0]["WAVE"])
    dn  = dc.iloc[0]["DESCRIPTION"]
    g_sub = glucose_obs[(glucose_obs["pid8"] == pid8) & (glucose_obs["WAVE"] == dw)]
    if g_sub.empty: continue
    g_val = round(float(g_sub["VALUE"].mean()), 1)
    units = str(g_sub["UNITS"].iloc[0]) if "UNITS" in g_sub.columns else "mg/dL"
    add_q("MH_V2_008", "multi-hop", pid8, f"Wave {dw}",
          f"What was patient {pid8}'s recorded glucose level in the wave "
          f"they were first diagnosed with {dn}?",
          f"{g_val} {units} in Wave {dw}")
    gen = True; break
if not gen: skip("MH_V2_008", "No qualifying patient")

# ── B9: Was Hypertension diagnosed before first medication? ────────────────────
cands = _pids([p for p in hyp_pids if p in medications["pid8"].unique()])
gen = False
for pid8 in cands:
    hyp = conditions[
        (conditions["pid8"] == pid8) &
        (conditions["DESCRIPTION"] == "Hypertension")
    ].sort_values("START")
    hyp_date = hyp.iloc[0]["START"]
    hyp_wave = int(hyp.iloc[0]["WAVE"])
    meds_sub = medications[medications["pid8"] == pid8].sort_values("START")
    fm_date  = meds_sub.iloc[0]["START"]
    fm_wave  = int(meds_sub.iloc[0]["WAVE"])
    fm_name  = meds_sub.iloc[0]["DESCRIPTION"][:45]
    before   = hyp_date < fm_date
    hd_str   = hyp_date.strftime("%Y-%m-%d")
    fm_str   = fm_date.strftime("%Y-%m-%d")
    add_q("MH_V2_009", "multi-hop", pid8, f"Wave {hyp_wave} vs Wave {fm_wave}",
          f"Was patient {pid8} already diagnosed with Hypertension before they were prescribed {fm_name}?",
          f"{'Yes' if before else 'No'} — Hypertension Wave {hyp_wave} ({hd_str}), "
          f"medication prescribed Wave {fm_wave} ({fm_str})")
    gen = True; break
if not gen: skip("MH_V2_009", "No qualifying patient")

# ── B10: HR change around first chronic condition ─────────────────────────────
cands = _pids(hr_wave_ct[hr_wave_ct >= 3].index)
gen = False
for pid8 in cands:
    chronic = conditions[
        (conditions["pid8"] == pid8) &
        (conditions["STOP"].isna())
    ].dropna(subset=["START"]).sort_values("START")
    if chronic.empty: continue
    cname  = chronic.iloc[0]["DESCRIPTION"]
    c_wave = chronic.iloc[0]["WAVE"]
    if pd.isna(c_wave): continue
    c_wave = int(c_wave)
    hr_p = hr_obs[hr_obs["pid8"] == pid8].groupby("WAVE")["VALUE"].mean()
    prior = [int(w) for w in hr_p.index if int(w) < c_wave]
    if not prior: continue
    p_wave = max(prior)
    hr_b   = round(float(hr_p[p_wave]), 1)
    if c_wave not in hr_p.index: continue
    hr_a   = round(float(hr_p[c_wave]), 1)
    add_q("MH_V2_010", "multi-hop", pid8, f"Wave {p_wave} to Wave {c_wave}",
          f"How did patient {pid8}'s average heart rate change between the wave before "
          f"and the wave of their first chronic condition diagnosis ({cname})?",
          f"HR was {hr_b} /min before (Wave {p_wave}), {hr_a} /min in diagnosis wave (Wave {c_wave})")
    gen = True; break
if not gen: skip("MH_V2_010", "No qualifying patient")

# ── B11: Medication count in peak BMI wave ────────────────────────────────────
cands = _pids(bmi_wave_ct[bmi_wave_ct >= 2].index)
gen = False
for pid8 in cands:
    sub = bmi_by_wave_all[bmi_by_wave_all["pid8"] == pid8]
    if sub.empty: continue
    pk_wave = int(sub.loc[sub["VALUE"].idxmax()]["WAVE"])
    pk_bmi  = round(float(sub["VALUE"].max()), 1)
    m_ct    = int(medications[(medications["pid8"] == pid8) & (medications["WAVE"] == pk_wave)]["DESCRIPTION"].nunique())
    add_q("MH_V2_011", "multi-hop", pid8, f"Wave {pk_wave}",
          f"How many medications was patient {pid8} taking in the wave when their BMI was highest?",
          f"{m_ct} medications in Wave {pk_wave} (peak BMI wave, avg BMI = {pk_bmi} kg/m2)")
    gen = True; break
if not gen: skip("MH_V2_011", "No qualifying patient")

# ── B12: Age at first chronic condition ───────────────────────────────────────
cands = _pids(cond_wave_ct3[cond_wave_ct3 >= 1].index)
gen = False
for pid8 in cands:
    chronic = conditions[
        (conditions["pid8"] == pid8) &
        (conditions["STOP"].isna())
    ].dropna(subset=["START"]).sort_values("START")
    if chronic.empty: continue
    pat_row = patients[patients["pid8"] == pid8]
    if pat_row.empty or pd.isna(pat_row.iloc[0]["BIRTHDATE"]): continue
    birth_year = int(pat_row.iloc[0]["BIRTHDATE"].year)
    diag_year  = int(chronic.iloc[0]["START"].year)
    cname      = chronic.iloc[0]["DESCRIPTION"]
    c_wave     = chronic.iloc[0]["WAVE"]
    if pd.isna(c_wave): continue
    c_wave     = int(c_wave)
    age        = diag_year - birth_year
    add_q("MH_V2_012", "multi-hop", pid8, f"Wave {c_wave}",
          f"How old was patient {pid8} when they were first diagnosed with a chronic condition (no stop date)?",
          f"Age {age} (born {birth_year}, diagnosed {diag_year} in Wave {c_wave}: {cname})",
          temporal=False)
    gen = True; break
if not gen: skip("MH_V2_012", "No qualifying patient")

# ── B13: Observation count change at first condition diagnosis ─────────────────
cands = _pids(obs_w_ct[obs_w_ct >= 2].index)
gen = False
for pid8 in cands:
    cond_sub = conditions[conditions["pid8"] == pid8].dropna(subset=["START"]).sort_values("START")
    if cond_sub.empty: continue
    cname  = cond_sub.iloc[0]["DESCRIPTION"]
    c_wave = cond_sub.iloc[0]["WAVE"]
    if pd.isna(c_wave): continue
    c_wave = int(c_wave)
    n_diag = int(len(observations[(observations["pid8"] == pid8) & (observations["WAVE"] == c_wave)]))
    prior_waves = sorted([w for w in _patient_all_waves(pid8) if w < c_wave])
    if not prior_waves: continue
    p_wave = prior_waves[-1]
    n_prev = int(len(observations[(observations["pid8"] == pid8) & (observations["WAVE"] == p_wave)]))
    if n_prev == 0: continue
    if n_diag > n_prev:   dir_ = "increased"
    elif n_diag < n_prev: dir_ = "decreased"
    else:                  dir_ = "remained stable"
    add_q("MH_V2_013", "multi-hop", pid8, f"Wave {p_wave} to Wave {c_wave}",
          f"How did the number of observations recorded for patient {pid8} change in the wave "
          f"they were diagnosed with {cname} compared to the prior wave?",
          f"Observations {dir_} from {n_prev} (Wave {p_wave}) to {n_diag} (Wave {c_wave})")
    gen = True; break
if not gen: skip("MH_V2_013", "No qualifying patient")

# ── B14: Two conditions first diagnosed in same wave ─────────────────────────
cands = _pids(cond_wave_ct3[cond_wave_ct3 >= 2].index)
gen = False
for pid8 in cands:
    cond_sub  = conditions[conditions["pid8"] == pid8].dropna(subset=["START","WAVE"])
    first_occ = cond_sub.sort_values("START").drop_duplicates("DESCRIPTION", keep="first")
    by_wave   = first_occ.groupby("WAVE").filter(lambda x: len(x) >= 2)
    if by_wave.empty: continue
    w = int(sorted(by_wave["WAVE"].unique())[0])
    rows_in_w = first_occ[first_occ["WAVE"] == w].sort_values("START")
    ca = rows_in_w.iloc[0]["DESCRIPTION"]; da = rows_in_w.iloc[0]["START"].strftime("%Y-%m-%d")
    cb = rows_in_w.iloc[1]["DESCRIPTION"]; db = rows_in_w.iloc[1]["START"].strftime("%Y-%m-%d")
    add_q("MH_V2_014", "multi-hop", pid8, f"Wave {w}",
          f"In which wave were both '{ca}' and '{cb}' first diagnosed for patient {pid8}?",
          f"Both first diagnosed in Wave {w} ({ca}: {da}; {cb}: {db})")
    gen = True; break
if not gen: skip("MH_V2_014", "No qualifying patient")

# ── B15: SBP before vs after first medication ─────────────────────────────────
sbp_pids_set = set(sbp_obs["pid8"].unique())
med_pids_set = set(medications["pid8"].unique())
cands = _pids(list(sbp_pids_set & med_pids_set))
gen = False
for pid8 in cands:
    meds_sub     = medications[medications["pid8"] == pid8].sort_values("START")
    if meds_sub.empty: continue
    fm_date = meds_sub.iloc[0]["START"]
    fm_name = meds_sub.iloc[0]["DESCRIPTION"][:45]
    sbp_sub = sbp_obs[sbp_obs["pid8"] == pid8]
    before  = sbp_sub[sbp_sub["DATE"] < fm_date]["VALUE"]
    after   = sbp_sub[sbp_sub["DATE"] >= fm_date]["VALUE"]
    if before.empty or after.empty: continue
    avg_b = round(float(before.mean()), 1)
    avg_a = round(float(after.mean()), 1)
    lower = avg_a < avg_b
    add_q("MH_V2_015", "multi-hop", pid8, "Pre/Post first medication",
          f"Was patient {pid8}'s average Systolic Blood Pressure lower after they started "
          f"taking {fm_name} compared to before?",
          f"{'Yes' if lower else 'No'} — avg SBP before: {avg_b} mm[Hg]; after: {avg_a} mm[Hg]")
    gen = True; break
if not gen: skip("MH_V2_015", "No qualifying patient")

# ── B16: Weight change at first condition diagnosis ────────────────────────────
weight_pids = set(weight_obs["pid8"].unique())
cands = _pids(list(weight_pids & set(conditions["pid8"].unique())))
gen = False
for pid8 in cands:
    cond_sub = conditions[conditions["pid8"] == pid8].dropna(subset=["START"]).sort_values("START")
    if cond_sub.empty: continue
    cname  = cond_sub.iloc[0]["DESCRIPTION"]
    c_wave = cond_sub.iloc[0]["WAVE"]
    if pd.isna(c_wave): continue
    c_wave = int(c_wave)
    w_diag = weight_obs[(weight_obs["pid8"] == pid8) & (weight_obs["WAVE"] == c_wave)]["VALUE"]
    prior_waves = sorted([w for w in _patient_all_waves(pid8) if w < c_wave])
    if not prior_waves or w_diag.empty: continue
    p_wave  = prior_waves[-1]
    w_prior = weight_obs[(weight_obs["pid8"] == pid8) & (weight_obs["WAVE"] == p_wave)]["VALUE"]
    if w_prior.empty: continue
    wb = round(float(w_prior.mean()), 1)
    wa = round(float(w_diag.mean()), 1)
    gained = wa > wb
    add_q("MH_V2_016", "multi-hop", pid8, f"Wave {p_wave} to Wave {c_wave}",
          f"Did patient {pid8}'s body weight increase in the same wave they were diagnosed with {cname}?",
          f"{'Yes' if gained else 'No'} — weight {wb} kg (Wave {p_wave}), {wa} kg (Wave {c_wave})")
    gen = True; break
if not gen: skip("MH_V2_016", "No qualifying patient")

# ── B17: BMI above 25 at first medication wave ────────────────────────────────
cands = _pids(list(set(bmi_obs["pid8"].unique()) & set(medications["pid8"].unique())))
gen = False
for pid8 in cands:
    meds_sub = medications[medications["pid8"] == pid8].sort_values("START")
    if meds_sub.empty: continue
    fm_wave  = meds_sub.iloc[0]["WAVE"]
    if pd.isna(fm_wave): continue
    fm_wave  = int(fm_wave)
    bmi_in_w = bmi_obs[(bmi_obs["pid8"] == pid8) & (bmi_obs["WAVE"] == fm_wave)]
    if bmi_in_w.empty: continue
    bmi_val  = round(float(bmi_in_w["VALUE"].mean()), 1)
    above25  = bmi_val >= 25
    add_q("MH_V2_017", "multi-hop", pid8, f"Wave {fm_wave}",
          f"Was patient {pid8}'s BMI above 25 kg/m2 in the wave they were first prescribed any medication?",
          f"{'Yes' if above25 else 'No'} — BMI {bmi_val} kg/m2 in Wave {fm_wave} (first medication wave)")
    gen = True; break
if not gen: skip("MH_V2_017", "No qualifying patient")

# ── B18: Conditions per wave grouping (early/mid/late) ───────────────────────
cands = _pids(wave_count_per_pid[wave_count_per_pid >= 4].index)
gen = False
for pid8 in cands:
    sub = conditions[conditions["pid8"] == pid8].dropna(subset=["WAVE"])
    if sub.empty: continue
    sub["WAVE"] = sub["WAVE"].astype(int)
    c_per_w = sub.groupby("WAVE")["DESCRIPTION"].nunique()
    early = int(c_per_w.get(1, 0) + c_per_w.get(2, 0))
    mid   = int(c_per_w.get(3, 0) + c_per_w.get(4, 0))
    late  = int(c_per_w.get(5, 0) + c_per_w.get(6, 0))
    if (early + mid + late) >= 5 and len({early, mid, late}) >= 2:
        add_q("MH_V2_018", "multi-hop", pid8, "Wave 1-2 vs 3-4 vs 5-6",
              f"How did patient {pid8}'s condition burden change across decades "
              f"(Waves 1-2 vs Waves 3-4 vs Waves 5-6)?",
              f"Early waves (1-2): {early}, Mid waves (3-4): {mid}, Late waves (5-6): {late} conditions")
        gen = True; break
if not gen: skip("MH_V2_018", "No qualifying patient")

# ── B19: Were observations recorded before first condition? ───────────────────
cands = _pids(obs_w_ct[obs_w_ct >= 2].index)
gen = False
for pid8 in cands:
    cond_sub = conditions[conditions["pid8"] == pid8].dropna(subset=["START"])
    obs_sub  = observations[observations["pid8"] == pid8].dropna(subset=["DATE"])
    if cond_sub.empty or obs_sub.empty: continue
    fc_date = cond_sub["START"].min()
    fo_date = obs_sub["DATE"].min()
    if pd.isna(fc_date) or pd.isna(fo_date): continue
    obs_first  = fo_date < fc_date
    fo_wave    = int(_wave(fo_date))
    fc_wave    = int(_wave(fc_date))
    fo_str     = fo_date.strftime("%Y-%m-%d")
    fc_str     = fc_date.strftime("%Y-%m-%d")
    add_q("MH_V2_019", "multi-hop", pid8, "All waves",
          f"Did patient {pid8} have any clinical observations recorded before their first condition was diagnosed?",
          f"{'Yes' if obs_first else 'No'} — first obs Wave {fo_wave} ({fo_str}), "
          f"first diagnosis Wave {fc_wave} ({fc_str})")
    gen = True; break
if not gen: skip("MH_V2_019", "No qualifying patient")

# ── B20: Medication stopped while condition still active ───────────────────────
cands = _pids(wave_count_per_pid[wave_count_per_pid >= 2].index)
gen = False
for pid8 in cands:
    stopped_meds = medications[
        (medications["pid8"] == pid8) &
        (medications["STOP"].notna())
    ].sort_values("START")
    if stopped_meds.empty: continue
    cond_sub = conditions[conditions["pid8"] == pid8]
    found = False
    for _, med_row in stopped_meds.iterrows():
        med_stop = med_row["STOP"]
        med_name = med_row["DESCRIPTION"][:50]
        active = cond_sub[
            (cond_sub["START"] <= med_stop) &
            (cond_sub["STOP"].isna() | (cond_sub["STOP"] > med_stop))
        ]
        if not active.empty:
            cname    = active.iloc[0]["DESCRIPTION"]
            c_start  = active.iloc[0]["START"].strftime("%Y-%m-%d")
            ms_str   = med_stop.strftime("%Y-%m-%d")
            m_wave   = int(_wave(med_stop))
            add_q("MH_V2_020", "multi-hop", pid8, f"Wave {m_wave}",
                  f"Did patient {pid8} have any medications stopped while a condition was still active?",
                  f"Yes — '{med_name}' stopped {ms_str} while '{cname}' (diagnosed {c_start}) was still active")
            found = True; break
    if found: gen = True; break
if not gen: skip("MH_V2_020", "No qualifying patient")

# ── B21: Was HR above personal average in peak BMI wave? ─────────────────────
cands = _pids(list(set(bmi_obs["pid8"].unique()) & set(hr_obs["pid8"].unique())))
gen = False
for pid8 in cands:
    sub = bmi_by_wave_all[bmi_by_wave_all["pid8"] == pid8]
    if len(sub) < 3: continue
    pk_wave = int(sub.loc[sub["VALUE"].idxmax()]["WAVE"])
    pk_bmi  = round(float(sub["VALUE"].max()), 1)
    hr_sub  = hr_obs[hr_obs["pid8"] == pid8]
    hr_peak = hr_sub[hr_sub["WAVE"] == pk_wave]["VALUE"]
    if hr_peak.empty: continue
    hr_avg_all  = round(float(hr_sub["VALUE"].mean()), 1)
    hr_avg_peak = round(float(hr_peak.mean()), 1)
    above = hr_avg_peak > hr_avg_all
    add_q("MH_V2_021", "multi-hop", pid8, f"Wave {pk_wave}",
          f"In the wave patient {pid8} had their highest BMI, was their heart rate also above their personal average?",
          f"{'Yes' if above else 'No'} — peak BMI {pk_bmi} kg/m2 (Wave {pk_wave}), "
          f"HR {hr_avg_peak} /min (personal avg: {hr_avg_all} /min)")
    gen = True; break
if not gen: skip("MH_V2_021", "No qualifying patient")

# ── B22: New medication in every wave with new condition? ─────────────────────
cands = _pids(wave_count_per_pid[wave_count_per_pid >= 3].index)
gen = False
for pid8 in cands:
    cond_sub = conditions[conditions["pid8"] == pid8].dropna(subset=["WAVE"])
    meds_sub = medications[medications["pid8"] == pid8].dropna(subset=["WAVE"])
    if cond_sub.empty or meds_sub.empty: continue
    cond_sub["WAVE"] = cond_sub["WAVE"].astype(int)
    meds_sub["WAVE"] = meds_sub["WAVE"].astype(int)
    first_c  = cond_sub.sort_values("START").drop_duplicates("DESCRIPTION", keep="first")
    first_m  = meds_sub.sort_values("START").drop_duplicates("DESCRIPTION", keep="first")
    new_c_waves = set(first_c["WAVE"].unique())
    new_m_waves = set(first_m["WAVE"].unique())
    if len(new_c_waves) < 2: continue
    no_match = sorted(new_c_waves - new_m_waves)
    n_total  = len(new_c_waves)
    if no_match:
        answer = f"No — Wave {no_match[0]} had new conditions but no new medications"
    else:
        answer = f"Yes — new medications matched in all {n_total} waves with new conditions"
    add_q("MH_V2_022", "multi-hop", pid8, "All waves",
          f"Did patient {pid8} receive a new medication in every wave they received a new condition diagnosis?",
          answer)
    gen = True; break
if not gen: skip("MH_V2_022", "No qualifying patient")

# ── B23: Longest gap (in waves) between condition diagnoses ───────────────────
cands = _pids(cond_wave_ct3[cond_wave_ct3 >= 3].index)
gen = False
for pid8 in cands:
    cond_waves = sorted(
        conditions[conditions["pid8"] == pid8]["WAVE"]
        .dropna().astype(int).unique()
    )
    if len(cond_waves) < 3: continue
    gaps = [(cond_waves[i+1] - cond_waves[i], cond_waves[i], cond_waves[i+1])
            for i in range(len(cond_waves)-1)]
    mx_gap, w_from, w_to = max(gaps, key=lambda x: x[0])
    if mx_gap >= 2:
        add_q("MH_V2_023", "multi-hop", pid8, f"Wave {w_from} to Wave {w_to}",
              f"What was the longest gap (in waves) between condition diagnoses for patient {pid8}?",
              f"{mx_gap} waves between Wave {w_from} and Wave {w_to}")
        gen = True; break
if not gen: skip("MH_V2_023", "No qualifying patient")

# ── B24: Was most recent wave also most clinically intense (obs count)? ────────
cands = _pids(obs_w_ct[obs_w_ct >= 2].index)
gen = False
for pid8 in cands:
    all_w = _patient_all_waves(pid8)
    if len(all_w) < 2: continue
    most_recent = max(all_w)
    obs_by_w    = observations[observations["pid8"] == pid8].groupby("WAVE").size()
    if obs_by_w.empty: continue
    peak_w    = int(obs_by_w.idxmax())
    peak_ct   = int(obs_by_w.max())
    recent_ct = int(obs_by_w.get(most_recent, 0))
    most_intense = (most_recent == peak_w)
    if most_intense:
        answer = f"Yes — Wave {most_recent} (most recent) had {peak_ct} observations (highest)"
    else:
        answer = (f"No — Wave {peak_w} had more ({peak_ct} observations) "
                  f"vs Wave {most_recent} ({recent_ct} observations)")
    add_q("MH_V2_024", "multi-hop", pid8, f"Wave {most_recent}",
          f"Was patient {pid8}'s most recent wave also their most clinically intense "
          f"(highest observation count)?",
          answer)
    gen = True; break
if not gen: skip("MH_V2_024", "No qualifying patient")

# ── B25: Cardiovascular vs metabolic — which came first? ──────────────────────
CARDIO   = ["hypertension", "heart", "coronary", "cardiac", "atrial", "stroke",
            "angina", "arrhythmia", "hyperlipidemia"]
METABOLIC = ["diabetes", "obesity", "prediabetes", "metabolic", "insulin", "overweight",
             "body mass index 30"]
cands = _pids(cond_wave_ct3[cond_wave_ct3 >= 2].index)
gen = False
for pid8 in cands:
    sub = conditions[conditions["pid8"] == pid8].dropna(subset=["START"]).sort_values("START")
    if sub.empty: continue
    desc_lower = sub["DESCRIPTION"].str.lower()
    cardio_mask   = desc_lower.apply(lambda d: any(k in d for k in CARDIO))
    metabolic_mask = desc_lower.apply(lambda d: any(k in d for k in METABOLIC))
    c_rows = sub[cardio_mask]
    m_rows = sub[metabolic_mask]
    if c_rows.empty or m_rows.empty: continue
    fc = c_rows.iloc[0]; fm = m_rows.iloc[0]
    if fc["START"] <= fm["START"]:
        first_type = "cardiovascular"; first_row = fc
    else:
        first_type = "metabolic"; first_row = fm
    cname   = first_row["DESCRIPTION"]
    c_wave  = int(first_row["WAVE"]) if pd.notna(first_row["WAVE"]) else "?"
    dstr    = first_row["START"].strftime("%Y-%m-%d")
    add_q("MH_V2_025", "multi-hop", pid8, "All waves",
          f"Which came first for patient {pid8} — their cardiovascular condition or their metabolic condition?",
          f"{cname} ({first_type}) diagnosed first (Wave {c_wave}, {dstr})")
    gen = True; break
if not gen: skip("MH_V2_025", "No qualifying patient")

# ══════════════════════════════════════════════════════════════════════════════
# Save to CSV
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
out_path = BENCH / "qa_pairs_v2.csv"
df = pd.DataFrame(records)
df.to_csv(out_path, index=False)

print(f"\nGenerated {len(records)}/50 questions -> {out_path}")
if skipped:
    print(f"Skipped ({len(skipped)}): {[s[0] for s in skipped]}")
else:
    print("All 50 questions generated successfully.")

# Print full table
print("\nFull Q&A summary:")
print(f"{'ID':<15} {'Category':<11} {'Patient':<10} {'Question (first 60)'}")
print("-" * 100)
for r in records:
    print(f"{r['question_id']:<15} {r['category']:<11} {r['patient_id']:<10} {r['question'][:60]}")
