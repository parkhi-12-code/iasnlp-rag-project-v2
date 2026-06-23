"""
src/generate_extended_benchmark.py
Generate / extend benchmark/qa_pairs_extended.csv.
Section A  — Multi-wave trajectory       (EXT_TRAJ_001–005)
Section B  — Co-occurrence / comorbidity  (EXT_COMORBID_001–005)
Section C  — Medication involvement       (EXT_MED_001–005)
Section D  — Intra-wave depth             (EXT_INTRAWAVE_001–005)
Section E  — Cross-table patterns         (EXT_PATTERN_001–005)
All answers computed deterministically from raw CSVs.
DO NOT touch benchmark/qa_pairs_final.csv.
"""
import sys
import os
import warnings
from collections import Counter

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore")

ROOT     = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(ROOT, "data", "raw")
OUT_PATH = os.path.join(ROOT, "benchmark", "qa_pairs_extended.csv")

COLS = ["question_id", "category", "patient_id", "wave_reference",
        "question", "answer", "verified", "notes", "temporal"]

# ── wave constants & helpers ───────────────────────────────────────────────────

WAVE_START = {1: 1990, 2: 1995, 3: 2000, 4: 2005, 5: 2010, 6: 2015, 7: 2020}
WAVE_END   = {1: 1994, 2: 1999, 3: 2004, 4: 2009, 5: 2014, 6: 2019, 7: 2030}


def wave_num(year):
    for w in range(1, 8):
        if WAVE_START[w] <= int(year) <= WAVE_END[w]:
            return w
    return None


def _years_to_waves(year_series):
    w = pd.Series(pd.NA, index=year_series.index, dtype="Int8")
    w[(year_series >= 1990) & (year_series <= 1994)] = 1
    w[(year_series >= 1995) & (year_series <= 1999)] = 2
    w[(year_series >= 2000) & (year_series <= 2004)] = 3
    w[(year_series >= 2005) & (year_series <= 2009)] = 4
    w[(year_series >= 2010) & (year_series <= 2014)] = 5
    w[(year_series >= 2015) & (year_series <= 2019)] = 6
    w[year_series >= 2020]                           = 7
    return w


def _wave_col(date_series):
    years = pd.to_numeric(date_series.str[:4], errors="coerce")
    return _years_to_waves(years)


def active_expand(df, start_col="start_year", stop_col="stop_year"):
    """Expand df: one row per (original_row, active_wave) where the record is active."""
    parts = []
    for w in range(1, 8):
        ws, we      = WAVE_START[w], WAVE_END[w]
        started     = df[start_col] <= we
        not_stopped = df[stop_col].isna() | (df[stop_col] >= ws)
        chunk       = df[started & not_stopped].copy()
        chunk["active_wave"] = w
        parts.append(chunk)
    return (pd.concat(parts, ignore_index=True)
            if parts else pd.DataFrame(columns=list(df.columns) + ["active_wave"]))


# ── data loading ───────────────────────────────────────────────────────────────

def load_data():
    print("Loading CSVs...")
    patients    = pd.read_csv(os.path.join(DATA_DIR, "patients.csv"),     dtype=str)
    conditions  = pd.read_csv(os.path.join(DATA_DIR, "conditions.csv"),   dtype=str)
    obs         = pd.read_csv(os.path.join(DATA_DIR, "observations.csv"), dtype=str)
    medications = pd.read_csv(os.path.join(DATA_DIR, "medications.csv"),  dtype=str)

    conditions["wave"]  = _wave_col(conditions["START"])
    obs["wave"]         = _wave_col(obs["DATE"])
    medications["wave"] = _wave_col(medications["START"])

    obs["value_num"] = pd.to_numeric(obs["VALUE"], errors="coerce")

    conditions["start_year"]  = pd.to_numeric(conditions["START"].str[:4], errors="coerce")
    conditions["stop_year"]   = pd.to_numeric(conditions["STOP"].str[:4],  errors="coerce")
    medications["start_year"] = pd.to_numeric(medications["START"].str[:4], errors="coerce")
    medications["stop_year"]  = pd.to_numeric(medications["STOP"].str[:4],  errors="coerce")

    for df in (patients, conditions, obs, medications):
        id_col = "Id" if "Id" in df.columns else "PATIENT"
        df["pid8"] = df[id_col].str[:8]

    print(f"  patients={len(patients)}, conditions={len(conditions)}, "
          f"obs={len(obs)}, medications={len(medications)}")
    return patients, conditions, obs, medications


# ── row builder ────────────────────────────────────────────────────────────────

def mkrow(qid, category, patient_id, wave_ref, question, answer):
    return {"question_id": qid, "category": category, "patient_id": patient_id,
            "wave_reference": wave_ref, "question": question, "answer": answer,
            "verified": "correct", "notes": "deterministic", "temporal": True}


def _print_qa(r):
    sep = "-" * 72
    print(sep)
    print(f"[{r['question_id']}]  patient={r['patient_id']}  wave_ref={r['wave_reference']}")
    print(f"Q: {r['question']}")
    print(f"A: {r['answer']}")


# ── Section A: Multi-wave trajectory ──────────────────────────────────────────

def gen_A1(obs):
    bmi = obs[obs["DESCRIPTION"] == "Body Mass Index"].dropna(subset=["wave","value_num"]).copy()
    bmi["wave"] = bmi["wave"].astype(int)
    wc = bmi.groupby("pid8")["wave"].nunique()
    for m in (5,4,3,2,1):
        el = wc[wc >= m].index
        if len(el): break
    peak = bmi[bmi["pid8"].isin(el)].groupby(["pid8","wave"])["value_num"].max().reset_index(name="max_bmi")
    bpp  = peak.loc[peak.groupby("pid8")["max_bmi"].idxmax()].reset_index(drop=True)
    c    = bpp.sort_values("max_bmi", ascending=False).iloc[0]
    p,w,v = c["pid8"], int(c["wave"]), round(float(c["max_bmi"]),1)
    return mkrow("EXT_TRAJ_001","trajectory",p,f"Wave {w}",
                 f"Across all waves, in which wave did patient {p} reach their peak recorded Body Mass Index?",
                 f"Wave {w} ({v} kg/m2)"), p


def gen_A2(obs, anchor_pid=None):
    bmi = obs[obs["DESCRIPTION"] == "Body Mass Index"].dropna(subset=["wave","value_num"]).copy()
    bmi["wave"] = bmi["wave"].astype(int)
    wc = bmi.groupby("pid8")["wave"].nunique()
    for m in (5,4,3,2):
        el = wc[wc >= m].index
        if len(el): break
    wa = bmi[bmi["pid8"].isin(el)].groupby(["pid8","wave"])["value_num"].mean().reset_index(name="avg_bmi")
    best_pid,best_d,best_row = None,-1e9,None
    for pid,g in wa.groupby("pid8"):
        g = g.sort_values("wave")
        fw,fv = int(g.iloc[0]["wave"]), round(float(g.iloc[0]["avg_bmi"]),1)
        lw,lv = int(g.iloc[-1]["wave"]),round(float(g.iloc[-1]["avg_bmi"]),1)
        d = lv-fv
        if d > best_d: best_d,best_pid,best_row = d,pid,(fw,fv,lw,lv)
    fw,fv,lw,lv = best_row
    a = (f"Yes -- BMI increased from {fv} kg/m2 (Wave {fw}) to {lv} kg/m2 (Wave {lw})" if best_d>0
         else f"No -- BMI did not increase overall: {fv} kg/m2 (Wave {fw}) to {lv} kg/m2 (Wave {lw})")
    return mkrow("EXT_TRAJ_002","trajectory",best_pid,"All waves",
                 f"Did patient {best_pid}'s Body Mass Index show an overall increasing trend across all waves they appear in?", a)


def gen_A3(obs):
    sbp = obs[obs["DESCRIPTION"] == "Systolic Blood Pressure"].dropna(subset=["wave","value_num"]).copy()
    sbp["wave"] = sbp["wave"].astype(int)
    high = sbp[sbp["value_num"] > 130]
    wc = high.groupby("pid8")["wave"].nunique().reset_index(name="n_waves").sort_values("n_waves",ascending=False)
    for m in (3,2,1):
        el = wc[wc["n_waves"] >= m]
        if len(el): break
    c = el.iloc[0]; p,n = c["pid8"],int(c["n_waves"])
    ws = sorted(high[high["pid8"]==p]["wave"].unique())
    return mkrow("EXT_TRAJ_003","trajectory",p,"All waves",
                 f"In how many waves did patient {p} have a recorded Systolic Blood Pressure above 130 mm[Hg]?",
                 f"{n} waves ({', '.join(f'Wave {w}' for w in ws)})")


def gen_A4(obs):
    wt = obs[obs["DESCRIPTION"] == "Body Weight"].dropna(subset=["wave","value_num"]).copy()
    wt["wave"] = wt["wave"].astype(int)
    wc = wt.groupby("pid8")["wave"].nunique()
    for m in (3,2,1):
        el = wc[wc >= m].index
        if len(el): break
    agg = (wt[wt["pid8"].isin(el)].groupby("pid8")["value_num"]
           .agg(["min","max"]).assign(r=lambda d: d["max"]-d["min"]).sort_values("r",ascending=False))
    p   = agg.index[0]; pw = wt[wt["pid8"]==p]
    mnr,mxr = pw.loc[pw["value_num"].idxmin()],pw.loc[pw["value_num"].idxmax()]
    return mkrow("EXT_TRAJ_004","trajectory",p,"All waves",
                 f"Across all waves, what was the range of Body Weight recorded for patient {p} (min to max)?",
                 f"{round(float(mnr['value_num']),1)} kg (Wave {int(mnr['wave'])}) to "
                 f"{round(float(mxr['value_num']),1)} kg (Wave {int(mxr['wave'])})")


def gen_A5(obs):
    val = obs.dropna(subset=["wave","DESCRIPTION"]).copy()
    val["wave"] = val["wave"].astype(int)
    dist = val.groupby(["pid8","wave"])["DESCRIPTION"].nunique().reset_index(name="n").sort_values("n",ascending=False)
    c = dist.iloc[0]; p,w,n = c["pid8"],int(c["wave"]),int(c["n"])
    return mkrow("EXT_TRAJ_005","trajectory",p,f"Wave {w}",
                 f"In which wave did patient {p} have the highest number of distinct observations recorded?",
                 f"Wave {w} ({n} distinct observation types)")


# ── Section B: Co-occurrence and comorbidity ───────────────────────────────────

def _htn_mask(cond):
    return cond["DESCRIPTION"].str.contains("hypertension", case=False, na=False)


def gen_B1(cond):
    mask = _htn_mask(cond)
    htn_exp   = active_expand(cond[mask].copy())
    other_exp = active_expand(cond[~mask].copy())
    htn_pw    = set(zip(htn_exp["pid8"],   htn_exp["active_wave"].astype(int)))
    other_pw  = set(zip(other_exp["pid8"], other_exp["active_wave"].astype(int)))
    both      = htn_pw & other_pw
    counts    = Counter(pid for pid,_ in both) or Counter(pid for pid,_ in htn_pw)
    p,n       = counts.most_common(1)[0]
    return mkrow("EXT_COMORBID_001","comorbidity",p,"All waves",
                 f"In how many waves was patient {p} recorded with both Hypertension and at least one other active condition?",
                 f"{n} waves")


def gen_B2(cond):
    mask    = _htn_mask(cond)
    chronic = cond[~mask & cond["STOP"].isna()].copy()
    if len(chronic) == 0:
        chronic = cond[~mask].copy()
    htn_f = (cond[mask].sort_values("START").groupby("pid8").first()[["START"]]
             .rename(columns={"START":"htn_start"}).reset_index())
    chr_f = (chronic.sort_values("START").groupby("pid8")
             .first()[["START","DESCRIPTION"]]
             .rename(columns={"START":"chr_start","DESCRIPTION":"chr_desc"}).reset_index())
    merged = htn_f.merge(chr_f, on="pid8")
    # require both dates in valid wave range
    in_range = merged[
        (merged["htn_start"].str[:4].astype(int) >= 1990) &
        (merged["chr_start"].str[:4].astype(int) >= 1990)]
    if len(in_range): merged = in_range
    hfirst = merged[merged["htn_start"] < merged["chr_start"]]
    c      = hfirst.iloc[0] if len(hfirst) else merged.iloc[0]
    p,hd,cd,desc = c["pid8"],str(c["htn_start"])[:10],str(c["chr_start"])[:10],c["chr_desc"]
    hw,cw = wave_num(int(hd[:4])), wave_num(int(cd[:4]))
    a = (f"Hypertension was diagnosed first (Wave {hw}, {hd})"
         if hd <= cd else f"{desc} was diagnosed first (Wave {cw}, {cd})")
    return mkrow("EXT_COMORBID_002","comorbidity",p,"All waves",
                 f"Which condition was diagnosed first for patient {p} -- Hypertension or {desc}?", a)


def gen_B3(cond):
    val = cond.dropna(subset=["wave"]).copy(); val["wave"] = val["wave"].astype(int)
    cnt = (val.groupby(["pid8","wave"])["DESCRIPTION"].count()
           .reset_index(name="n").sort_values("n",ascending=False))
    for m in (3,2,1):
        el = cnt[cnt["n"] >= m]
        if len(el): break
    c = el.iloc[0]; p,w,n = c["pid8"],int(c["wave"]),int(c["n"])
    names = val[(val["pid8"]==p)&(val["wave"]==w)]["DESCRIPTION"].tolist()
    ns = "; ".join(names[:5]) + (f" (and {len(names)-5} more)" if len(names)>5 else "")
    a = (f"Yes -- Wave {w} had {n} conditions: {ns}" if n>=3
         else f"No -- maximum was {n} condition in Wave {w}: {ns}")
    return mkrow("EXT_COMORBID_003","comorbidity",p,f"Wave {w}",
                 f"Did patient {p} ever have three or more conditions diagnosed in the same wave?", a)


def gen_B4(cond):
    mask  = _htn_mask(cond)
    htn_f = (cond[mask].sort_values("START").groupby("pid8").first()[["START"]]
             .rename(columns={"START":"htn_start"}).reset_index())
    oth_f = (cond[~mask].sort_values("START").groupby("pid8")
             .first()[["START","DESCRIPTION"]]
             .rename(columns={"START":"oth_start","DESCRIPTION":"oth_desc"}).reset_index())
    merged = htn_f.merge(oth_f, on="pid8")
    ir = merged[
        (merged["htn_start"].str[:4].astype(int) >= 1990) &
        (merged["oth_start"].str[:4].astype(int) >= 1990)]
    if len(ir): merged = ir
    yes = merged[merged["htn_start"] < merged["oth_start"]]
    c   = yes.iloc[0] if len(yes) else merged.iloc[0]
    p,hd,od,desc = c["pid8"],str(c["htn_start"])[:10],str(c["oth_start"])[:10],c["oth_desc"]
    hw,ow = wave_num(int(hd[:4])), wave_num(int(od[:4]))
    a = (f"Yes -- Hypertension Wave {hw} ({hd}), {desc} Wave {ow} ({od})"
         if hd < od else f"No -- {desc} Wave {ow} ({od}) preceded Hypertension Wave {hw} ({hd})")
    return mkrow("EXT_COMORBID_004","comorbidity",p,"All waves",
                 f"Was patient {p} already diagnosed with Hypertension when they were first diagnosed with {desc}?", a)


def gen_B5(cond):
    exp = active_expand(cond)
    cnt = (exp.groupby(["pid8","active_wave"])["DESCRIPTION"].count()
           .reset_index(name="n").sort_values("n",ascending=False))
    c = cnt.iloc[0]; p,w,n = c["pid8"],int(c["active_wave"]),int(c["n"])
    return mkrow("EXT_COMORBID_005","comorbidity",p,f"Wave {w}",
                 f"In which wave did patient {p} have the highest comorbidity burden (most simultaneous active conditions)?",
                 f"Wave {w} with {n} active conditions")


# ── Section C: Medication involvement ─────────────────────────────────────────

def gen_C1(meds):
    mwc = meds.dropna(subset=["wave"]).groupby("pid8")["wave"].nunique()
    for m in (3,2,1):
        el = mwc[mwc >= m].index
        if len(el): break
    uc = (meds[meds["pid8"].isin(el)].groupby("pid8")["DESCRIPTION"].nunique()
          .reset_index(name="n").sort_values("n",ascending=False))
    c = uc.iloc[0]; p,n = c["pid8"],int(c["n"])
    return mkrow("EXT_MED_001","medication",p,"All waves",
                 f"How many unique medications was patient {p} ever prescribed across all waves?",
                 f"{n} unique medications")


def gen_C2(meds):
    val = meds.dropna(subset=["wave"]).sort_values("START")
    first = val.groupby("pid8").first().reset_index().sort_values("START").iloc[0]
    p,w,d,m = first["pid8"],int(first["wave"]),str(first["START"])[:10],first["DESCRIPTION"]
    return mkrow("EXT_MED_002","medication",p,f"Wave {w}",
                 f"In which wave was patient {p} first prescribed any medication?",
                 f"Wave {w} ({d}, {m})")


def gen_C3(cond, meds):
    htn = cond[_htn_mask(cond)].dropna(subset=["wave"]).copy()
    htn["wave"] = htn["wave"].astype(int)
    mv = meds.dropna(subset=["wave"]).copy(); mv["wave"] = mv["wave"].astype(int)
    htn_f = (htn.sort_values("START").groupby("pid8")
             .first()[["wave","START"]].rename(columns={"wave":"htn_wave","START":"htn_date"}).reset_index())
    p = w = a = None
    for _,hr in htn_f.iterrows():
        pid,hw,hd = hr["pid8"],int(hr["htn_wave"]),str(hr["htn_date"])[:10]
        sw = mv[(mv["pid8"]==pid)&(mv["wave"]==hw)]
        if len(sw):
            mr = sw.sort_values("START").iloc[0]
            p,w = pid,hw
            a = (f"Yes -- {mr['DESCRIPTION']} prescribed Wave {w} ({str(mr['START'])[:10]}), "
                 f"same wave as Hypertension diagnosis ({hd})")
            break
    if p is None:
        mf = (mv.sort_values("START").groupby("pid8").first()
              [["wave","START","DESCRIPTION"]].rename(columns={"wave":"mw","START":"md"}).reset_index())
        mg = htn_f.merge(mf, on="pid8")
        if len(mg):
            c2 = mg.iloc[0]; p = c2["pid8"]; w = int(c2["htn_wave"])
            a = (f"No -- Hypertension diagnosed Wave {w} ({str(c2['htn_date'])[:10]}), "
                 f"first medication came in Wave {int(c2['mw'])}")
        else:
            return mkrow("EXT_MED_003","medication","unknown","All waves",
                         "Did patient receive a new medication in the same wave as Hypertension?","Insufficient data")
    return mkrow("EXT_MED_003","medication",p,f"Wave {w}",
                 f"Did patient {p} receive a new medication in the same wave they were first diagnosed with Hypertension?", a)


def gen_C4(meds):
    exp = active_expand(meds)
    wc  = (exp.groupby("pid8")["active_wave"].nunique()
           .reset_index(name="n").sort_values("n",ascending=False))
    c = wc.iloc[0]; p,n = c["pid8"],int(c["n"])
    waves = sorted(exp[exp["pid8"]==p]["active_wave"].unique())
    return mkrow("EXT_MED_004","medication",p,"All waves",
                 f"How many waves did patient {p} have at least one active medication?",
                 f"{n} waves ({', '.join(f'Wave {w}' for w in waves)})")


def gen_C5(meds):
    val = meds.dropna(subset=["wave"]).copy(); val["wave"] = val["wave"].astype(int)
    rows = []
    for pid,g in val.groupby("pid8"):
        pw = sorted(g["wave"].unique())
        if len(pw) < 2: continue
        mid = len(pw)//2
        ne = g[g["wave"].isin(pw[:mid])]["DESCRIPTION"].nunique()
        nl = g[g["wave"].isin(pw[mid:])]["DESCRIPTION"].nunique()
        rows.append((pid,ne,nl,nl-ne,pw[:mid],pw[mid:]))
    rd = pd.DataFrame(rows,columns=["pid8","ne","nl","d","ew","lw"])
    yes = rd[rd["d"]>0].sort_values("d",ascending=False)
    c   = yes.iloc[0] if len(yes) else rd.sort_values("ne",ascending=False).iloc[0]
    p,ne,nl = c["pid8"],int(c["ne"]),int(c["nl"])
    es = "/".join(f"W{w}" for w in c["ew"]); ls = "/".join(f"W{w}" for w in c["lw"])
    is_yes = c["d"] > 0
    return mkrow("EXT_MED_005","medication",p,"All waves",
                 f"Was patient {p} prescribed more medications in later waves than earlier waves?",
                 f"{'Yes' if is_yes else 'No'} -- {ne} medications in early waves ({es}) vs {nl} in later waves ({ls})")


# ── Section D: Intra-wave depth ───────────────────────────────────────────────

def gen_D1(obs):
    """Most distinct observation type names in a single patient+wave."""
    val = obs.dropna(subset=["wave","DESCRIPTION"]).copy()
    val["wave"] = val["wave"].astype(int)
    dist = (val.groupby(["pid8","wave"])["DESCRIPTION"].nunique()
            .reset_index(name="n").sort_values("n", ascending=False))
    c = dist.iloc[0]; p,w,n = c["pid8"],int(c["wave"]),int(c["n"])
    return mkrow("EXT_INTRAWAVE_001","intrawave",p,f"Wave {w}",
                 f"How many distinct vital sign types were recorded for patient {p} in Wave {w}?",
                 f"{n} distinct types")


def gen_D2(obs):
    """Highest single SBP reading across all patient+wave combinations."""
    sbp = obs[obs["DESCRIPTION"] == "Systolic Blood Pressure"].dropna(subset=["wave","value_num"]).copy()
    sbp["wave"] = sbp["wave"].astype(int)
    mx  = (sbp.groupby(["pid8","wave"])["value_num"].max()
           .reset_index(name="max_sbp").sort_values("max_sbp", ascending=False))
    c   = mx.iloc[0]; p,w,v = c["pid8"],int(c["wave"]),int(c["max_sbp"])
    row = sbp[(sbp["pid8"]==p)&(sbp["wave"]==w)&(sbp["value_num"]==c["max_sbp"])].iloc[0]
    d   = str(row["DATE"])[:10]
    return mkrow("EXT_INTRAWAVE_002","intrawave",p,f"Wave {w}",
                 f"What was the maximum recorded Systolic Blood Pressure for patient {p} within Wave {w}?",
                 f"{v} mm[Hg] ({d})")


def gen_D3(obs):
    """Patient+wave that has both a BMI reading and an SBP reading."""
    val  = obs.dropna(subset=["wave","value_num"]).copy()
    val["wave"] = val["wave"].astype(int)
    bmi_a = (val[val["DESCRIPTION"]=="Body Mass Index"]
             .groupby(["pid8","wave"])["value_num"].mean().reset_index(name="avg_bmi"))
    sbp_a = (val[val["DESCRIPTION"]=="Systolic Blood Pressure"]
             .groupby(["pid8","wave"])["value_num"].max().reset_index(name="max_sbp"))
    mg = bmi_a.merge(sbp_a, on=["pid8","wave"])
    if len(mg) == 0:
        return mkrow("EXT_INTRAWAVE_003","intrawave","unknown","All waves",
                     "Did patient have both BMI and SBP?","Insufficient data")
    c   = mg.sort_values("avg_bmi", ascending=False).iloc[0]
    p,w = c["pid8"],int(c["wave"])
    bv  = round(float(c["avg_bmi"]),1); sv = int(round(float(c["max_sbp"]),0))
    return mkrow("EXT_INTRAWAVE_003","intrawave",p,f"Wave {w}",
                 f"Did patient {p} have both a BMI observation AND a blood pressure observation in Wave {w}?",
                 f"Yes -- BMI: {bv} kg/m2, SBP: {sv} mm[Hg]")


def gen_D4(cond, obs, meds):
    """Patient+wave with most total health events (conditions + obs + medications)."""
    def cnt(df, wcol="wave"):
        v = df.dropna(subset=[wcol]).copy(); v[wcol] = v[wcol].astype(int)
        return v.groupby(["pid8",wcol])["DESCRIPTION"].count().reset_index(name="n")
    cc = cnt(cond); oc = cnt(obs); mc = cnt(meds)
    mg = (cc.merge(oc, on=["pid8","wave"], suffixes=("_c","_o"))
            .merge(mc, on=["pid8","wave"]))
    mg = mg.rename(columns={"n_c":"nc","n_o":"no","n":"nm"})
    mg["total"] = mg["nc"]+mg["no"]+mg["nm"]
    c = mg.sort_values("total",ascending=False).iloc[0]
    p,w = c["pid8"],int(c["wave"])
    nc,no,nm,tot = int(c["nc"]),int(c["no"]),int(c["nm"]),int(c["total"])
    return mkrow("EXT_INTRAWAVE_004","intrawave",p,f"Wave {w}",
                 f"What was the total number of health events (conditions + observations + medications) "
                 f"for patient {p} in Wave {w}?",
                 f"{tot} total ({nc} conditions, {no} observations, {nm} medications)")


def gen_D5(cond):
    """Wave with the most conditions appearing for the first time for each patient."""
    val = cond.dropna(subset=["wave"]).copy(); val["wave"] = val["wave"].astype(int)
    # First occurrence of each (patient, description) pair — that wave = "new condition" wave
    first_occ = (val.sort_values("START")
                 .groupby(["pid8","DESCRIPTION"]).first()[["wave"]].reset_index())
    new_pw = (first_occ.groupby(["pid8","wave"])["DESCRIPTION"]
              .agg(n_new="count", sample="first").reset_index()
              .sort_values("n_new", ascending=False))
    c = new_pw.iloc[0]; p,w,n = c["pid8"],int(c["wave"]),int(c["n_new"])
    names = first_occ[(first_occ["pid8"]==p)&(first_occ["wave"]==w)]["DESCRIPTION"].tolist()
    ns = "; ".join(names[:5]) + (f" (and {len(names)-5} more)" if len(names)>5 else "")
    return mkrow("EXT_INTRAWAVE_005","intrawave",p,f"Wave {w}",
                 f"How many of patient {p}'s conditions in Wave {w} were new (not present in any prior wave)?",
                 f"{n} new conditions: {ns}")


# ── Section E: Cross-table patterns ───────────────────────────────────────────

def gen_E1(cond, meds):
    """Wave where medication count increased AND a new condition appeared — vectorized."""
    vc = cond.dropna(subset=["wave"]).copy(); vc["wave"] = vc["wave"].astype(int)
    vm = meds.dropna(subset=["wave"]).copy(); vm["wave"] = vm["wave"].astype(int)

    # First occurrence of each condition per patient (= "new" condition wave)
    fo = (vc.sort_values("START").groupby(["pid8","DESCRIPTION"]).first()[["wave"]].reset_index())
    new_cond = (fo.groupby(["pid8","wave"])
                .agg(n_new=("DESCRIPTION","count"), first_desc=("DESCRIPTION","first"))
                .reset_index())

    # Medication counts per patient per wave; shift within group for previous-wave count
    mc = (vm.groupby(["pid8","wave"])["DESCRIPTION"].count()
          .reset_index(name="n_meds").sort_values(["pid8","wave"]))
    mc["prev_n"] = mc.groupby("pid8")["n_meds"].shift(1).fillna(0).astype(int)
    mc["inc"]    = mc["n_meds"] > mc["prev_n"]

    # Rows where meds increased this wave
    inc_rows = mc[mc["inc"]]

    # Join with waves that have new conditions
    both = inc_rows.merge(new_cond, on=["pid8","wave"])

    if len(both) == 0:
        # Fallback: report the most dramatic medication increase wave (without new condition)
        c = mc.sort_values("n_meds", ascending=False).iloc[0]
        return mkrow("EXT_PATTERN_001","pattern",c["pid8"],f"Wave {int(c['wave'])}",
                     f"Did patient {c['pid8']}'s medication count increase in the same wave they were diagnosed with a new condition?",
                     "No concurrent medication spike and new condition found in the same wave")

    c = both.sort_values("n_meds", ascending=False).iloc[0]
    p,w,prev,curr,desc = (c["pid8"],int(c["wave"]),int(c["prev_n"]),
                           int(c["n_meds"]),c["first_desc"])
    return mkrow("EXT_PATTERN_001","pattern",p,f"Wave {w}",
                 f"Did patient {p}'s medication count increase in the same wave they were diagnosed with a new condition?",
                 f"Yes -- Wave {w}: new condition '{desc}', medications increased from {prev} to {curr}")


def gen_E2(cond, obs):
    """BMI at the time of first Hypertension diagnosis."""
    htn = cond[_htn_mask(cond)].dropna(subset=["wave"]).copy()
    htn["wave"] = htn["wave"].astype(int)
    # Restrict to valid wave range
    htn = htn[htn["start_year"] >= 1990]

    bmi = obs[obs["DESCRIPTION"] == "Body Mass Index"].dropna(subset=["wave","value_num"]).copy()
    bmi["wave"] = bmi["wave"].astype(int)

    htn_f = (htn.sort_values("START").groupby("pid8")
             .first()[["wave","START"]].rename(columns={"wave":"hw","START":"hd"}).reset_index())
    bmi_a = bmi.groupby(["pid8","wave"])["value_num"].mean().reset_index(name="avg_bmi")

    # Join: same wave as HTN first diagnosis
    mg = htn_f.merge(bmi_a.rename(columns={"wave":"hw"}), on=["pid8","hw"])

    if len(mg) == 0:
        return mkrow("EXT_PATTERN_002","pattern","unknown","All waves",
                     "Was BMI above 30 at Hypertension diagnosis?","Insufficient data")

    yes = mg[mg["avg_bmi"] > 30].sort_values("avg_bmi", ascending=False)
    c   = yes.iloc[0] if len(yes) else mg.sort_values("avg_bmi", ascending=False).iloc[0]
    p,w  = c["pid8"],int(c["hw"])
    bv   = round(float(c["avg_bmi"]),1)
    hd   = str(c["hd"])[:10]
    is_y = bv > 30
    a = (f"Yes -- BMI was {bv} kg/m2 in Wave {w} when Hypertension diagnosed ({hd})"
         if is_y else f"No -- BMI was {bv} kg/m2 (below 30) in Wave {w} when Hypertension diagnosed ({hd})")
    return mkrow("EXT_PATTERN_002","pattern",p,f"Wave {w}",
                 f"Was patient {p}'s BMI above 30 in the wave they were first diagnosed with Hypertension?", a)


def gen_E3(cond):
    """Active condition burden per wave for the patient with the most sustained comorbidity."""
    exp = active_expand(cond)
    ac  = (exp.groupby(["pid8","active_wave"])["DESCRIPTION"]
           .count().reset_index(name="n_active"))

    # Patients with 4+ waves of any active conditions
    wc = ac.groupby("pid8")["active_wave"].nunique()
    for m in (4,3,2,1):
        el = wc[wc >= m].index
        if len(el): break

    # Among eligible, pick patient with highest total cumulative burden
    total = ac[ac["pid8"].isin(el)].groupby("pid8")["n_active"].sum().sort_values(ascending=False)
    p     = total.index[0]

    pw   = ac[ac["pid8"]==p].sort_values("active_wave")
    burden_str = ", ".join(
        f"Wave {int(r['active_wave'])}: {int(r['n_active'])}" for _,r in pw.iterrows())
    return mkrow("EXT_PATTERN_003","pattern",p,"All waves",
                 f"How did patient {p}'s total condition burden change across all waves?",
                 burden_str)


def gen_E4(obs, meds):
    """Patient whose peak-BMI wave also has a medication start."""
    bmi = obs[obs["DESCRIPTION"] == "Body Mass Index"].dropna(subset=["wave","value_num"]).copy()
    bmi["wave"] = bmi["wave"].astype(int)
    vm  = meds.dropna(subset=["wave"]).copy(); vm["wave"] = vm["wave"].astype(int)

    peak = (bmi.groupby(["pid8","wave"])["value_num"].max()
            .reset_index(name="max_bmi"))
    pp   = peak.loc[peak.groupby("pid8")["max_bmi"].idxmax()].reset_index(drop=True)

    # Medication first prescription per patient per wave
    mfpw = (vm.groupby(["pid8","wave"])
            .agg(n_meds=("DESCRIPTION","count"), first_med=("DESCRIPTION","first"))
            .reset_index())

    mg = pp.merge(mfpw, on=["pid8","wave"])

    if len(mg) > 0:
        c  = mg.sort_values("max_bmi", ascending=False).iloc[0]
        p,w,bv,m = c["pid8"],int(c["wave"]),round(float(c["max_bmi"]),1),c["first_med"]
        a = f"Yes -- BMI peaked at {bv} kg/m2 in Wave {w}, '{m}' prescribed that wave"
    else:
        c  = pp.sort_values("max_bmi", ascending=False).iloc[0]
        p,w,bv = c["pid8"],int(c["wave"]),round(float(c["max_bmi"]),1)
        a = f"No -- BMI peaked at {bv} kg/m2 in Wave {w}, no new medication that wave"

    return mkrow("EXT_PATTERN_004","pattern",p,f"Wave {w}",
                 f"Did patient {p} receive a new medication in the same wave as their highest recorded BMI?", a)


def gen_E5(obs):
    """Percentage of observations that are cardiovascular for the most data-rich patient."""
    val  = obs[obs["value_num"].notna()].copy()
    cardio = {"Systolic Blood Pressure","Diastolic Blood Pressure","Heart rate"}
    val["is_cardio"] = val["DESCRIPTION"].isin(cardio).astype(int)
    agg = (val.groupby("pid8").agg(nc=("is_cardio","sum"), nt=("is_cardio","count")).reset_index())
    el  = agg[agg["nt"] >= 50] if (agg["nt"] >= 50).any() else agg
    c   = el.sort_values("nt", ascending=False).iloc[0]
    p,nc,nt = c["pid8"],int(c["nc"]),int(c["nt"])
    pct = round(100*nc/nt, 1)
    return mkrow("EXT_PATTERN_005","pattern",p,"All waves",
                 f"Across patient {p}'s full history, what percentage of their observations were "
                 f"cardiovascular measurements (Systolic Blood Pressure, Diastolic Blood Pressure, Heart rate)?",
                 f"{pct}% ({nc} cardiovascular out of {nt} total observations)")


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    patients, conditions, obs, medications = load_data()

    # Load existing rows — never regenerate already-present questions
    if os.path.exists(OUT_PATH):
        existing = pd.read_csv(OUT_PATH)
        done_ids = set(existing["question_id"].tolist())
        print(f"\nExisting rows: {len(existing)}  already done: {sorted(done_ids)}")
    else:
        existing = pd.DataFrame(columns=COLS)
        done_ids = set()

    new_rows = []

    def maybe(fn, qid, *args):
        if qid in done_ids:
            print(f"  {qid} already present — skipping")
            return
        r = fn(*args)
        _print_qa(r)
        new_rows.append(r)

    # ── Section A (skip if already written) ───────────────────────────────────
    A_ids = {"EXT_TRAJ_001","EXT_TRAJ_002","EXT_TRAJ_003","EXT_TRAJ_004","EXT_TRAJ_005"}
    if not (done_ids & A_ids):
        print("\nGenerating Section A — Multi-wave trajectory...")
        r1, anchor = gen_A1(obs)
        _print_qa(r1); new_rows.append(r1)
        for fn,qid,args in [
            (gen_A2,"EXT_TRAJ_002",(obs,anchor)),
            (gen_A3,"EXT_TRAJ_003",(obs,)),
            (gen_A4,"EXT_TRAJ_004",(obs,)),
            (gen_A5,"EXT_TRAJ_005",(obs,)),
        ]:
            maybe(fn, qid, *args)

    # ── Section B ─────────────────────────────────────────────────────────────
    print("\nGenerating Section B — Co-occurrence / comorbidity...")
    for fn,qid in [(gen_B1,"EXT_COMORBID_001"),(gen_B2,"EXT_COMORBID_002"),
                   (gen_B3,"EXT_COMORBID_003"),(gen_B4,"EXT_COMORBID_004"),
                   (gen_B5,"EXT_COMORBID_005")]:
        maybe(fn, qid, conditions)

    # ── Section C ─────────────────────────────────────────────────────────────
    print("\nGenerating Section C — Medication involvement...")
    maybe(gen_C1,"EXT_MED_001", medications)
    maybe(gen_C2,"EXT_MED_002", medications)
    maybe(gen_C3,"EXT_MED_003", conditions, medications)
    maybe(gen_C4,"EXT_MED_004", medications)
    maybe(gen_C5,"EXT_MED_005", medications)

    # ── Section D ─────────────────────────────────────────────────────────────
    print("\nGenerating Section D — Intra-wave depth...")
    maybe(gen_D1,"EXT_INTRAWAVE_001", obs)
    maybe(gen_D2,"EXT_INTRAWAVE_002", obs)
    maybe(gen_D3,"EXT_INTRAWAVE_003", obs)
    maybe(gen_D4,"EXT_INTRAWAVE_004", conditions, obs, medications)
    maybe(gen_D5,"EXT_INTRAWAVE_005", conditions)

    # ── Section E ─────────────────────────────────────────────────────────────
    print("\nGenerating Section E — Cross-table patterns...")
    maybe(gen_E1,"EXT_PATTERN_001", conditions, medications)
    maybe(gen_E2,"EXT_PATTERN_002", conditions, obs)
    maybe(gen_E3,"EXT_PATTERN_003", conditions)
    maybe(gen_E4,"EXT_PATTERN_004", obs, medications)
    maybe(gen_E5,"EXT_PATTERN_005", obs)

    # ── append & save ──────────────────────────────────────────────────────────
    if new_rows:
        appended = pd.concat([existing, pd.DataFrame(new_rows, columns=COLS)], ignore_index=True)
        appended.to_csv(OUT_PATH, index=False)
        print(f"\nAppended {len(new_rows)} new rows.")
    else:
        appended = existing
        print("\nNo new rows to add.")

    # ── summary ────────────────────────────────────────────────────────────────
    total = len(appended)
    print(f"\nTotal questions in extended benchmark: {total}")
    print()
    print(f"{'Category':<15} | Count")
    print("-" * 22)
    cc = appended["category"].value_counts()
    for cat in ["trajectory","comorbidity","medication","intrawave","pattern"]:
        print(f"{cat:<15} | {cc.get(cat, 0)}")
    print("-" * 22)
    print(f"{'TOTAL':<15} | {total}")
    print("-" * 72)


if __name__ == "__main__":
    main()
