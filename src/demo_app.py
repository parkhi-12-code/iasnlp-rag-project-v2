"""
src/demo_app.py
IASNLP Patient Explorer — Streamlit demo app.

Run: streamlit run src/demo_app.py
"""

import os
import sys
import tempfile
import pickle
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
load_dotenv(ROOT / ".env")

DATA_RAW   = ROOT / "data" / "raw"
GRAPH_PATH = ROOT / "data" / "knowledge_graph.gpickle"

try:
    from graph_rag_pipeline import run_pipeline
    _PIPELINE_OK = True
except Exception as _pe:
    _PIPELINE_OK = False
    _PIPELINE_ERR = str(_pe)

try:
    from pyvis.network import Network as PyvisNetwork
    _PYVIS_OK = True
except ImportError:
    _PYVIS_OK = False

# ── wave constants ─────────────────────────────────────────────────────────────

WAVE_RANGES = {
    1: "1990-1994", 2: "1995-1999", 3: "2000-2004",
    4: "2005-2009", 5: "2010-2014", 6: "2015-2019", 7: "2020+",
}
WAVE_START = {1: 1990, 2: 1995, 3: 2000, 4: 2005, 5: 2010, 6: 2015, 7: 2020}
WAVE_END   = {1: 1994, 2: 1999, 3: 2004, 4: 2009, 5: 2014, 6: 2019, 7: 2030}


def assign_wave(year):
    if year <= 1994: return 1
    elif year <= 1999: return 2
    elif year <= 2004: return 3
    elif year <= 2009: return 4
    elif year <= 2014: return 5
    elif year <= 2019: return 6
    else: return 7


# ── page config (must be first Streamlit call) ────────────────────────────────

st.set_page_config(
    page_title="IASNLP Patient Explorer",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
/* Sidebar navy background */
[data-testid="stSidebar"] { background-color: #1a2744 !important; }
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] div,
[data-testid="stSidebar"] small { color: #e0e8f8 !important; }
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 { color: #ffffff !important; }
[data-testid="stSidebar"] input {
    background-color: #2a3a5a !important;
    color: #ffffff !important;
    border: 1px solid #2196F3 !important;
    border-radius: 4px !important;
}
[data-testid="stSidebar"] hr { border-color: #2a3a5a !important; }

/* Info card */
.info-card {
    background: #1a2744; border: 1px solid #2196F3;
    border-radius: 8px; padding: 14px 16px; color: white; margin: 8px 0;
}
.info-card h4 { color: #2196F3; margin: 0 0 10px 0; font-size: 15px; }
.info-card .row {
    display: flex; justify-content: space-between; align-items: center;
    font-size: 12px; padding: 3px 0;
    border-bottom: 1px solid #2a3a5a;
}
.info-card .row:last-child { border-bottom: none; }
.info-card .lbl { color: #90CAF9; }
.info-card code { background: #2a3a5a; padding: 1px 5px;
    border-radius: 3px; font-size: 11px; color: #90CAF9; }

/* Answer box */
.answer-box {
    background: #E3F2FD; border-left: 4px solid #2196F3;
    border-radius: 0 6px 6px 0; padding: 14px 16px;
    margin: 10px 0; color: #1a2744;
}
.badge {
    display: inline-block; background: #1a2744; color: #2196F3;
    font-size: 10px; padding: 2px 10px; border-radius: 12px;
    margin-bottom: 10px; text-transform: uppercase;
    font-weight: bold; letter-spacing: 0.8px;
    border: 1px solid #2196F3;
}

/* Section headers */
.sec-hdr {
    color: #1a2744; font-size: 17px; font-weight: 700;
    border-left: 4px solid #2196F3; padding-left: 10px;
    margin: 28px 0 12px 0;
}
</style>
""", unsafe_allow_html=True)


# ── data loaders ──────────────────────────────────────────────────────────────

@st.cache_data(show_spinner="Loading patients...")
def load_patients():
    df = pd.read_csv(DATA_RAW / "patients.csv", dtype=str)
    df["pid8"] = df["Id"].str[:8]
    df["FULL_NAME"] = (
        df.get("FIRST", pd.Series(dtype=str)).fillna("") + " " +
        df.get("LAST",  pd.Series(dtype=str)).fillna("")
    ).str.strip()
    return df


@st.cache_data(show_spinner="Loading conditions...")
def load_conditions():
    df = pd.read_csv(DATA_RAW / "conditions.csv", dtype=str)
    df["pid8"]       = df["PATIENT"].str[:8]
    df["start_year"] = pd.to_numeric(df["START"].str[:4], errors="coerce")
    df["stop_year"]  = pd.to_numeric(df["STOP"].str[:4],  errors="coerce")
    df["wave"] = df["start_year"].apply(
        lambda y: assign_wave(int(y)) if pd.notna(y) else None
    )
    return df


@st.cache_data(show_spinner="Loading observations...")
def load_observations():
    df = pd.read_csv(DATA_RAW / "observations.csv", dtype=str)
    df["pid8"] = df["PATIENT"].str[:8]
    df["year"] = pd.to_numeric(df["DATE"].str[:4], errors="coerce")
    df["wave"] = df["year"].apply(
        lambda y: assign_wave(int(y)) if pd.notna(y) else None
    )
    return df


@st.cache_data(show_spinner="Loading medications...")
def load_medications():
    df = pd.read_csv(DATA_RAW / "medications.csv", dtype=str)
    df["pid8"]       = df["PATIENT"].str[:8]
    df["start_year"] = pd.to_numeric(df["START"].str[:4], errors="coerce")
    df["stop_year"]  = pd.to_numeric(df["STOP"].str[:4],  errors="coerce")
    df["wave"] = df["start_year"].apply(
        lambda y: assign_wave(int(y)) if pd.notna(y) else None
    )
    return df


@st.cache_resource(show_spinner="Loading knowledge graph...")
def load_graph_cached():
    if not GRAPH_PATH.exists():
        return None
    with open(GRAPH_PATH, "rb") as f:
        return pickle.load(f)


# ── helpers ───────────────────────────────────────────────────────────────────

def patient_waves(pid8, cond, obs, meds):
    waves = set()
    for df in (cond, obs, meds):
        sub = df[df["pid8"] == pid8]["wave"].dropna()
        waves.update(int(w) for w in sub)
    return sorted(waves)


# ── charts ────────────────────────────────────────────────────────────────────

def _style_fig(fig, ax, title, ylabel=""):
    fig.patch.set_facecolor("#f8fafc")
    ax.set_facecolor("#f8fafc")
    ax.set_title(title, fontsize=10, fontweight="bold", color="#1a2744", pad=5)
    ax.set_xlabel("Wave", fontsize=8, color="#666")
    ax.set_ylabel(ylabel, fontsize=8, color="#666")
    ax.tick_params(labelsize=8, colors="#666")
    for sp in ax.spines.values():
        sp.set_edgecolor("#dee2e6")
    ax.grid(axis="y", color="#dee2e6", linewidth=0.5)
    fig.tight_layout(pad=1.2)


def chart_bmi(pid8, obs):
    sub = obs[
        (obs["pid8"] == pid8) &
        obs["DESCRIPTION"].str.contains("Body Mass Index|BMI", case=False, na=False)
    ].copy()
    sub["VALUE"] = pd.to_numeric(sub["VALUE"], errors="coerce")
    sub = sub.dropna(subset=["VALUE", "wave"])
    if sub.empty:
        return None
    grp = sub.groupby("wave")["VALUE"].mean().sort_index()
    fig, ax = plt.subplots(figsize=(4, 2.8))
    xvals = grp.index.astype(int).tolist()
    ax.plot(xvals, grp.values, marker="o", color="#2196F3", linewidth=2,
            markersize=6, markerfacecolor="white", markeredgewidth=2)
    ax.set_xticks(xvals)
    _style_fig(fig, ax, "BMI per Wave", ylabel="kg/m²")
    return fig


def chart_conditions(pid8, cond):
    sub = cond[cond["pid8"] == pid8].dropna(subset=["wave"])
    if sub.empty:
        return None
    grp = sub.groupby("wave")["DESCRIPTION"].nunique().sort_index()
    fig, ax = plt.subplots(figsize=(4, 2.8))
    xvals = grp.index.astype(int).tolist()
    ax.bar(xvals, grp.values, color="#1a2744", alpha=0.82, width=0.55)
    ax.set_xticks(xvals)
    _style_fig(fig, ax, "Conditions per Wave", ylabel="Unique Conditions")
    return fig


def chart_medications(pid8, meds):
    sub = meds[meds["pid8"] == pid8].dropna(subset=["wave"])
    if sub.empty:
        return None
    grp = sub.groupby("wave")["DESCRIPTION"].nunique().sort_index()
    fig, ax = plt.subplots(figsize=(4, 2.8))
    xvals = grp.index.astype(int).tolist()
    ax.bar(xvals, grp.values, color="#4CAF50", alpha=0.82, width=0.55)
    ax.set_xticks(xvals)
    _style_fig(fig, ax, "Medications per Wave", ylabel="Unique Medications")
    return fig


# ── condition timeline ─────────────────────────────────────────────────────────

def build_timeline(pid8, cond, meds, waves):
    pc = cond[cond["pid8"] == pid8].copy()
    pm = meds[meds["pid8"] == pid8].copy()

    # first wave each condition was diagnosed
    first_wave: dict[str, int] = {}
    if not pc.empty:
        for desc, grp in pc.dropna(subset=["wave"]).groupby("DESCRIPTION"):
            first_wave[desc] = int(grp["wave"].min())

    rows = []
    for w in waves:
        ws, we = WAVE_START[w], WAVE_END[w]

        if not pc.empty and "start_year" in pc.columns:
            active_c = pc[
                (pc["start_year"].fillna(9999).astype(float) <= we) &
                (pc["stop_year"].fillna(9999).astype(float)  >= ws)
            ]["DESCRIPTION"].nunique()
        else:
            active_c = 0

        new_c = sum(1 for fw in first_wave.values() if fw == w)

        if not pm.empty and "start_year" in pm.columns:
            active_m = pm[
                (pm["start_year"].fillna(9999).astype(float) <= we) &
                (pm["stop_year"].fillna(9999).astype(float)  >= ws)
            ]["DESCRIPTION"].nunique()
        else:
            active_m = 0

        rows.append({
            "Wave":               f"Wave {w}",
            "Year Range":         WAVE_RANGES[w],
            "Conditions Active":  active_c,
            "New This Wave":      new_c,
            "Medications Active": active_m,
        })

    return pd.DataFrame(rows)


# ── pyvis subgraph ─────────────────────────────────────────────────────────────

_NODE_COLOR = {
    "patient":     "#1a2744",
    "wave":        "#2196F3",
    "condition":   "#4CAF50",
    "observation": "#FF9800",
    "medication":  "#F44336",
}
_NODE_SHAPE = {
    "condition":   "dot",
    "observation": "square",
    "medication":  "triangle",
}
_MAX_PER_TYPE = 15


def build_pyvis_html(pid8, G):
    if not _PYVIS_OK or G is None or pid8 not in G:
        return None

    net = PyvisNetwork(
        height="460px", width="100%",
        bgcolor="#1a2744", font_color="white",
        directed=False,
    )
    net.set_options("""{
      "nodes": {"font": {"size": 11, "color": "white"}},
      "edges": {"smooth": {"type": "dynamic"}, "color": {"opacity": 0.45}},
      "physics": {
        "stabilization": {"iterations": 80, "fit": true},
        "barnesHut": {"gravitationalConstant": -5000, "springLength": 130}
      }
    }""")

    # Patient node
    net.add_node(
        pid8, label=f"Patient\n{pid8}",
        color=_NODE_COLOR["patient"], size=28,
        shape="star", title=f"Patient {pid8}",
        borderWidth=3,
    )

    # Wave nodes (successors of patient with type="wave")
    wave_nodes = []
    for nb in G.successors(pid8):
        nd = G.nodes.get(nb, {})
        if nd.get("type") != "wave":
            continue
        wn  = nd.get("wave_number", "?")
        lbl = f"Wave {wn}\n{WAVE_RANGES.get(wn, '')}"
        net.add_node(nb, label=lbl, color=_NODE_COLOR["wave"],
                     size=18, shape="ellipse", title=f"Wave {wn}")
        net.add_edge(pid8, nb, color="#90CAF9", width=2)
        wave_nodes.append(nb)

    # Clinical nodes under each wave
    for wave_node in wave_nodes:
        counts: dict[str, int] = {"condition": 0, "observation": 0, "medication": 0}
        for clin in G.successors(wave_node):
            cd    = G.nodes.get(clin, {})
            ntype = cd.get("type", "")
            if ntype not in counts or counts[ntype] >= _MAX_PER_TYPE:
                continue
            counts[ntype] += 1
            desc  = str(cd.get("description", clin))
            label = desc[:22] + ("..." if len(desc) > 22 else "")
            net.add_node(
                clin, label=label,
                color=_NODE_COLOR.get(ntype, "#9E9E9E"),
                size=9,
                shape=_NODE_SHAPE.get(ntype, "dot"),
                title=desc,
            )
            net.add_edge(wave_node, clin, color="#546E7A", width=1)

    tmp = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
            tmp = f.name
        net.save_graph(tmp)
        with open(tmp, "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return None
    finally:
        if tmp:
            try:
                os.unlink(tmp)
            except OSError:
                pass


# ── app ───────────────────────────────────────────────────────────────────────

def main():
    patients = load_patients()
    cond     = load_conditions()
    obs      = load_observations()
    meds     = load_medications()
    G        = load_graph_cached()

    selected_pid8: str | None = None

    # ── SIDEBAR ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## IASNLP Patient Explorer")
        st.markdown("---")

        search_q = st.text_input(
            "Search by name or date of birth",
            placeholder="e.g. Johnson  or  1952-08",
            key="search_input",
        )

        if search_q.strip():
            mask = (
                patients["FULL_NAME"].str.contains(search_q, case=False, na=False) |
                patients["BIRTHDATE"].astype(str).str.contains(search_q, case=False, na=False)
            )
            matches = patients[mask].head(30)
        else:
            matches = patients.head(30)

        if matches.empty:
            st.warning("No patients found.")
        else:
            radio_keys = matches["pid8"].tolist()
            radio_fmt  = {
                row["pid8"]: (
                    f"{row['FULL_NAME']}  |  "
                    f"{row.get('BIRTHDATE','?')}  |  "
                    f"{row.get('GENDER','?')}"
                )
                for _, row in matches.iterrows()
            }
            chosen = st.radio(
                "Select patient:",
                radio_keys,
                format_func=lambda k: radio_fmt[k],
                key="patient_radio",
            )
            selected_pid8 = chosen

        # ── Info card ─────────────────────────────────────────────────────────
        if selected_pid8:
            prow    = patients[patients["pid8"] == selected_pid8].iloc[0]
            pc      = cond[cond["pid8"] == selected_pid8]
            pm      = meds[meds["pid8"] == selected_pid8]
            n_waves = int(pc["wave"].dropna().nunique())
            total_c = len(pc)
            total_m = len(pm)

            st.markdown(f"""
<div class="info-card">
  <h4>{prow.get("FULL_NAME", "Unknown")}</h4>
  <div class="row"><span class="lbl">DOB</span><span>{prow.get("BIRTHDATE","--")}</span></div>
  <div class="row"><span class="lbl">Gender</span><span>{prow.get("GENDER","--")}</span></div>
  <div class="row"><span class="lbl">Race</span><span>{prow.get("RACE","--")}</span></div>
  <div class="row"><span class="lbl">Waves active</span><span>{n_waves}</span></div>
  <div class="row"><span class="lbl">Conditions</span><span>{total_c}</span></div>
  <div class="row"><span class="lbl">Medications</span><span>{total_m}</span></div>
  <div class="row"><span class="lbl">Patient ID</span><code>{selected_pid8}</code></div>
</div>
            """, unsafe_allow_html=True)

            # ── Quick question buttons ────────────────────────────────────────
            st.markdown("**Quick Questions:**")
            quick_qs = [
                (
                    f"What conditions were present in Wave 5 for patient {selected_pid8}?",
                    "Wave 5 conditions",
                ),
                (
                    f"How did BMI change across waves for patient {selected_pid8}?",
                    "BMI trend",
                ),
                (
                    f"What medications were prescribed to patient {selected_pid8}?",
                    "Medications",
                ),
            ]
            for q_full, q_label in quick_qs:
                if st.button(q_label, key=f"qq_{q_label}", use_container_width=True):
                    st.session_state["question_text"] = q_full

    # ── MAIN AREA ─────────────────────────────────────────────────────────────
    st.markdown("# IASNLP Patient Explorer")

    if not selected_pid8:
        st.info("Select a patient from the sidebar to begin exploring.")
        return

    prow = patients[patients["pid8"] == selected_pid8].iloc[0]
    name = prow.get("FULL_NAME", selected_pid8)
    st.markdown(f"Showing data for **{name}** — Patient ID: `{selected_pid8}`")

    waves = patient_waves(selected_pid8, cond, obs, meds)

    # ── SECTION 1: Ask a Question ─────────────────────────────────────────────
    st.markdown('<div class="sec-hdr">Ask a Question</div>', unsafe_allow_html=True)

    question = st.text_input(
        "Your question:",
        placeholder=f"Ask anything about {name}...",
        key="question_text",
    )

    col_sub, col_play, _col_gap = st.columns([1, 2, 7])
    with col_sub:
        submit = st.button("Submit", type="primary")
    with col_play:
        play_btn = st.button("Play answer")

    if submit and question.strip():
        if not _PIPELINE_OK:
            st.error(f"Pipeline not available: {_PIPELINE_ERR}")
        else:
            with st.spinner("Querying knowledge graph..."):
                try:
                    result = run_pipeline(
                        question=question,
                        patient_id=selected_pid8,
                        wave_reference="All waves",
                    )
                    st.session_state["last_answer"]   = result.get("answer", "No answer returned.")
                    st.session_state["last_category"] = result.get("category", "lookup")
                except Exception as exc:
                    st.error(f"Pipeline error: {exc}")

    if play_btn:
        st.success("Audio playback coming soon")

    if "last_answer" in st.session_state:
        cat = st.session_state.get("last_category", "lookup")
        st.markdown(
            f'<div class="answer-box">'
            f'<span class="badge">{cat}</span><br><br>'
            f'{st.session_state["last_answer"]}'
            f'</div>',
            unsafe_allow_html=True,
        )

    # ── SECTION 2: Longitudinal Charts ────────────────────────────────────────
    st.markdown('<div class="sec-hdr">Longitudinal Charts</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    for col, fn, label in [
        (col1, lambda: chart_bmi(selected_pid8, obs),         "BMI"),
        (col2, lambda: chart_conditions(selected_pid8, cond), "condition"),
        (col3, lambda: chart_medications(selected_pid8, meds),"medication"),
    ]:
        with col:
            fig = fn()
            if fig:
                st.pyplot(fig)
                plt.close(fig)
            else:
                st.caption(f"No {label} data found for this patient.")

    # ── SECTION 3: Condition Timeline ─────────────────────────────────────────
    st.markdown('<div class="sec-hdr">Condition Timeline</div>', unsafe_allow_html=True)

    if waves:
        tdf    = build_timeline(selected_pid8, cond, meds, waves)
        styled = tdf.style.background_gradient(
            subset=["Conditions Active", "New This Wave", "Medications Active"],
            cmap="Blues",
        )
        st.dataframe(styled, use_container_width=True, hide_index=True)
    else:
        st.caption("No wave data found for this patient.")

    # ── SECTION 4: Knowledge Graph Subgraph ───────────────────────────────────
    st.markdown('<div class="sec-hdr">Knowledge Graph Subgraph</div>',
                unsafe_allow_html=True)

    if G is None:
        st.warning(
            "Knowledge graph not found at `data/knowledge_graph.gpickle`. "
            "Run `python src/graph_builder.py` first."
        )
    elif not _PYVIS_OK:
        st.warning("Install pyvis to enable graph visualization: `pip install pyvis`")
    else:
        with st.spinner("Generating subgraph..."):
            html = build_pyvis_html(selected_pid8, G)

        if html:
            components.html(html, height=470, scrolling=False)
            st.markdown(
                '<div style="font-size:12px; margin-top:6px; color:#555;">'
                '<span style="color:#1a2744; font-weight:bold;">&#9733; Patient</span>&nbsp;&nbsp;'
                '<span style="color:#2196F3;">&#9679; Wave</span>&nbsp;&nbsp;'
                '<span style="color:#4CAF50;">&#9679; Condition</span>&nbsp;&nbsp;'
                '<span style="color:#FF9800;">&#9632; Observation</span>&nbsp;&nbsp;'
                '<span style="color:#F44336;">&#9650; Medication</span>'
                '</div>',
                unsafe_allow_html=True,
            )
        else:
            st.info(f"Patient `{selected_pid8}` not found in the knowledge graph.")


if __name__ == "__main__":
    main()
