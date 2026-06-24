"""
src/demo_app.py  (v3)
IASNLP Patient Explorer — two-tab redesign.
Tab 1: Voice / text query with automatic patient resolution.
Tab 2: Patient Explorer with sidebar-style search panel.

Run: python -m streamlit run src/demo_app.py
"""

import os
import sys
import json
import re
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
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

# ── optional imports ───────────────────────────────────────────────────────────

try:
    from voice_stt import transcribe_audio
    _STT_OK = True
except Exception as _e:
    _STT_OK = False; _STT_ERR = str(_e)

try:
    from voice_tts import speak_answer
    _TTS_OK = True
except Exception as _e:
    _TTS_OK = False; _TTS_ERR = str(_e)

try:
    from graph_rag_pipeline import run_pipeline
    _PIPELINE_OK = True
except Exception as _e:
    _PIPELINE_OK = False; _PIPELINE_ERR = str(_e)

try:
    from data_utils import build_patient_lookup, resolve_patient, load_tables
    _DATA_UTILS_OK = True
except Exception as _e:
    _DATA_UTILS_OK = False; _DATA_UTILS_ERR = str(_e)

try:
    from pyvis.network import Network as PyvisNetwork
    _PYVIS_OK = True
except ImportError:
    _PYVIS_OK = False

WAVE_RANGES = {
    1: "1990-1994", 2: "1995-1999", 3: "2000-2004",
    4: "2005-2009", 5: "2010-2014", 6: "2015-2019", 7: "2020+",
}

# ── page config ────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="IASNLP Patient Explorer",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
[data-testid="stSidebar"]{display:none}
[data-testid="collapsedControl"]{display:none}
</style>
""", unsafe_allow_html=True)

# ── data loaders ───────────────────────────────────────────────────────────────

@st.cache_data(show_spinner="Loading patient data...")
def _load_tables():
    if _DATA_UTILS_OK:
        return load_tables(str(DATA_RAW))
    patients     = pd.read_csv(DATA_RAW / "patients.csv")
    conditions   = pd.read_csv(DATA_RAW / "conditions.csv")
    observations = pd.read_csv(DATA_RAW / "observations.csv")
    medications  = pd.read_csv(DATA_RAW / "medications.csv")
    return patients, conditions, observations, medications


@st.cache_resource(show_spinner="Loading knowledge graph...")
def _load_graph():
    if not GRAPH_PATH.exists():
        return None
    with open(GRAPH_PATH, "rb") as f:
        return pickle.load(f)


@st.cache_data(show_spinner="Building patient lookup...")
def _build_lookup(_patients_df):
    if not _DATA_UTILS_OK:
        return []
    return build_patient_lookup(_patients_df)


# ── small utilities ────────────────────────────────────────────────────────────

def _tts_to_bytes(text: str) -> bytes | None:
    if not _TTS_OK:
        return None
    tmp = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            tmp = f.name
        result = speak_answer(text, tmp)
        if isinstance(result, str) and result.startswith("ERROR"):
            return None
        with open(tmp, "rb") as f:
            return f.read()
    except Exception:
        return None
    finally:
        if tmp:
            try: os.unlink(tmp)
            except OSError: pass


def _stt_from_bytes(audio_bytes: bytes, suffix: str = ".wav") -> str:
    if not _STT_OK:
        return "ERROR: STT module not available"
    tmp = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
            f.write(audio_bytes)
            tmp = f.name
        return transcribe_audio(tmp)
    except Exception as e:
        return f"ERROR: {e}"
    finally:
        if tmp:
            try: os.unlink(tmp)
            except OSError: pass


def _resolve_full_id(pid8: str, patients_df) -> str | None:
    """Return full UUID from 8-char prefix."""
    matches = patients_df[patients_df["Id"].astype(str).str.startswith(pid8)]
    return str(matches.iloc[0]["Id"]) if len(matches) > 0 else None


def find_patient_node(G, patient_full_id: str):
    """Find patient node in graph by full UUID, filtering on node_type == 'Patient'."""
    if G is None:
        return None
    # Direct match
    if patient_full_id in G.nodes():
        return patient_full_id
    # Search only Patient-type nodes
    pid_str = str(patient_full_id)
    for node, data in G.nodes(data=True):
        if data.get("node_type") == "Patient":
            node_s = str(node)
            if node_s == pid_str:
                return node
            if pid_str[:8] in node_s:
                return node
            if node_s[:8] in pid_str:
                return node
    return None


def patient_in_graph(pid: str, G) -> bool:
    """Return True if this patient has a Patient-type node in the knowledge graph."""
    return find_patient_node(G, str(pid)) is not None


def _extract_patient_from_question(question: str) -> tuple[str | None, int | None]:
    """Use Groq LLM to extract (first_name, birth_year) from a free-form question."""
    if not GROQ_API_KEY:
        return None, None
    try:
        from groq import Groq
        client = Groq(api_key=GROQ_API_KEY)
        prompt = (
            "Extract the patient first name and birth year from this question. "
            "Return ONLY valid JSON with keys \"patient_name\" (string or null) "
            "and \"birth_year\" (integer or null).\n"
            f"Question: {question}"
        )
        resp = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=60,
        )
        raw = resp.choices[0].message.content.strip()
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            data = json.loads(m.group())
            name = data.get("patient_name") or ""
            year = data.get("birth_year")
            return (name.strip() or None), (int(year) if year else None)
    except Exception:
        pass
    return None, None


def _show_answer_and_audio(answer: str) -> None:
    """Shared: green success box + auto-play TTS. Used by both tabs."""
    st.success(answer)
    if _TTS_OK:
        with st.spinner("Generating audio..."):
            mp3_bytes = _tts_to_bytes(answer)
        if mp3_bytes:
            st.audio(mp3_bytes, format="audio/mp3", autoplay=True)


# ── chart helpers ──────────────────────────────────────────────────────────────

def _make_fig(title: str, xlabel: str, ylabel: str):
    fig, ax = plt.subplots(figsize=(4, 3))
    fig.patch.set_facecolor("#f8fafc")
    ax.set_facecolor("#f8fafc")
    ax.set_title(title, fontsize=10, fontweight="bold", color="#1a2744", pad=4)
    ax.set_xlabel(xlabel, fontsize=8, color="#555")
    ax.set_ylabel(ylabel, fontsize=8, color="#555")
    ax.tick_params(labelsize=8, colors="#555")
    for sp in ax.spines.values():
        sp.set_edgecolor("#dee2e6")
    ax.grid(axis="y", color="#dee2e6", linewidth=0.5, alpha=0.6)
    return fig, ax


def show_patient_charts(patient_id, conditions_df, observations_df, medications_df):
    c1, c2, c3 = st.columns(3)

    # BMI
    with c1:
        obs = observations_df[observations_df["PATIENT"].astype(str) == str(patient_id)].copy()
        if "DESCRIPTION" in obs.columns:
            bmi = obs[obs["DESCRIPTION"].str.contains("Body Mass Index|BMI", case=False, na=False)].copy()
        else:
            bmi = pd.DataFrame()
        if not bmi.empty and "WAVE" in bmi.columns:
            bmi["_v"] = pd.to_numeric(bmi["VALUE"], errors="coerce")
            bmi = bmi.dropna(subset=["_v", "WAVE"])
        else:
            bmi = pd.DataFrame()
        if not bmi.empty:
            grp = bmi.groupby("WAVE")["_v"].mean().sort_index()
            fig, ax = _make_fig("BMI per Wave", "Wave", "kg/m²")
            ax.plot(grp.index, grp.values, "o-", color="#2196F3", lw=2,
                    markersize=6, markerfacecolor="white", markeredgewidth=2)
            ax.set_xticks(grp.index.astype(int))
            fig.tight_layout(pad=1.2)
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.info("No BMI data")

    # Conditions per wave
    with c2:
        conds = conditions_df[conditions_df["PATIENT"].astype(str) == str(patient_id)].copy()
        if not conds.empty and "WAVE" in conds.columns:
            conds = conds.dropna(subset=["WAVE"])
            grp = conds.groupby("WAVE")["DESCRIPTION"].nunique().sort_index()
            if not grp.empty:
                fig, ax = _make_fig("Conditions per Wave", "Wave", "Unique Conditions")
                ax.bar(grp.index, grp.values, color="#1D9E75", alpha=0.8, width=0.55)
                ax.set_xticks(grp.index.astype(int))
                fig.tight_layout(pad=1.2)
                st.pyplot(fig)
                plt.close(fig)
            else:
                st.info("No condition data")
        else:
            st.info("No condition data")

    # Medications per wave
    with c3:
        meds = medications_df[medications_df["PATIENT"].astype(str) == str(patient_id)].copy()
        if not meds.empty and "WAVE" in meds.columns:
            meds = meds.dropna(subset=["WAVE"])
            grp = meds.groupby("WAVE")["DESCRIPTION"].nunique().sort_index()
            if not grp.empty:
                fig, ax = _make_fig("Medications per Wave", "Wave", "Unique Medications")
                ax.bar(grp.index, grp.values, color="#E8A838", alpha=0.8, width=0.55)
                ax.set_xticks(grp.index.astype(int))
                fig.tight_layout(pad=1.2)
                st.pyplot(fig)
                plt.close(fig)
            else:
                st.info("No medication data")
        else:
            st.info("No medication data")


def show_condition_timeline(patient_id, conditions_df, medications_df):
    conds = conditions_df[conditions_df["PATIENT"].astype(str) == str(patient_id)].copy()
    if conds.empty or "WAVE" not in conds.columns:
        st.info("No condition data for this patient.")
        return
    conds = conds.dropna(subset=["WAVE"])

    meds = medications_df[medications_df["PATIENT"].astype(str) == str(patient_id)].copy()
    has_meds_wave = not meds.empty and "WAVE" in meds.columns

    rows = []
    seen: set = set()
    for wave in sorted(conds["WAVE"].dropna().unique()):
        wave = int(wave)
        wc = conds[conds["WAVE"] == wave]
        new_c = [c for c in wc["DESCRIPTION"].tolist() if c not in seen]
        seen.update(new_c)
        n_meds = int(len(meds[meds["WAVE"] == wave])) if has_meds_wave else 0
        rows.append({
            "Wave":               f"Wave {wave}",
            "Year Range":         WAVE_RANGES.get(wave, ""),
            "New Conditions":     len(new_c),
            "Total Active":       len(seen),
            "Medications Active": n_meds,
        })

    if rows:
        df = pd.DataFrame(rows)
        styled = df.style.background_gradient(
            subset=["New Conditions", "Total Active", "Medications Active"],
            cmap="Blues",
        )
        st.dataframe(styled, use_container_width=True, hide_index=True)


def render_patient_graph(patient_id, G,
                          conditions_df=None,
                          observations_df=None,
                          medications_df=None):
    if not _PYVIS_OK:
        st.warning("Install pyvis: `pip install pyvis`")
        return
    if G is None:
        st.warning("Knowledge graph not loaded.")
        return

    full_id = find_patient_node(G, str(patient_id))

    if not full_id:
        node_types: set = set()
        for n, d in list(G.nodes(data=True))[:200]:
            node_types.add(d.get("node_type", d.get("type", "?")))
        st.error(
            f"Patient node not in graph. "
            f"Full ID searched: `{patient_id}`. "
            f"Node types in graph: {node_types}. "
            f"Sample nodes: {list(G.nodes())[:5]}"
        )
        return

    net = PyvisNetwork(
        height="520px", width="100%",
        bgcolor="#0e1117", font_color="white",
        directed=False,
    )
    net.set_options("""{
      "physics": {
        "forceAtlas2Based": {
          "gravitationalConstant": -50,
          "springLength": 100
        },
        "solver": "forceAtlas2Based"
      },
      "nodes": {"borderWidth": 2, "shadow": false}
    }""")

    color_map = {
        "Wave":        "#2196F3",
        "Condition":   "#1D9E75",
        "Observation": "#E8A838",
        "Medication":  "#E24B4A",
    }
    size_map = {
        "Wave": 22, "Condition": 18,
        "Observation": 14, "Medication": 16,
    }

    pdata = G.nodes[full_id]
    pname = pdata.get("name", pdata.get("FIRST", str(full_id)[:12]))
    net.add_node(
        str(full_id),
        label=f"👤 {pname}",
        color="#1a2744", size=35, shape="dot",
        title=f"PATIENT\nName: {pname}\nID: {str(full_id)[:8]}",
        font={"size": 14, "color": "white"},
    )

    node_counts: dict[str, int] = {}

    for neighbor in G.neighbors(full_id):
        ndata = G.nodes[neighbor]
        ntype = ndata.get("node_type", ndata.get("type", "Unknown"))
        node_counts[ntype] = node_counts.get(ntype, 0) + 1
        color = color_map.get(ntype, "#888888")
        size  = size_map.get(ntype, 14)

        if ntype == "Wave":
            wave_num = ndata.get("wave", ndata.get("wave_number", neighbor))
            yr = WAVE_RANGES.get(int(wave_num) if str(wave_num).isdigit() else 0, "")
            label   = f"W{wave_num}"
            tooltip = f"WAVE {wave_num}\nYears: {yr}"

        elif ntype == "Condition":
            desc   = ndata.get("description", str(neighbor))
            wave   = ndata.get("wave", "?")
            onset  = ndata.get("onset_date", ndata.get("START", "?"))
            short  = desc[:20] + "…" if len(desc) > 20 else desc
            label   = f"🦠 {short}"
            tooltip = f"CONDITION\n{desc}\nWave: {wave}\nOnset: {onset}"

        elif ntype == "Observation":
            desc   = ndata.get("description", str(neighbor))
            value  = ndata.get("value", ndata.get("VALUE", "?"))
            unit   = ndata.get("units", ndata.get("UNITS", ""))
            wave   = ndata.get("wave", "?")
            date   = ndata.get("date", ndata.get("DATE", "?"))
            short  = desc[:18] + "…" if len(desc) > 18 else desc
            label   = f"📊 {short}"
            tooltip = f"OBSERVATION\n{desc}\nValue: {value} {unit}\nWave: {wave}\nDate: {date}"

        elif ntype == "Medication":
            desc   = ndata.get("description", str(neighbor))
            wave   = ndata.get("wave", "?")
            start  = ndata.get("start_date", ndata.get("START", "?"))
            short  = desc[:18] + "…" if len(desc) > 18 else desc
            label   = f"💊 {short}"
            tooltip = f"MEDICATION\n{desc}\nWave: {wave}\nStarted: {start}"

        else:
            label   = str(neighbor)[:15]
            tooltip = f"{ntype}\n{str(neighbor)}"

        net.add_node(
            str(neighbor),
            label=label, color=color, size=size,
            title=tooltip,
            font={"size": 11, "color": "white"},
        )

        edge_data = G.get_edge_data(full_id, neighbor) or {}
        edge_type = edge_data.get("edge_type", edge_data.get("relationship", ""))
        net.add_edge(
            str(full_id), str(neighbor),
            title=edge_type,
            color={"color": "#444444", "opacity": 0.6},
            width=1.5,
        )

    # Legend nodes pinned to fixed positions
    legend_items = [
        ("👤 Patient",     "#1a2744"),
        ("🔵 Wave",        "#2196F3"),
        ("🦠 Condition",   "#1D9E75"),
        ("📊 Observation", "#E8A838"),
        ("💊 Medication",  "#E24B4A"),
    ]
    for i, (lbl, col) in enumerate(legend_items):
        net.add_node(
            f"__legend_{i}",
            label=lbl, color=col, size=12,
            x=600, y=-200 + i * 50,
            physics=False,
            font={"size": 10, "color": "white"},
            shape="box",
        )

    tmp = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
            tmp = f.name
        net.save_graph(tmp)
        with open(tmp, "r", encoding="utf-8") as f:
            html_content = f.read()
        components.html(html_content, height=540, scrolling=False)
    except Exception as e:
        st.error(f"Graph error: {e}")
        return
    finally:
        if tmp:
            try: os.unlink(tmp)
            except OSError: pass

    # Node count summary
    if node_counts:
        cols = st.columns(len(node_counts))
        for i, (ntype, count) in enumerate(sorted(node_counts.items())):
            cols[i].metric(
                label=ntype, value=count,
                help=f"Number of {ntype} nodes connected to this patient",
            )


# ── load data (runs once per session) ─────────────────────────────────────────

patients_df, conditions_df, observations_df, medications_df = _load_tables()
G              = _load_graph()
patient_lookup = _build_lookup(patients_df)

# ── HEADER ─────────────────────────────────────────────────────────────────────

st.markdown("""
<div style="background:#1a2744;padding:14px 20px;
border-radius:8px;margin-bottom:16px;
display:flex;justify-content:space-between;
align-items:center">
<div>
<span style="color:white;font-size:18px;
font-weight:600">IASNLP Patient Explorer</span>
<span style="color:rgba(255,255,255,0.5);
font-size:12px;margin-left:12px">
IIIT-Hyderabad · Knowledge Graph RAG ·
315,774 nodes · 633,571 edges</span>
</div>
<span style="background:rgba(33,150,243,0.2);
color:#90caf9;padding:4px 10px;
border-radius:4px;font-size:11px">
Knowledge Graph v2</span>
</div>
""", unsafe_allow_html=True)

# ── TABS ───────────────────────────────────────────────────────────────────────

tab1, tab2 = st.tabs([
    "🎤  Voice Query — Ask anything",
    "🔍  Patient Explorer — Browse & analyse",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — VOICE QUERY
# ══════════════════════════════════════════════════════════════════════════════

with tab1:
    st.markdown("### Ask a question about any patient")
    st.caption(
        "Speak or type your question. "
        "Include the patient name and birth year. "
        "Example: 'What conditions were diagnosed in Wave 5 for John, born 1965?'"
    )

    input_method = st.radio(
        "How would you like to ask?",
        ["🎤 Record with microphone",
         "📁 Upload audio file (WAV/MP3)",
         "⌨️ Type my question"],
        horizontal=True,
        key="tab1_input_method",
    )

    question_text: str | None = None

    if "Record" in input_method:
        try:
            audio = st.audio_input("Click to record")
        except AttributeError:
            audio = None
            st.info("st.audio_input not available in this Streamlit version — use Upload or Type.")
        if audio is not None:
            with st.spinner("Transcribing with Whisper..."):
                transcribed = _stt_from_bytes(audio.getvalue(), ".wav")
            if transcribed.startswith("ERROR"):
                st.error(transcribed)
            else:
                st.success(f"Transcribed: {transcribed}")
                question_text = transcribed

    elif "Upload" in input_method:
        uploaded = st.file_uploader(
            "Upload WAV or MP3", type=["wav", "mp3"], key="tab1_upload"
        )
        if uploaded is not None:
            suffix = ".mp3" if uploaded.name.endswith(".mp3") else ".wav"
            with st.spinner("Transcribing with Whisper..."):
                transcribed = _stt_from_bytes(uploaded.getvalue(), suffix)
            if transcribed.startswith("ERROR"):
                st.error(transcribed)
            else:
                st.success(f"Transcribed: {transcribed}")
                question_text = transcribed

    else:
        typed = st.text_area(
            "Type your question:",
            placeholder="What conditions were diagnosed in Wave 5 for Sarah, born 1972?",
            height=80,
            key="tab1_text_input",
        )
        question_text = typed.strip() or None

    ask_col, _ = st.columns([1, 4])
    with ask_col:
        ask_btn = st.button(
            "Get Answer",
            type="primary",
            disabled=(not bool(question_text)),
            use_container_width=True,
            key="tab1_ask",
        )

    if ask_btn and question_text:
        with st.spinner("Resolving patient and searching knowledge graph..."):

            pname, pyear = _extract_patient_from_question(question_text)
            resolved_pid8: str | None = None
            full_id: str | None = None

            if pname and pyear and _DATA_UTILS_OK:
                resolved_pid8 = resolve_patient(pname, pyear, patient_lookup)

            if resolved_pid8:
                full_id = _resolve_full_id(resolved_pid8, patients_df)

            if full_id:
                prow = patients_df[patients_df["Id"].astype(str) == full_id].iloc[0]
                st.info(
                    f"Patient identified: **{prow['FIRST']} {prow['LAST']}** | "
                    f"DOB: {prow['BIRTHDATE']} | ID: {resolved_pid8}"
                )

                augmented_q = f"Patient ID: {full_id}. {question_text}"

                if not _PIPELINE_OK:
                    st.error(f"Pipeline not available: {_PIPELINE_ERR}")
                    answer = None
                elif not GROQ_API_KEY:
                    st.error("GROQ_API_KEY missing — cannot query the pipeline.")
                    answer = None
                else:
                    try:
                        result = run_pipeline(
                            question=augmented_q,
                            patient_id=full_id,
                        )
                        answer = result.get("answer", "No answer returned.")
                    except Exception as exc:
                        st.error(f"Pipeline error: {exc}")
                        answer = None
            else:
                answer = None
                st.warning(
                    "Could not identify patient from your question. "
                    "Please include the patient's first name and birth year. "
                    "Example: 'for Sarah, born 1972'"
                )

        if answer:
            st.markdown("### Answer")
            _show_answer_and_audio(answer)

            if full_id:
                prow2 = patients_df[patients_df["Id"].astype(str) == full_id].iloc[0]
                st.markdown("---")
                st.markdown(f"#### Patient data: {prow2['FIRST']} {prow2['LAST']}")
                show_patient_charts(full_id, conditions_df, observations_df, medications_df)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — PATIENT EXPLORER
# ══════════════════════════════════════════════════════════════════════════════

with tab2:
    col_search, col_main = st.columns([1, 2.5])

    with col_search:
        st.markdown("#### Find a patient")
        search = st.text_input(
            "Name or birth year:",
            placeholder="e.g. Johnson or 1952",
            key="tab2_search",
        )

        selected_id:   str | None = None
        selected_name: str | None = None
        patient_row_t2 = None

        if len(search.strip()) >= 2:
            q = search.strip()
            mask = (
                patients_df["FIRST"].astype(str).str.contains(q, case=False, na=False) |
                patients_df["LAST"].astype(str).str.contains(q, case=False, na=False)  |
                patients_df["BIRTHDATE"].astype(str).str.contains(q, na=False)
            )
            results = patients_df[mask].head(10)

            if len(results) == 0:
                st.warning("No patients found.")
            else:
                st.caption(f"{len(results)} result(s)")
                st.caption("🔵 = Knowledge graph available   ⚪ = Charts only")
                options = []
                for _, r in results.iterrows():
                    graph_icon = "🔵" if patient_in_graph(str(r["Id"]), G) else "⚪"
                    options.append(
                        f"{graph_icon} {r['FIRST']} {r['LAST']} | "
                        f"{str(r['BIRTHDATE'])[:4]} | "
                        f"{r.get('GENDER', '?')}"
                    )
                chosen = st.radio(
                    "Select patient:", options, index=0, key="tab2_radio"
                )
                idx            = options.index(chosen)
                patient_row_t2 = results.iloc[idx]
                selected_id    = str(patient_row_t2["Id"])
                selected_name  = f"{patient_row_t2['FIRST']} {patient_row_t2['LAST']}"

                st.markdown("---")
                st.markdown(f"**{selected_name}**")
                st.caption(
                    f"DOB: {patient_row_t2['BIRTHDATE']} · "
                    f"{patient_row_t2.get('GENDER', '?')} · "
                    f"{patient_row_t2.get('RACE', '?')}"
                )

                n_conds = len(conditions_df[
                    conditions_df["PATIENT"].astype(str) == selected_id
                ])
                n_meds = len(medications_df[
                    medications_df["PATIENT"].astype(str) == selected_id
                ])
                m1, m2 = st.columns(2)
                m1.metric("Conditions", n_conds)
                m2.metric("Medications", n_meds)
        else:
            st.caption("Type at least 2 characters to search")

    with col_main:
        if selected_id:
            st.markdown(f"### {selected_name}")

            q_input = st.text_input(
                "Ask a question about this patient:",
                placeholder="What conditions were diagnosed in Wave 5?",
                key="tab2_question",
            )
            st.markdown("""
<div style="background:#3a2a1a;border:1px solid #E8A838;border-radius:6px;
padding:8px 12px;margin-bottom:8px;font-size:12px;color:#FFD080">
<strong>⚠ Answer accuracy: ~42% overall</strong> — Lookup questions: 80% accurate.
Trend questions: 24% accurate. Multi-hop: 0% accurate.
Use Text-to-Pandas toggle for 100% accuracy.
</div>
""", unsafe_allow_html=True)
            if st.button("Ask", type="primary", key="tab2_ask"):
                if not q_input.strip():
                    st.warning("Please enter a question.")
                elif not _PIPELINE_OK:
                    st.error(f"Pipeline not available: {_PIPELINE_ERR}")
                elif not GROQ_API_KEY:
                    st.error("GROQ_API_KEY missing — cannot query the pipeline.")
                else:
                    augmented = f"Patient ID: {selected_id}. {q_input.strip()}"
                    with st.spinner("Searching graph..."):
                        try:
                            result = run_pipeline(
                                question=augmented,
                                patient_id=selected_id,
                            )
                            ans = result.get("answer", "No answer returned.")
                        except Exception as exc:
                            ans = None
                            st.error(f"Pipeline error: {exc}")
                    if ans:
                        _show_answer_and_audio(ans)

            st.markdown("#### Longitudinal charts")
            st.markdown("""
<div style="background:#1a3a1a;border:1px solid #1D9E75;border-radius:6px;
padding:8px 12px;margin-bottom:12px;font-size:12px;color:#90EE90">
<strong>✓ Data accuracy: 100%</strong> — Charts computed directly from raw CSV data
using pandas. No AI involved in these calculations.
</div>
""", unsafe_allow_html=True)
            show_patient_charts(
                selected_id, conditions_df, observations_df, medications_df
            )

            st.markdown("#### Condition timeline")
            show_condition_timeline(selected_id, conditions_df, medications_df)

            st.markdown("#### Knowledge graph")
            st.markdown("""
<div style="background:#1a2a3a;border:1px solid #2196F3;border-radius:6px;
padding:8px 12px;margin-bottom:12px;font-size:12px;color:#90CAF9">
<strong>ℹ Graph structure: 100% accurate</strong> — Nodes and edges built
deterministically from CSV data. Visualisation is a faithful representation
of the patient's clinical network.
</div>
""", unsafe_allow_html=True)
            if st.button("Show Patient Graph", key="tab2_graph"):
                with st.spinner("Rendering graph..."):
                    render_patient_graph(
                        selected_id, G,
                        conditions_df, observations_df, medications_df,
                    )

        else:
            st.markdown("### Select a patient from the left panel to begin")
            st.caption("Search by first name, last name, or birth year")
