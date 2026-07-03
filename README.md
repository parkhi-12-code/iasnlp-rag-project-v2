# RAG for Temporal TableQA as a Conversational AI

** Best Use Case Project — IASNLP Summer School 2026**
**14th IIIT Advanced Summer School on Natural Language Processing**
**IIIT-Hyderabad · June 2026**

> *Does better retrieval alone solve temporal reasoning over longitudinal tabular data, or does it require symbolic computation?*

---

## Overview

This project benchmarks four architecturally distinct approaches to question answering over longitudinal patient health records. The dataset is Synthea synthetic EHR data spanning 30 years and 352,233 rows across four relational tables. The central finding: **retrieval quality alone is insufficient — symbolic computation is the correct architectural choice for temporal multi-hop reasoning.**

**Team:** Parkhi Yadav · Vijaya Lakshmi · Jeevitha Sasi

---

## Results at a Glance

| Approach | Lookup | Trend | Aggregate | Multi-hop | Overall |
|---|---|---|---|---|---|
| Naive RAG | 20% | 0% | 16% | 8% | **11%** |
| Metadata Filtering | 48% | 8% | 28% | 8% | **23%** |
| Knowledge Graph (v2) | 80% | 24% | 64% | 0% | **42%** |
| Text-to-Pandas | 100% | 100% | 100% | 100% | **100%** |

---

## Dataset

Source: [Synthea](https://synthetichealth.github.io/synthea/) synthetic longitudinal patient records.

| File | Rows | Contains |
|---|---|---|
| `patients.csv` | 1,171 | Demographics, birthdate, location |
| `conditions.csv` | 8,376 | Diagnoses with START/STOP dates |
| `observations.csv` | 299,697 | Vitals and lab readings with timestamps |
| `medications.csv` | 42,989 | Prescriptions with START/STOP dates |

**Wave structure** — data is binned into 5-year longitudinal waves:

| Wave | Years | Wave | Years |
|---|---|---|---|
| Wave 1 | 1990–1994 | Wave 5 | 2010–2014 |
| Wave 2 | 1995–1999 | Wave 6 | 2015–2019 |
| Wave 3 | 2000–2004 | Wave 7 | 2020+ |
| Wave 4 | 2005–2009 | | |

---

## Four Approaches

### 1. Naive RAG — Baseline (11%)
CSV rows are converted to text chunks, embedded with a sentence-transformer, and stored in a vector index. At query time, top-k chunks are retrieved by cosine similarity and passed to the LLM. No wave awareness, no patient identity, no temporal ordering. Trend questions fail almost entirely.

### 2. Metadata Filtering (23%)
Extends Naive RAG with a pre-filtering step: patient ID and wave number are extracted from the question and used to hard-filter the chunk pool before similarity search. Improves lookup substantially but flat text still reaches the LLM — averaging, counting, and cross-table joins remain broken.

### 3. Knowledge Graph RAG — v2 (42%)
Builds a NetworkX heterogeneous graph with **315,774 nodes** and **633,571 edges** across five node types (`Patient`, `Wave`, `Condition`, `Observation`, `Medication`) and five edge types (`appeared_in`, `diagnosed_with`, `measured`, `prescribed`, `contains`). At query time, patient ID and wave(s) are extracted, the relevant subgraph is traversed, and structured context is passed to the LLM (`llama-3.3-70b-versatile` via Groq). Graph traversal guarantees patient-wave isolation. Lookup improves from 20% → 80% (+60 points over Naive RAG). Multi-hop fails at 0% — the data is retrieved correctly but the LLM cannot compute cross-table arithmetic from a context string.

### 4. Text-to-Pandas — Oracle Upper Bound (100%)
The LLM generates pandas code to query the raw DataFrames directly. Provides exact symbolic computation — averages, counts, joins, pre/post-diagnosis comparisons — with no retrieval step at all. Achieves 100% across all categories. **This is the upper bound and the architectural answer.**

---

## Key Findings

**Finding 1 — Graph retrieval dominates lookup (+60 pp over Naive RAG):**
Guaranteed patient-wave subgraph isolation eliminates irrelevant context and makes simple lookup nearly solved.

**Finding 2 — Trend questions expose an averaging gap:**
13 of 19 trend failures are `MISSING_EDGE` — the data exists in the graph but is not aggregated in the context. The retriever passes the most-recent observation per wave; gold answers require the average of all readings.

**Finding 3 — Multi-hop is a computation problem, not a retrieval problem:**
All 25 multi-hop failures are `CROSS_TABLE` — they require joining condition diagnosis dates with observation time-series and computing pre/post averages. The LLM cannot do this from a flat text context regardless of retrieval quality. Text-to-Pandas solves all 25 trivially.

**Finding 4 — The research question is answered:**
Better retrieval alone is insufficient for temporal reasoning over longitudinal data. The 58-point gap between Knowledge Graph (42%) and the oracle upper bound (100%) is attributable to symbolic computation requirements — not retrieval quality.

---

## Patient Cohort Deep Dive

Three patients were selected algorithmically for longitudinal cohort analysis.

| Role | Patient ID | Waves | Conditions | Key Feature |
|---|---|---|---|---|
| ★ Star | `3f336702` | 7 | 22 unique | Coronary Heart Disease persistent across 6 waves; 7× condition burden increase |
| ◇ Contrast | `1930a1b6` | 7 | 1 | Most stable patient; maximum wave coverage with minimal disease burden |
| ⊕ Comorbid | `3b95da79` | 7 | 14 | Both Hypertension and Diabetes from Wave 2; monotonically increasing burden |

Clinical narrative summaries: `benchmark/cohort_summary.txt`

---

## How to Run

```bash
# 0. Set up environment
cd iasnlp-rag-project-v2
python -m venv venv
.\venv\Scripts\Activate.ps1      # Windows
pip install -r requirements.txt

# 1. Add Groq API key to .env
#    GROQ_API_KEY=gsk_...

# 2. Build the knowledge graph (~2 min, no API key needed)
python src/graph_builder.py

# 3. Verify graph + retriever
python src/graph_retriever.py 9358d6df 5

# 4. Test a single pipeline question end-to-end
python src/graph_rag_pipeline.py

# 5. Run full evaluation (~7 min, uses Groq API)
python src/eval_harness_v2.py

# 6. Patient cohort deep dive (no API key needed)
python src/patient_deep_dive.py

# 7. Visual exploration
jupyter notebook notebooks/day1_explore.ipynb
```

---

## Output Files

| File | Description |
|---|---|
| `data/knowledge_graph.gpickle` | Serialised NetworkX graph |
| `benchmark/qa_pairs_final.csv` | 100-question benchmark — do not modify |
| `benchmark/graph_rag_results.csv` | Per-question results with predicted answers |
| `benchmark/graph_rag_failure_log.csv` | Failures only |
| `benchmark/failure_analysis.csv` | Failures with `failure_bucket` categorisation |
| `benchmark/comparison_table.csv` | All 4 approaches × 5 categories |
| `benchmark/patient_deep_dive_results.json` | 15+5+5 question cohort analysis |
| `benchmark/cohort_summary.txt` | Clinical narratives |
| `benchmark/charts/` | All matplotlib charts |

---

## Tech Stack

`Python` · `pandas` · `NetworkX` · `sentence-transformers` · `Groq API` · `llama-3.3-70b-versatile` · `Streamlit` · `Whisper` · `gTTS` · `matplotlib`

---

## Citation / Reference

If you use this project or build on it, please reference:

```
Parkhi Yadav
RAG for Temporal TableQA as a Conversational AI
IASNLP Summer School 2026, IIIT-Hyderabad

github.com/parkhi-12-code/iasnlp-rag-project-v2
```
