# IASNLP RAG Project v2 — Knowledge Graph-Based Retrieval over Longitudinal Patient Data

**IIIT-Hyderabad · Information Access and Search (IASNLP) · 2026**

---

## Research Question

> **Does better retrieval alone solve temporal reasoning over longitudinal tabular data, or does it require symbolic computation?**

This project benchmarks four retrieval-augmented generation (RAG) approaches on a dataset of Synthea longitudinal patient records, focusing on whether graph-based retrieval improves over vector-similarity methods for temporal and multi-hop questions.

---

## Dataset

**Source:** [Synthea](https://synthetichealth.github.io/synthea/) synthetic patient records simulating longitudinal EHR data.

| File | Rows | Description |
|---|---|---|
| `data/raw/patients.csv` | 1,171 | Demographics, birthdate, location |
| `data/raw/conditions.csv` | 8,376 | Diagnoses with START/STOP dates |
| `data/raw/observations.csv` | 299,697 | Vitals and lab readings with timestamps |
| `data/raw/medications.csv` | 42,989 | Prescriptions with START/STOP dates |

**Wave Structure:** Data is binned into 5-year longitudinal waves using the event date column of each table.

| Wave | Years | Wave | Years |
|---|---|---|---|
| Wave 1 | 1990–1994 | Wave 5 | 2010–2014 |
| Wave 2 | 1995–1999 | Wave 6 | 2015–2019 |
| Wave 3 | 2000–2004 | Wave 7 | 2020+ |
| Wave 4 | 2005–2009 | | |

**Benchmark:** `benchmark/qa_pairs_final.csv` — 100 questions (25 per category), verified gold answers, do not modify.

---

## Approaches

### 1. Naive RAG (Baseline)
Text chunks from each CSV row are embedded with a sentence-transformer model and stored in a vector index. At query time, top-k chunks are retrieved by cosine similarity and passed to the LLM. Has no understanding of wave structure, patient identity, or temporal ordering. Temporal questions fail almost entirely because the relevant chunks may not rank highly by semantic similarity alone.

### 2. Metadata Filtering
Extends Naive RAG with a pre-filtering step: patient ID and wave number are extracted from the question and used to hard-filter the chunk pool before similarity search. This improves lookup accuracy substantially but still passes flat text to the LLM, making averaging, counting, and cross-table joins difficult.

### 3. Knowledge Graph RAG (v2 — this project)
Builds a NetworkX heterogeneous graph with five node types (Patient, Wave, Condition, Observation, Medication) and five edge types (appeared_in, diagnosed_with, measured, prescribed, contains). At query time, patient ID and wave(s) are extracted, the relevant subgraph is traversed, and a structured text context is passed to the LLM (llama-3.3-70b-versatile via Groq). Graph traversal guarantees patient-wave isolation and eliminates irrelevant context. Lookup and aggregate accuracy improve dramatically; however, trend averaging and multi-hop cross-table joins remain beyond the context format.

### 4. Text-to-Pandas (Oracle Upper Bound)
The LLM generates pandas code to query the raw DataFrames directly. Provides exact symbolic computation — averages, counts, joins, pre/post-diagnosis comparisons — with no retrieval step. Achieves 100% on all categories and serves as the upper bound for what is achievable with the available data.

---

## Results

| Approach | Lookup | Trend | Aggregate | Multi-hop | Overall |
|---|---|---|---|---|---|
| Naive RAG | 20% | 0% | 16% | 8% | 11% |
| Metadata Filtering | 48% | 8% | 28% | 8% | 23% |
| **Knowledge Graph (v2)** | **80%** | **24%** | **64%** | **0%** | **42%** |
| Text-to-Pandas (oracle) | 100% | 100% | 100% | 100% | 100% |

Full data: [`benchmark/comparison_table.csv`](benchmark/comparison_table.csv)  
Comparison chart: [`benchmark/charts/comparison_chart.png`](benchmark/charts/comparison_chart.png)

---

## Key Findings

- **Graph retrieval dominates lookup (+60 pp over Naive RAG, +32 pp over Metadata Filtering):** Guaranteed patient-wave subgraph isolation eliminates irrelevant context, making simple lookup nearly solved.
- **Trend questions expose an averaging gap:** The retriever passes only the most-recent observation per measurement type per wave. Gold answers require the average of all readings. 13 of 19 trend failures are `MISSING_EDGE` — the data exists but is not aggregated in the context.
- **Multi-hop is a computation problem, not a retrieval problem:** All 25 multi-hop failures are `CROSS_TABLE` — they require joining condition diagnosis dates with observation time-series and computing pre/post averages, which the LLM cannot do from a flat text context regardless of how good the retrieval is. Text-to-Pandas solves all 25 trivially.
- **The research question is answered:** Better retrieval alone (graph vs. vector) is insufficient for temporal reasoning over longitudinal data. The remaining 58% gap to the oracle upper bound is attributable primarily to symbolic computation requirements (averaging, counting, cross-table joins), not retrieval quality.

---

## Patient Deep Dive

Three patients were selected for longitudinal cohort analysis (see [`benchmark/patient_deep_dive_results.json`](benchmark/patient_deep_dive_results.json)):

| Role | Patient ID | Waves | Conditions | Key Feature |
|---|---|---|---|---|
| Star | `3f336702` | 7 | 22 unique | Coronary Heart Disease persistent across 6 waves; 7× condition burden increase |
| Contrast | `1930a1b6` | 7 | 1 | Most stable patient; maximum wave coverage with minimal disease |
| Comorbid | `3b95da79` | 7 | 14 | Both Hypertension and Diabetes from Wave 2; monotonically increasing burden |

Clinical narrative summaries: [`benchmark/cohort_summary.txt`](benchmark/cohort_summary.txt)  
Charts: [`benchmark/charts/`](benchmark/charts/)

---

## How to Run Everything

```powershell
# 0. Set up environment (once)
cd E:\iasnlp-rag-project-v2
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 1. Add your Groq API key to .env
#    GROQ_API_KEY=gsk_...

# 2. Build the knowledge graph (~2 min, no API key needed)
python src\graph_builder.py

# 3. Verify graph + retriever (spot-check)
python src\graph_retriever.py 9358d6df 5

# 4. Test a single pipeline question end-to-end
python src\graph_rag_pipeline.py

# 5. Run full evaluation (~7 min, uses Groq API)
python src\eval_harness_v2.py

# 6. Patient cohort deep dive (no API key needed)
python src\patient_deep_dive.py

# 7. Open the notebook for visual exploration
jupyter notebook notebooks\day1_explore.ipynb
```

**Output files produced:**

| File | Description |
|---|---|
| `data/knowledge_graph.gpickle` | Serialized NetworkX graph |
| `benchmark/graph_rag_results.csv` | Per-question results with predicted answers |
| `benchmark/graph_rag_failure_log.csv` | Failures only |
| `benchmark/failure_analysis.csv` | Failures with `failure_bucket` categorization |
| `benchmark/comparison_table.csv` | All 4 approaches × 5 categories |
| `benchmark/patient_deep_dive_results.json` | 15+5+5 question analysis |
| `benchmark/cohort_summary.txt` | Clinical narratives |
| `benchmark/charts/` | All 5 matplotlib charts |
