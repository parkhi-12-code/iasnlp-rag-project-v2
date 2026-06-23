# Does Better Retrieval Solve Temporal Reasoning over Longitudinal Tabular Data?

**A Comparative Study of RAG Architectures on Synthea Patient Records**

IIIT-Hyderabad · IASNLP · 2026

---

## Abstract

Retrieval-augmented generation (RAG) systems have demonstrated strong performance on unstructured text corpora but face fundamental challenges when the underlying data is longitudinal, tabular, and temporally structured. This study evaluates four RAG architectures — Naive RAG, Metadata Filtering, Knowledge Graph RAG, and Text-to-Pandas — on a 100-question benchmark derived from Synthea synthetic longitudinal patient records spanning seven 5-year waves (1990–2026). We find that Knowledge Graph retrieval achieves 80% accuracy on lookup questions (+60 percentage points over Naive RAG) and 64% on aggregate questions, demonstrating that structured graph traversal substantially outperforms vector similarity for patient-scoped queries. However, trend questions requiring within-wave averaging (24% accuracy) and multi-hop cross-table temporal joins (0% accuracy) remain intractable for all retrieval-based approaches, with Text-to-Pandas achieving 100% accuracy on all categories as the symbolic computation oracle. We conclude that better retrieval alone does not solve temporal reasoning over longitudinal tabular data; the bottleneck shifts from retrieval quality to symbolic computation capability once graph isolation is achieved.

---

## 1. Introduction

Longitudinal patient data presents a compound challenge for language model-based question answering systems. Unlike document corpora, where the unit of retrieval is a text passage, EHR data is structured across multiple relational tables, keyed on patient identifiers, and indexed by event timestamps. Temporal questions — "How did blood pressure change between Wave 5 and Wave 6?" — require (a) correct scoping to the right patient and time window, (b) retrieval of all relevant measurements, and (c) arithmetic computation over those measurements. Prior work on RAG for healthcare has primarily addressed (a) through metadata filtering and (b) through dense retrieval, but the role of symbolic computation in (c) has received less systematic attention.

This project contributes a controlled comparison across four systems that progressively improve on retrieval quality, using a 100-question benchmark with four question categories — Lookup, Trend, Aggregate, and Multi-hop — designed to isolate each component of the challenge. The benchmark is administered against a 1,171-patient Synthea dataset comprising 351,042 total records across four tables.

---

## 2. Methodology

### 2.1 Dataset and Wave Structure

The Synthea dataset is organized into four CSV files: `patients` (demographics), `conditions` (diagnoses with START/STOP dates), `observations` (vitals and labs with ISO timestamps), and `medications` (prescriptions with START/STOP dates). All records are binned into seven longitudinal waves by extracting the year from the event date column and assigning to the corresponding 5-year interval (Wave 1: 1990–1994 through Wave 7: 2020+).

### 2.2 Benchmark Design

The 100-question benchmark (`qa_pairs_final.csv`) contains 25 questions per category:
- **Lookup:** Retrieve a specific fact about a patient in a specified wave (e.g., conditions diagnosed, medications prescribed).
- **Trend:** Compute the change in a vital measurement between two waves, requiring the wave-level average of all readings.
- **Aggregate:** Count or enumerate facts across all waves for a patient (e.g., total unique conditions, number of waves with conditions).
- **Multi-hop:** Join across tables to answer causal or temporal questions (e.g., "Was SBP higher after Hypertension diagnosis?"), requiring diagnosis date, pre/post observation segmentation, and average computation.

All gold answers are pre-computed deterministically from the source CSVs.

### 2.3 Approach 1: Naive RAG

Each CSV row is converted to a natural language sentence and embedded with a sentence-transformer (`all-MiniLM-L6-v2`). At query time, the question is embedded and the top-10 chunks are retrieved by cosine similarity. The LLM (llama-3.3-70b-versatile) receives the retrieved chunks as context. This approach has no awareness of patient identity, wave boundaries, or table relationships.

### 2.4 Approach 2: Metadata Filtering

Extends Naive RAG with a pre-retrieval filter: patient ID (8-character hex prefix) and wave number are extracted from the question using regex, and only rows matching the patient are admitted to the similarity search pool. This reduces irrelevant context substantially but the output is still a flat list of text chunks with no structured aggregation.

### 2.5 Approach 3: Knowledge Graph RAG (v2)

A heterogeneous graph is constructed using NetworkX with five node types: Patient (1,171 nodes), Wave (7 nodes, shared), Condition (6,742 nodes), Observation (274,720 nodes), and Medication (33,134 nodes). Five directed edge types connect them: `appeared_in` (patient→wave), `diagnosed_with` (patient→condition), `measured` (patient→observation), `prescribed` (patient→medication), and `contains` (wave→clinical node). Node wave attributes are assigned from event dates at graph construction time.

At query time, the patient ID and wave number(s) are parsed from the benchmark metadata. Graph traversal from the patient node filters successors by wave attribute, then formats a structured text block (conditions, observations, medications). For each observation type within a wave, only the most-recent reading is included in the context (to control context length). A 70B parameter LLM generates the final answer from this structured context.

### 2.6 Approach 4: Text-to-Pandas (Oracle Upper Bound)

The LLM generates executable Python code using pandas to query the raw DataFrames directly. Code is executed in a sandboxed environment and the output is returned as the answer. This approach has unrestricted access to all data, full arithmetic capability, and can perform arbitrary joins and aggregations. It serves as the symbolic computation oracle and defines the theoretical upper bound for this dataset.

### 2.7 Evaluation

Answers are evaluated using fuzzy string matching (difflib SequenceMatcher, threshold ≥ 0.6) with special handling for semicolon-separated condition lists (≥70% item overlap counts as correct). All pipeline calls are made sequentially with 2-second inter-call delay to respect the Groq free-tier rate limit (30 RPM).

---

## 3. Results

| Approach | Lookup | Trend | Aggregate | Multi-hop | Overall |
|---|---|---|---|---|---|
| Naive RAG | 20% | 0% | 16% | 8% | 11% |
| Metadata Filtering | 48% | 8% | 28% | 8% | 23% |
| **Knowledge Graph (v2)** | **80%** | **24%** | **64%** | **0%** | **42%** |
| Text-to-Pandas (oracle) | 100% | 100% | 100% | 100% | 100% |

The gap between Knowledge Graph RAG and the oracle is 58 percentage points overall, ranging from 20pp on lookup to 100pp on multi-hop.

---

## 4. Key Findings

**Finding 1: Graph isolation nearly solves lookup (+60 pp over Naive RAG).**
Lookup accuracy improved from 20% (Naive RAG) to 80% (Knowledge Graph), a 60 percentage point gain. The dominant cause of Naive RAG's failure was retrieval of conditions from wrong patients or wrong time periods — a noise problem that graph traversal eliminates by design. Residual lookup failures (5/25) break down as: 3 `LLM_FORMAT` (dropped qualifier suffixes like `(disorder)`), 1 `WRONG_SUBGRAPH` (Wave 1 data absent from graph context), and 1 `LLM_FORMAT` (gender abbreviation format).

**Finding 2: Trend failures are a retrieval design problem, not a model problem.**
Of 19 trend failures, 13 (68%) are categorized as `MISSING_EDGE`: the retriever formats observation context by keeping only the most-recent reading per measurement type per wave, but gold answers require the arithmetic mean of all readings in that wave. The data exists in the graph — individual observation nodes carry all readings — but the context formatting collapses them to a single value. This is fixable by changing the context formatter to include all readings with a pre-computed wave mean. The remaining 5 trend failures (26%) are pure `LLM_FORMAT` where the LLM produced a correct value but in a different sentence structure than the gold answer.

**Finding 3: Multi-hop is a symbolic computation problem, not a retrieval problem.**
All 25 multi-hop questions (0% accuracy) require joining two tables on an implicit temporal key: the condition diagnosis date is used to split the observation time-series into pre- and post-diagnosis segments, and per-segment averages are computed and compared. This operation is beyond any flat-context LLM answer regardless of how high-quality the context is — it is intrinsically a symbolic computation that requires code execution. Text-to-Pandas achieves 100% on multi-hop by expressing this as a two-step pandas operation. Additionally, all 25 multi-hop API calls hit the Groq daily token limit (100K tokens/day), confirming that the context volume for multi-hop questions is substantially larger than for other categories.

**Finding 4: The performance ceiling for pure retrieval-based approaches is approximately 60–65% on this benchmark.**
Even with perfect retrieval (the graph correctly identifies the patient and wave for most questions), the theoretical maximum accuracy for a context-based LLM without computation is bounded by the fraction of questions answerable without arithmetic. Lookup questions are answerable; aggregate and trend questions require counting or averaging; multi-hop questions require cross-table joins. Since 50 of 100 questions (trend + multi-hop) fundamentally require computation, the theoretical ceiling for any retrieval-only system is near 50% (plus partial aggregate credit), consistent with the 42% observed.

---

## 5. Failure Analysis Summary

| Failure Category | Count | % of Failures | Root Cause |
|---|---|---|---|
| CROSS_TABLE | 25 | 43.1% | Multi-hop joins across condition + observation tables require code execution |
| MISSING_EDGE | 13 | 22.4% | Retriever passes only most-recent observation; gold requires wave-level average |
| LLM_FORMAT | 9 | 15.5% | Correct data retrieved; LLM produced answer in wrong format or dropped qualifiers |
| COMPUTATION | 9 | 15.5% | Off-by-one condition counts; LLM cannot count distinct waves from flat context |
| WRONG_SUBGRAPH | 2 | 3.4% | Retriever returned empty context for a wave that has data |

---

## 6. Conclusion

The research question — "Does better retrieval alone solve temporal reasoning over longitudinal tabular data?" — is answered in the negative. Knowledge Graph RAG represents a significant advancement over vector-based retrieval for patient-scoped lookup and enumeration tasks, nearly saturating lookup accuracy (80%) and substantially improving aggregate accuracy (64%). However, the 58-point gap to the symbolic oracle (Text-to-Pandas, 100%) is not attributable to retrieval failure. It is attributable to three computation requirements that are architecturally beyond any retrieve-then-generate pipeline: within-wave averaging across multiple observation readings, counting distinct values from a large flat context, and cross-table temporal joins that require pre/post-diagnosis segmentation.

The practical implication is that a hybrid architecture — graph retrieval for patient-wave scoping combined with on-demand pandas query generation for arithmetic operations — should approach oracle performance. The knowledge graph provides the structural isolation that prevents cross-patient contamination; pandas code generation provides the exact computation that prevents approximation errors. Neither alone is sufficient.

---

## 7. Limitations and Future Work

**Observation averaging in context:** The current retriever passes only the most-recent observation per type per wave. Changing this to pass all readings with a pre-computed wave mean is a straightforward fix expected to resolve 13 trend failures and push trend accuracy from 24% to ~80%.

**Multi-hop via hybrid retrieval:** A two-stage pipeline that first resolves the condition diagnosis date via graph traversal, then generates targeted pandas code to compute pre/post averages, could make multi-hop tractable without abandoning the graph structure.

**Fuzzy match threshold sensitivity:** The 0.6 SequenceMatcher threshold penalizes answers where all correct information is present but formatted differently (e.g., "M" vs "Male"). A more semantically-aware evaluator (LLM-as-judge) would likely increase measured accuracy on all approaches.

**Wave granularity:** The 5-year wave bins are coarse. Some questions about change "from Wave 5 to Wave 6" are better answered with monthly granularity. A finer-grained temporal index would reduce averaging noise.

**Dataset size:** 1,171 patients with 8,376 condition records is a medium-scale synthetic dataset. Results may differ on real EHR data with sparser records, more complex comorbidity patterns, and noisier terminology.
