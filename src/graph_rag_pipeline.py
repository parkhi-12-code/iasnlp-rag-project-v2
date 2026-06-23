"""
graph_rag_pipeline.py
Full pipeline: question -> graph context -> LLM answer.

Extraction model : llama-3.1-8b-instant  (fast, free tier)
Answer model     : llama-3.3-70b-versatile (high quality)
"""

import os
import re
import json
import sys
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq

sys.path.insert(0, str(Path(__file__).parent))
from graph_retriever import load_graph, format_subgraph_context

load_dotenv(Path(__file__).parent.parent / ".env")

_graph = None
_client = None

WAVE_KEYWORDS = {
    "wave 1": 1, "wave1": 1,
    "wave 2": 2, "wave2": 2,
    "wave 3": 3, "wave3": 3,
    "wave 4": 4, "wave4": 4,
    "wave 5": 5, "wave5": 5,
    "wave 6": 6, "wave6": 6,
    "wave 7": 7, "wave7": 7,
}


def _get_graph():
    global _graph
    if _graph is None:
        _graph = load_graph()
    return _graph


def _get_client() -> Groq:
    global _client
    if _client is None:
        key = os.environ.get("GROQ_API_KEY", "")
        if not key or key == "your_key_here":
            raise RuntimeError("Set GROQ_API_KEY in your .env file before running the pipeline.")
        _client = Groq(api_key=key)
    return _client


def parse_wave_reference(wave_ref: str) -> list[int]:
    """Convert 'Wave 5' or 'Wave 3 and Wave 5' to [5] or [3, 5]."""
    text = str(wave_ref).lower()
    waves = []
    for kw, num in WAVE_KEYWORDS.items():
        if kw in text and num not in waves:
            waves.append(num)
    return sorted(waves)


def _extract_params_regex(question: str) -> dict:
    """Fast regex extraction – no LLM call."""
    q = question.lower()

    # Patient ID: 8 lowercase hex chars surrounded by word boundaries
    id_match = re.search(r'\b([0-9a-f]{8})\b', q)
    patient_id = id_match.group(1) if id_match else None

    # Wave numbers
    waves = []
    for kw, num in WAVE_KEYWORDS.items():
        if kw in q and num not in waves:
            waves.append(num)
    waves.sort()

    # Category heuristic
    if any(w in q for w in ("trend", "change over", "across wave", "compare", "between wave", "progression")):
        category = "trend"
    elif any(w in q for w in ("how many", "count", "total", "average", "mean", "most common", "percentage")):
        category = "aggregate"
    elif any(w in q for w in ("which patient", "who had", "across all patient", "multi", "related to", "both")):
        category = "multi_hop"
    else:
        category = "lookup"

    mode = "comparison" if len(waves) >= 2 else ("full_history" if not waves else "specific")

    return {"patient_id": patient_id, "waves": waves, "category": category, "mode": mode}


_EXTRACT_SYSTEM = (
    "You are a medical query parser. Extract structured info from the question. "
    "Return ONLY a JSON object, no prose."
)

_EXTRACT_USER = """\
Question: {question}

Return JSON with exactly these keys:
- patient_id  : 8-char hex string or null
- waves       : list of integers (1-7), empty if all/unclear
- category    : one of "lookup" | "trend" | "aggregate" | "multi_hop"
- mode        : one of "specific" | "comparison" | "full_history"

Wave map: 1=1990-1994, 2=1995-1999, 3=2000-2004, 4=2005-2009, 5=2010-2014, 6=2015-2019, 7=2020+"""


def _extract_params_llm(question: str) -> dict | None:
    """LLM-based extraction fallback."""
    try:
        client = _get_client()
        resp = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": _EXTRACT_SYSTEM},
                {"role": "user", "content": _EXTRACT_USER.format(question=question)},
            ],
            temperature=0,
            max_tokens=150,
        )
        raw = resp.choices[0].message.content.strip()
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            return json.loads(m.group())
    except Exception as e:
        print(f"  [WARN] LLM extraction failed: {e}", flush=True)
    return None


_ANSWER_PROMPT = """\
You are a medical AI assistant. Answer the question using ONLY the patient data below.

{context}

Question: {question}

Rules:
- Be precise and concise, no preamble.
- When listing multiple items, separate them with a semicolon (e.g., "Condition A; Condition B").
- Include value and units for numeric answers.
- If the data does not contain the answer, say "Not found in records."
- Do not infer beyond what is explicitly stated.

Answer:"""


def run_pipeline(
    question: str,
    patient_id: str | None = None,
    wave_reference: str | None = None,
    category: str | None = None,
) -> dict:
    """
    Run the full graph-RAG pipeline.

    When called from the eval harness, pass patient_id / wave_reference / category
    directly from the benchmark row to skip the LLM extraction step.

    Returns a dict: {answer, patient_id, waves, category, context}
    """
    # --- Parameter resolution ---
    if patient_id and wave_reference is not None:
        # Use benchmark-provided metadata directly
        waves = parse_wave_reference(wave_reference)
        cat = category or "lookup"
        mode = "comparison" if len(waves) >= 2 else ("full_history" if not waves else "specific")
        params = {"patient_id": patient_id, "waves": waves, "category": cat, "mode": mode}
    else:
        params = _extract_params_regex(question)
        if not params["patient_id"]:
            llm_params = _extract_params_llm(question)
            if llm_params:
                params.update(llm_params)

    pid = params.get("patient_id")
    waves = params.get("waves", [])
    cat = params.get("category", "lookup")
    mode = params.get("mode", "specific")

    if not pid:
        return {
            "answer": "Not found in records.",
            "patient_id": None,
            "waves": waves,
            "category": cat,
            "context": "",
        }

    # --- Graph retrieval ---
    G = _get_graph()
    context = format_subgraph_context(G, pid, waves or None, mode)

    # --- LLM answer generation ---
    try:
        client = _get_client()
        resp = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{
                "role": "user",
                "content": _ANSWER_PROMPT.format(context=context, question=question),
            }],
            temperature=0,
            max_tokens=300,
        )
        answer = resp.choices[0].message.content.strip()
    except Exception as e:
        answer = f"ERROR: {e}"

    return {
        "answer": answer,
        "patient_id": pid,
        "waves": waves,
        "category": cat,
        "context": context,
    }


if __name__ == "__main__":
    test_cases = [
        {
            "question": "What condition(s) were diagnosed in Wave 5 for patient 9358d6df?",
            "patient_id": "9358d6df",
            "wave_reference": "Wave 5",
            "category": "lookup",
        },
        {
            "question": "What medications was patient c868a615 prescribed in Wave 3?",
            "patient_id": "c868a615",
            "wave_reference": "Wave 3",
            "category": "lookup",
        },
    ]
    for tc in test_cases:
        print(f"\nQ: {tc['question']}")
        result = run_pipeline(**tc)
        print(f"A: {result['answer']}")
        print(f"   [patient={result['patient_id']}, waves={result['waves']}, cat={result['category']}]")
