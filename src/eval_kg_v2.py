import pandas as pd
import time
import difflib
import os
import sys

sys.path.insert(0, "src")

from graph_retriever import load_graph
from graph_rag_pipeline import run_pipeline
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Load benchmark questions
qa_df = pd.read_csv("benchmark/qa_pairs_v2.csv")
print(f"Loaded {len(qa_df)} questions")

# Load knowledge graph once
print("Loading knowledge graph...")
G = load_graph()
print("Graph loaded. Starting evaluation...")

results = []

for i, row in qa_df.iterrows():
    question = row["question"]
    gold = str(row["answer"])
    qid = row["question_id"]
    pid = str(row["patient_id"])

    # Add patient ID to the question
    augmented = f"Patient ID: {pid}. {question}"

    # Get answer from Graph RAG
    try:
        result = run_pipeline(augmented)
        if isinstance(result, dict):
            predicted = str(
                result.get('answer',
                result.get('response',
                result.get('text', str(result)))))
        else:
            predicted = str(result)
    except Exception as e:
        predicted = f"ERROR: {e}"
        time.sleep(10)

    # Calculate similarity
    sim = difflib.SequenceMatcher(
        None,
        predicted.lower(),
        gold.lower()
    ).ratio()

    passed = sim >= 0.6

    print(
        f"[{i+1:02d}/50] "
        f"{qid} | "
        f"{'PASS' if passed else 'FAIL'} | "
        f"sim={sim:.2f}"
    )

    results.append({
        "question_id": qid,
        "category": row["category"],
        "question": question,
        "gold_answer": gold,
        "predicted": predicted,
        "similarity": round(sim, 3),
        "passed": passed
    })

    time.sleep(3)

# Save results
df = pd.DataFrame(results)
df.to_csv("benchmark/kg_v2_results.csv", index=False)

# Calculate scores
agg = df[df["category"] == "aggregate"]["passed"].sum()
mh = df[df["category"] == "multi-hop"]["passed"].sum()
total = df["passed"].sum()

print("\n" + "=" * 40)
print("KNOWLEDGE GRAPH V2 RESULTS")
print("=" * 40)
print(f"AGGREGATE: {agg}/25 ({agg * 4}%)")
print(f"MULTI-HOP: {mh}/25 ({mh * 4}%)")
print(f"OVERALL:   {total}/50 ({total * 2}%)")


