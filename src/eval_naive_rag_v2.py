import pandas as pd
import numpy as np
import time, difflib, os
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY)

DATA_DIR = "data/raw"
patients_df = pd.read_csv(f"{DATA_DIR}/patients.csv")
conditions_df = pd.read_csv(f"{DATA_DIR}/conditions.csv")
observations_df = pd.read_csv(f"{DATA_DIR}/observations.csv")
medications_df = pd.read_csv(f"{DATA_DIR}/medications.csv")
qa_df = pd.read_csv("benchmark/qa_pairs_v2.csv")

def assign_wave(year):
    if year <= 1994: return 1
    elif year <= 1999: return 2
    elif year <= 2004: return 3
    elif year <= 2009: return 4
    elif year <= 2014: return 5
    elif year <= 2019: return 6
    else: return 7

print("Building text chunks...")
chunks = []

for _, r in conditions_df.iterrows():
    try:
        w = assign_wave(pd.to_datetime(r['START']).year)
        chunks.append(
            f"Patient {str(r['PATIENT'])[:8]}, "
            f"condition: {r['DESCRIPTION']}, "
            f"start: {r['START']}, wave: {w}")
    except: pass

obs_sample = observations_df.sample(
    min(40000, len(observations_df)), random_state=42)
for _, r in obs_sample.iterrows():
    try:
        w = assign_wave(pd.to_datetime(r['DATE']).year)
        chunks.append(
            f"Patient {str(r['PATIENT'])[:8]}, "
            f"observation: {r['DESCRIPTION']}, "
            f"value: {r['VALUE']} {r.get('UNITS','')}, "
            f"date: {r['DATE']}, wave: {w}")
    except: pass

for _, r in medications_df.iterrows():
    try:
        w = assign_wave(pd.to_datetime(r['START']).year)
        chunks.append(
            f"Patient {str(r['PATIENT'])[:8]}, "
            f"medication: {r['DESCRIPTION']}, "
            f"start: {r['START']}, wave: {w}")
    except: pass

for _, r in patients_df.iterrows():
    chunks.append(
        f"Patient {str(r['Id'])[:8]}, "
        f"name: {r['FIRST']} {r['LAST']}, "
        f"born: {r['BIRTHDATE']}, "
        f"gender: {r['GENDER']}")

print(f"Total chunks: {len(chunks)}")
print("Embedding — takes ~5 minutes...")

model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(chunks, batch_size=256, show_progress_bar=True)
embeddings = np.array(embeddings)
print("Embeddings done. Starting evaluation...")

results = []
for i, row in qa_df.iterrows():
    question = row['question']
    gold = str(row['answer'])
    qid = row['question_id']

    q_emb = model.encode([question])
    sims = cosine_similarity(q_emb, embeddings)[0]
    top_idx = np.argsort(sims)[-10:][::-1]
    context = "\n".join([chunks[j] for j in top_idx])

    prompt = (
        f"Answer this question using ONLY the context below. "
        f"Be concise and specific.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\nAnswer:")

    try:
        resp = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=150
        )
        predicted = resp.choices[0].message.content.strip()
    except Exception as e:
        predicted = f"ERROR: {e}"
        time.sleep(15)

    sim = difflib.SequenceMatcher(
        None, predicted.lower(), gold.lower()).ratio()
    passed = sim >= 0.6

    print(f"[{i+1:02d}/50] {qid} | {'PASS' if passed else 'FAIL'} | sim={sim:.2f}")

    results.append({
        'question_id': qid,
        'category': row['category'],
        'question': question,
        'gold_answer': gold,
        'predicted': predicted,
        'similarity': round(sim, 3),
        'passed': passed
    })
    time.sleep(3)

df = pd.DataFrame(results)
df.to_csv('benchmark/naive_rag_v2_results.csv', index=False)
df[~df['passed']].to_csv('benchmark/naive_rag_v2_failures.csv', index=False)

agg = df[df['category']=='aggregate']['passed'].sum()
mh = df[df['category']=='multi-hop']['passed'].sum()
total = df['passed'].sum()

print(f"\n{'='*40}")
print(f"NAIVE RAG V2 RESULTS")
print(f"{'='*40}")
print(f"AGGREGATE: {agg}/25 ({agg*4}%)")
print(f"MULTI-HOP: {mh}/25 ({mh*4}%)")
print(f"OVERALL:   {total}/50 ({total*2}%)")

