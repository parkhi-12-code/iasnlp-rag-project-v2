import pandas as pd
import networkx as nx
import pickle
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data" / "raw"
OUTPUT_PATH = Path(__file__).parent.parent / "data" / "knowledge_graph.gpickle"

WAVE_RANGES = {
    1: (1990, 1994),
    2: (1995, 1999),
    3: (2000, 2004),
    4: (2005, 2009),
    5: (2010, 2014),
    6: (2015, 2019),
    7: (2020, 9999),
}


def assign_wave(date_str) -> int | None:
    if pd.isna(date_str) or str(date_str).strip() in ("", "nan"):
        return None
    try:
        year = int(str(date_str)[:4])
        for wave_num, (start, end) in WAVE_RANGES.items():
            if start <= year <= end:
                return wave_num
        return None
    except (ValueError, TypeError):
        return None


def build_graph() -> nx.DiGraph:
    print("Loading CSVs...")
    patients = pd.read_csv(DATA_DIR / "patients.csv", dtype=str)
    conditions = pd.read_csv(DATA_DIR / "conditions.csv", dtype=str)
    observations = pd.read_csv(DATA_DIR / "observations.csv", dtype=str)
    medications = pd.read_csv(DATA_DIR / "medications.csv", dtype=str)

    print(f"  patients:     {len(patients):>6} rows")
    print(f"  conditions:   {len(conditions):>6} rows")
    print(f"  observations: {len(observations):>6} rows")
    print(f"  medications:  {len(medications):>6} rows")

    G = nx.DiGraph()
    node_counts = {t: 0 for t in ("Patient", "Wave", "Condition", "Observation", "Medication")}
    edge_counts = {t: 0 for t in ("appeared_in", "diagnosed_with", "measured", "prescribed", "contains")}

    # --- Wave nodes (7 total, shared across all patients) ---
    for wave_num, (s, e) in WAVE_RANGES.items():
        yr = f"{s}-{e}" if e < 9999 else f"{s}+"
        G.add_node(f"wave_{wave_num}", node_type="Wave", wave_number=wave_num, year_range=yr)
        node_counts["Wave"] += 1

    # --- Patient nodes ---
    print("Building Patient nodes...")
    patient_id_map: dict[str, str] = {}  # short_id (8 chars) -> full UUID
    for _, row in patients.iterrows():
        full_id = str(row["Id"])
        short_id = full_id[:8]
        patient_id_map[short_id] = full_id
        G.add_node(
            f"patient_{full_id}",
            node_type="Patient",
            full_id=full_id,
            short_id=short_id,
            first=str(row.get("FIRST", "")),
            last=str(row.get("LAST", "")),
            gender=str(row.get("GENDER", "")),
            race=str(row.get("RACE", "")),
            birthdate=str(row.get("BIRTHDATE", "")),
            city=str(row.get("CITY", "")),
            state=str(row.get("STATE", "")),
        )
        node_counts["Patient"] += 1

    # Track patient->wave appeared_in edges to avoid duplicates
    appeared_in_seen: set[tuple] = set()

    def _add_appeared_in(patient_full_id: str, wave_num: int):
        key = (patient_full_id, wave_num)
        if key not in appeared_in_seen:
            appeared_in_seen.add(key)
            G.add_edge(f"patient_{patient_full_id}", f"wave_{wave_num}", edge_type="appeared_in")
            edge_counts["appeared_in"] += 1

    # --- Condition nodes ---
    print("Building Condition nodes...")
    for idx, row in conditions.iterrows():
        patient_full_id = str(row["PATIENT"])
        wave_num = assign_wave(row.get("START"))
        if wave_num is None:
            continue
        code = str(row.get("CODE", ""))
        node_id = f"cond_{patient_full_id[:8]}_{code}_{wave_num}_{idx}"
        G.add_node(
            node_id,
            node_type="Condition",
            patient_id=patient_full_id,
            patient_short_id=patient_full_id[:8],
            wave=wave_num,
            code=code,
            description=str(row.get("DESCRIPTION", "")),
            start=str(row.get("START", "")),
            stop=str(row.get("STOP", "")),
        )
        node_counts["Condition"] += 1
        G.add_edge(f"patient_{patient_full_id}", node_id, edge_type="diagnosed_with")
        edge_counts["diagnosed_with"] += 1
        G.add_edge(f"wave_{wave_num}", node_id, edge_type="contains")
        edge_counts["contains"] += 1
        _add_appeared_in(patient_full_id, wave_num)

    # --- Observation nodes ---
    print("Building Observation nodes...")
    for idx, row in observations.iterrows():
        patient_full_id = str(row["PATIENT"])
        wave_num = assign_wave(row.get("DATE"))
        if wave_num is None:
            continue
        code = str(row.get("CODE", ""))
        node_id = f"obs_{patient_full_id[:8]}_{code}_{wave_num}_{idx}"
        G.add_node(
            node_id,
            node_type="Observation",
            patient_id=patient_full_id,
            patient_short_id=patient_full_id[:8],
            wave=wave_num,
            code=code,
            description=str(row.get("DESCRIPTION", "")),
            value=str(row.get("VALUE", "")),
            units=str(row.get("UNITS", "")),
            obs_type=str(row.get("TYPE", "")),
            date=str(row.get("DATE", "")),
        )
        node_counts["Observation"] += 1
        G.add_edge(f"patient_{patient_full_id}", node_id, edge_type="measured")
        edge_counts["measured"] += 1
        G.add_edge(f"wave_{wave_num}", node_id, edge_type="contains")
        edge_counts["contains"] += 1
        _add_appeared_in(patient_full_id, wave_num)

    # --- Medication nodes ---
    print("Building Medication nodes...")
    for idx, row in medications.iterrows():
        patient_full_id = str(row["PATIENT"])
        wave_num = assign_wave(row.get("START"))
        if wave_num is None:
            continue
        code = str(row.get("CODE", ""))
        node_id = f"med_{patient_full_id[:8]}_{code}_{wave_num}_{idx}"
        G.add_node(
            node_id,
            node_type="Medication",
            patient_id=patient_full_id,
            patient_short_id=patient_full_id[:8],
            wave=wave_num,
            code=code,
            description=str(row.get("DESCRIPTION", "")),
            start=str(row.get("START", "")),
            stop=str(row.get("STOP", "")),
            base_cost=str(row.get("BASE_COST", "")),
        )
        node_counts["Medication"] += 1
        G.add_edge(f"patient_{patient_full_id}", node_id, edge_type="prescribed")
        edge_counts["prescribed"] += 1
        G.add_edge(f"wave_{wave_num}", node_id, edge_type="contains")
        edge_counts["contains"] += 1
        _add_appeared_in(patient_full_id, wave_num)

    # Store lookup map as graph-level attribute
    G.graph["patient_id_map"] = patient_id_map
    G.graph["wave_ranges"] = WAVE_RANGES

    # Save
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving graph to {OUTPUT_PATH} ...")
    with open(OUTPUT_PATH, "wb") as f:
        pickle.dump(G, f)

    # Stats
    print("\n=== GRAPH STATISTICS ===")
    print(f"Total nodes : {G.number_of_nodes():>8,}")
    print(f"Total edges : {G.number_of_edges():>8,}")
    print("\nNodes by type:")
    for ntype, count in node_counts.items():
        print(f"  {ntype:<14}: {count:>6,}")
    print("\nEdges by type:")
    for etype, count in edge_counts.items():
        print(f"  {etype:<16}: {count:>6,}")

    return G


if __name__ == "__main__":
    build_graph()
