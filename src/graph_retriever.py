import pickle
from pathlib import Path
from typing import Optional

GRAPH_PATH = Path(__file__).parent.parent / "data" / "knowledge_graph.gpickle"

_graph_cache = None


def load_graph(path=None):
    global _graph_cache
    if _graph_cache is not None and path is None:
        return _graph_cache
    p = path or GRAPH_PATH
    with open(p, "rb") as f:
        g = pickle.load(f)
    if path is None:
        _graph_cache = g
    return g


def resolve_patient_node(G, patient_id: str) -> Optional[str]:
    """Return graph node ID for a patient given their short (8-char) or full UUID."""
    pid = str(patient_id).strip()

    # Direct full-UUID node
    if G.has_node(f"patient_{pid}"):
        return f"patient_{pid}"

    # Lookup via short_id map stored on the graph
    id_map: dict = G.graph.get("patient_id_map", {})
    if pid in id_map:
        return f"patient_{id_map[pid]}"

    # Fallback: scan nodes for prefix match (slower, last resort)
    for node_id, data in G.nodes(data=True):
        if data.get("node_type") == "Patient":
            if data.get("full_id", "").startswith(pid) or data.get("short_id") == pid:
                return node_id

    return None


def _get_clinical_data(G, patient_node: str, waves: Optional[list] = None) -> dict:
    """Collect conditions, observations, medications for a patient, optionally filtered by wave(s)."""
    result = {"conditions": [], "observations": [], "medications": []}
    wave_set = set(waves) if waves else None

    for neighbor in G.successors(patient_node):
        nd = G.nodes[neighbor]
        ntype = nd.get("node_type", "")
        if ntype not in ("Condition", "Observation", "Medication"):
            continue
        if wave_set and nd.get("wave") not in wave_set:
            continue
        bucket = {"Condition": "conditions", "Observation": "observations", "Medication": "medications"}[ntype]
        result[bucket].append(nd)

    return result


def _fmt_conditions(items: list) -> list[str]:
    lines = []
    seen: set[str] = set()
    for c in items:
        desc = c.get("description", "Unknown")
        if desc in seen:
            continue
        seen.add(desc)
        start = c.get("start", "")[:10]
        lines.append(f"  - {desc} (onset: {start or 'unknown'})")
    return lines


def _fmt_observations(items: list) -> list[str]:
    lines = []
    # Group by description, keep most recent
    latest: dict[str, dict] = {}
    for o in items:
        desc = o.get("description", "Unknown")
        if desc not in latest or o.get("date", "") > latest[desc].get("date", ""):
            latest[desc] = o
    for desc, o in list(latest.items())[:25]:
        val = o.get("value", "")
        units = o.get("units", "")
        date = o.get("date", "")[:10]
        lines.append(f"  - {desc}: {val} {units} (date: {date})".rstrip())
    return lines


def _fmt_medications(items: list) -> list[str]:
    lines = []
    seen: set[str] = set()
    for m in items:
        desc = m.get("description", "Unknown")
        if desc in seen:
            continue
        seen.add(desc)
        start = m.get("start", "")[:10]
        stop = m.get("stop", "")
        stop_str = stop[:10] if stop and stop not in ("", "nan") else "ongoing"
        lines.append(f"  - {desc} (from: {start}, to: {stop_str})")
    return lines


def _render_block(data: dict) -> list[str]:
    lines = []
    conds = data["conditions"]
    lines.append(f"CONDITIONS ({len(conds)}):")
    lines.extend(_fmt_conditions(conds) or ["  None recorded"])

    obs = data["observations"]
    lines.append(f"OBSERVATIONS ({len(obs)}):")
    lines.extend(_fmt_observations(obs) or ["  None recorded"])

    meds = data["medications"]
    lines.append(f"MEDICATIONS ({len(meds)}):")
    lines.extend(_fmt_medications(meds) or ["  None recorded"])

    return lines


def format_subgraph_context(
    G,
    patient_id: str,
    waves: Optional[list] = None,
    mode: str = "specific",
) -> str:
    """
    Build a plain-text context string from the patient's subgraph.

    mode values:
      specific      – retrieve exactly the listed wave(s)
      comparison    – render each wave as a labelled block side-by-side
      full_history  – all waves, one combined block
    """
    patient_node = resolve_patient_node(G, patient_id)
    if patient_node is None:
        return f"ERROR: Patient '{patient_id}' not found in the knowledge graph."

    pd = G.nodes[patient_node]
    wave_ranges = G.graph.get("wave_ranges", {})

    header = [
        "PATIENT RECORD",
        f"Patient ID : {pd.get('short_id', patient_id)}",
        f"Name       : {pd.get('first', '')} {pd.get('last', '')}",
        f"Gender     : {pd.get('gender', '')}  |  Race: {pd.get('race', '')}",
        f"Birthdate  : {pd.get('birthdate', '')}  |  Location: {pd.get('city', '')}, {pd.get('state', '')}",
        "",
    ]

    body: list[str] = []

    if mode == "comparison" and waves and len(waves) >= 2:
        for w in sorted(waves):
            yr = wave_ranges.get(w, (0, 0))
            yr_str = f"{yr[0]}-{yr[1]}" if yr[1] < 9999 else f"{yr[0]}+"
            body.append(f"=== WAVE {w} ({yr_str}) ===")
            data = _get_clinical_data(G, patient_node, [w])
            body.extend(_render_block(data))
            body.append("")
    elif mode == "full_history" or not waves:
        body.append("=== FULL HISTORY (all waves) ===")
        data = _get_clinical_data(G, patient_node, None)
        body.extend(_render_block(data))
    else:
        # specific – one or more waves, single block
        w_labels = ", ".join(
            f"Wave {w} ({wave_ranges[w][0]}-{wave_ranges[w][1] if wave_ranges[w][1] < 9999 else ''})"
            for w in sorted(waves)
            if w in wave_ranges
        )
        body.append(f"=== {w_labels} ===")
        data = _get_clinical_data(G, patient_node, waves)
        body.extend(_render_block(data))

    return "\n".join(header + body)


# ---------------------------------------------------------------------------
# Quick CLI smoke-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    pid = sys.argv[1] if len(sys.argv) > 1 else "9358d6df"
    wave_arg = int(sys.argv[2]) if len(sys.argv) > 2 else 5

    print(f"Loading graph from {GRAPH_PATH} ...")
    G = load_graph()
    print(f"Graph loaded: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges\n")

    ctx = format_subgraph_context(G, pid, [wave_arg], mode="specific")
    print(ctx)
