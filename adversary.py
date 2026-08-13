"""
Adversary Model and Attack Strategies.

This module implements:
  1) select_bribed_nodes: Picks spy nodes using a Greedy Set Cover approach
     to maximize topological vision within the budget ceiling.
  2) baseline_guess: One-hop backtrack (assumes the sender to the earliest
     spy is the origin).
  3) proposed_guess: Hybrid Attack (Topology Filter + TDOA Vector Profiling).
     This is the new, highly accurate default.
  4) phase4_guess: STEM-only backtrack model.
  5) Legacy/Comparison methods: common_ancestor_guess, vector_profiling_guess,
     and multilateration_guess are kept for benchmarking in phase2_simulator.
"""

import functools
import statistics
from typing import Dict, List, Optional, Set

import networkx as nx

from sim_log import read_log
from topology import JITTER_FRACTION, PLANE_SIZE


# ---------------------------------------------------------------------------
# Shared helper: all-pairs shortest-path delay
# ---------------------------------------------------------------------------
def build_distance_matrix(topo) -> Dict[int, Dict[int, float]]:
    """
    All-pairs shortest-path delay (ms), using each edge's delay_base_ms as
    weight. Computed once per topology and reused to save resources.
    """
    return dict(nx.all_pairs_dijkstra_path_length(topo.graph, weight="delay_base_ms"))


# ---------------------------------------------------------------------------
# 1) Spy Nodes Selection
# ---------------------------------------------------------------------------
def rank_spy_candidates(topo, max_k: Optional[int] = None) -> List[int]:
    """
    Produces an ORDERED ranking of candidate spy nodes (best first), up to
    `max_k` nodes (default: all nodes).
    
    Uses improved criteria: Coverage gain -> Separation (topology spread) -> Degree tie-break.
    """
    g = topo.graph
    n_total = g.number_of_nodes()
    if max_k is None:
        max_k = n_total

    covered: Set[int] = set()
    chosen: List[int] = []
    remaining = set(g.nodes)

    while len(chosen) < max_k and remaining:
        def score(u):
            coverage_gain = len(({u} | set(g.neighbors(u))) - covered)
            if chosen:
                separation = min(nx.shortest_path_length(g, u, s) for s in chosen)
            else:
                separation = 0
            return (coverage_gain, separation, g.degree(u))

        best = max(remaining, key=score)
        chosen.append(best)
        covered |= {best} | set(g.neighbors(best))
        remaining.discard(best)

    return chosen


def select_bribed_nodes(topo, budget_fraction: float = 0.3, k: Optional[int] = None,
                         **kwargs) -> Set[int]:
    """
    Selects spy nodes using the improved ranking from rank_spy_candidates().
    """
    g = topo.graph
    n_total = g.number_of_nodes()
    budget = k if k is not None else max(1, int(n_total * budget_fraction))

    spy_nodes = rank_spy_candidates(topo, max_k=budget)

    if kwargs.get("verbose"):
        print(f"  -> Ranked spy selection: {len(spy_nodes)} spies out of {budget} requested.")

    return set(spy_nodes)


def sweep_optimal_spy_count(topo, node_ids: Dict[int, str], seed: int, num_packets: int,
                             budget_fraction: float, log_path: str,
                             settle_time_s: float = 3.0) -> Dict:
    """
    Implements the project's requirement to sweep the spy count up to max budget
    to find the optimal number of bribed nodes in Phase 2.
    """
    from phase1_simulator import run_phase1  # Local import to avoid circular dependency

    n_total = topo.graph.number_of_nodes()
    max_k = max(1, int(n_total * budget_fraction))

    ranked_int = rank_spy_candidates(topo, max_k=max_k)
    ranked_ids = [node_ids[n] for n in ranked_int]
    full_spy_ids = set(ranked_ids)

    run_phase1(seed=seed, num_packets=num_packets, log_path=log_path, topo=topo,
               spy_ids=full_spy_ids, settle_time_s=settle_time_s)

    curve = []
    for kk in range(1, max_k + 1):
        subset = set(ranked_ids[:kk])
        r_base = evaluate_attack(log_path, subset, baseline_guess, topo)
        r_prop = evaluate_attack(log_path, subset, proposed_guess, topo)
        curve.append({
            "k": kk,
            "spy_ids": sorted(subset),
            "accuracy_baseline": r_base["accuracy"],
            "score_adv_baseline": r_base["score_adv"],
            "accuracy_proposed": r_prop["accuracy"],
            "score_adv_proposed": r_prop["score_adv"],
        })

    best = max(curve, key=lambda c: c["score_adv_proposed"])
    return {"curve": curve, "best": best, "ranked_ids": ranked_ids, "max_k": max_k}


# ---------------------------------------------------------------------------
# 2) Origin Guessing Methods
# ---------------------------------------------------------------------------
def _spy_sightings_by_packet(log_path: str, spy_ids: Set[str]):
    """
    Reads the ground-truth log and extracts the chronological list of
    packet sightings exclusively for spy nodes.
    """
    events = read_log(log_path)
    origins: Dict[str, dict] = {}
    sightings: Dict[str, List[dict]] = {}

    for e in events:
        if e["event"] == "origin":
            origins[e["packet_id"]] = e
        elif e["event"] == "receive" and e["node_id"] in spy_ids:
            sightings.setdefault(e["packet_id"], []).append(e)

    for pid in sightings:
        sightings[pid].sort(key=lambda e: e["wall_time"])

    return origins, sightings


def baseline_guess(sightings_for_packet: List[dict], topo=None, spy_ids=None, **kwargs) -> str:
    """
    Baseline method (Phase 2): one-hop backtrack.
    Assumes the neighbor who sent the packet to the earliest observing spy is the origin.
    """
    if not sightings_for_packet:
        return None
        
    earliest = sightings_for_packet[0]
    sender = earliest.get("sender_peer_id")

    return sender if sender is not None else earliest["node_id"]


def proposed_guess(sightings_for_packet: List[dict], topo=None, spy_ids=None, **kwargs) -> str:
    """
    Proposed method (Phase 2): Hybrid Attack (Topology Filter + TDOA Matching).
    Uses the first sender as a topology filter, then refines based on TDOA.
    """
    if topo is None or spy_ids is None or len(sightings_for_packet) < 2:
        return baseline_guess(sightings_for_packet, topo, spy_ids)

    g = topo.graph
    int_to_str = {n: f"n{n}" for n in g.nodes}
    str_to_int = {f"n{n}": n for n in g.nodes}

    first_spy_sighting = sightings_for_packet[0]
    first_sender_str = first_spy_sighting.get("sender_peer_id")
    
    if not first_sender_str or first_sender_str not in str_to_int:
        return first_spy_sighting["node_id"]

    first_sender_int = str_to_int[first_sender_str]
    
    # Candidate pool: The sender itself + its honest neighbors
    candidates_int = [first_sender_int] + list(g.neighbors(first_sender_int))
    valid_candidates_str = [int_to_str[c] for c in candidates_int if int_to_str[c] not in spy_ids]

    if not valid_candidates_str:
        return first_sender_str

    # Only consider the first 3 observing spies to minimize Jitter noise
    obs = sorted(sightings_for_packet, key=lambda e: e["wall_time"])[:3]
    if len(obs) < 2:
        return valid_candidates_str[0]

    t0 = obs[0]["wall_time"]
    observed_spy_ints = []
    observed_rel_ms = []
    
    for e in obs:
        sid = str_to_int.get(e["node_id"])
        if sid is not None:
            observed_spy_ints.append(sid)
            observed_rel_ms.append((e["wall_time"] - t0) * 1000.0)

    ref_spy = observed_spy_ints[0]
    best_candidate = None
    best_error = float("inf")

    # Match theoretical TDOA vs observed TDOA
    for c_str in valid_candidates_str:
        c_int = str_to_int[c_str]
        dist_from_c = nx.single_source_dijkstra_path_length(g, c_int, weight="delay_base_ms")
        
        if any(sid not in dist_from_c for sid in observed_spy_ints):
            continue

        predicted_rel_ms = [dist_from_c[sid] - dist_from_c[ref_spy] for sid in observed_spy_ints]
        error = sum((o - p) ** 2 for o, p in zip(observed_rel_ms, predicted_rel_ms))

        if error < best_error:
            best_error = error
            best_candidate = c_str

    return best_candidate if best_candidate is not None else first_sender_str


def phase4_guess(sightings_for_packet: List[dict], topo=None, spy_ids=None, **kwargs) -> str:
    """
    Phase 4 method: STEM-only backtrack. 
    Filters out FLUFF observations and uses the earliest STEM sighting.
    """
    stem_sightings = [e for e in sightings_for_packet if e.get("state") == "STEM"]
    if stem_sightings:
        earliest_stem = stem_sightings[0]
        sender = earliest_stem.get("sender_peer_id")
        return sender if sender is not None else earliest_stem["node_id"]

    return proposed_guess(sightings_for_packet, topo=topo, spy_ids=spy_ids)


# ---------------------------------------------------------------------------
# Legacy Origin Guessing Methods
# ---------------------------------------------------------------------------
def common_ancestor_guess(sightings_for_packet: List[dict], topo, spy_ids: Set[str], **kwargs) -> str:
    """Legacy Common-Ancestor attack."""
    if topo is None or spy_ids is None or len(sightings_for_packet) < 2:
        return baseline_guess(sightings_for_packet, topo, spy_ids)

    first_sightings = sightings_for_packet[:3]
    suspects = set()

    for s in first_sightings:
        sender = s.get("sender_peer_id")
        if sender and sender not in spy_ids:
            suspects.add(sender)

    if not suspects:
        return baseline_guess(sightings_for_packet, topo, spy_ids)

    g = topo.graph
    int_to_str = {n: f"n{n}" for n in g.nodes}
    str_to_int = {f"n{n}": n for n in g.nodes}

    suspects_int = [str_to_int[s] for s in suspects if s in str_to_int]

    best_candidates = []
    max_connections = -1

    for v_int in g.nodes:
        v_str = int_to_str[v_int]
        if v_str in spy_ids:
            continue

        connections = sum(1 for suspect_int in suspects_int if v_int == suspect_int or g.has_edge(v_int, suspect_int))

        if connections > max_connections:
            max_connections = connections
            best_candidates = [v_str]
        elif connections == max_connections:
            best_candidates.append(v_str)

    if best_candidates:
        base_guess = baseline_guess(sightings_for_packet, topo, spy_ids)
        if base_guess in best_candidates:
            return base_guess
        return best_candidates[0]

    return baseline_guess(sightings_for_packet, topo, spy_ids)


def build_reference_profiles(topo, spy_ids: Set[str]) -> Dict:
    """Offline profiling phase of the Vector Profiling attack."""
    dist_matrix = build_distance_matrix(topo)
    str_to_int = {f"n{n}": n for n in topo.graph.nodes}
    spy_order = sorted(spy_ids)
    spy_ints = [str_to_int[s] for s in spy_order if s in str_to_int]

    profiles: Dict[str, List[float]] = {}
    abs_delays: Dict[str, List[float]] = {}

    for n in topo.graph.nodes:
        n_str = f"n{n}"
        if n_str in spy_ids:
            continue

        dist_from_n = dist_matrix.get(n, {})
        delays = [dist_from_n.get(s, float("inf")) for s in spy_ints]
        finite = [d for d in delays if d != float("inf")]
        median_delay = statistics.median(finite) if finite else 0.0

        abs_delays[n_str] = delays
        profiles[n_str] = [d - median_delay if d != float("inf") else float("inf") for d in delays]

    return {"spy_order": spy_order, "profiles": profiles, "abs_delays": abs_delays}


def vector_profiling_guess(sightings_for_packet: List[dict], topo=None, spy_ids=None,
                            profile_data: Optional[Dict] = None, **kwargs) -> str:
    """Runtime guessing phase of the old global Vector Profiling attack."""
    if topo is None or spy_ids is None or profile_data is None or len(sightings_for_packet) < 2:
        return baseline_guess(sightings_for_packet, topo, spy_ids)

    profiles = profile_data["profiles"]
    abs_delays = profile_data["abs_delays"]
    spy_order = profile_data["spy_order"]
    spy_index = {s: i for i, s in enumerate(spy_order)}

    observed: Dict[str, float] = {}
    for e in sightings_for_packet:
        if e["node_id"] in spy_index and e["node_id"] not in observed:
            observed[e["node_id"]] = e["wall_time"]

    if len(observed) < 2:
        return baseline_guess(sightings_for_packet, topo, spy_ids)

    obs_median = statistics.median(observed.values())
    obs_vector_ms = {s: (t - obs_median) * 1000.0 for s, t in observed.items()}

    best_node, best_error = None, float("inf")
    for node_str, ref_vector in profiles.items():
        node_abs_delays = abs_delays[node_str]
        total_error, terms = 0.0, 0

        for s, obs_rel_ms in obs_vector_ms.items():
            idx = spy_index[s]
            ref_rel_ms = ref_vector[idx]
            if ref_rel_ms == float("inf"):
                continue

            expected_delay = node_abs_delays[idx]
            variance = max((JITTER_FRACTION * expected_delay) ** 2, 1.0)
            total_error += ((obs_rel_ms - ref_rel_ms) ** 2) / variance
            terms += 1

        if terms == 0:
            continue
        avg_error = total_error / terms

        if avg_error < best_error:
            best_error = avg_error
            best_node = node_str

    return best_node if best_node is not None else baseline_guess(sightings_for_packet, topo, spy_ids)


def multilateration_guess(sightings_for_packet: List[dict], topo=None, spy_ids=None,
                           dist_matrix: Optional[Dict[int, Dict[int, float]]] = None,
                           max_observations: int = 5, **kwargs) -> str:
    """TDOA multilateration globally (kept for comparison)."""
    if topo is None or spy_ids is None or len(sightings_for_packet) < 2:
        return baseline_guess(sightings_for_packet, topo, spy_ids)

    if dist_matrix is None:
        dist_matrix = build_distance_matrix(topo)

    str_to_int = {f"n{n}": n for n in topo.graph.nodes}

    obs = sorted(sightings_for_packet, key=lambda e: e["wall_time"])[:max_observations]
    t0 = obs[0]["wall_time"]

    observed_spy_ints: List[int] = []
    observed_rel_ms: List[float] = []
    for e in obs:
        sid = str_to_int.get(e["node_id"])
        if sid is None:
            continue
        observed_spy_ints.append(sid)
        observed_rel_ms.append((e["wall_time"] - t0) * 1000.0)

    if len(observed_spy_ints) < 2:
        return baseline_guess(sightings_for_packet, topo, spy_ids)

    ref_spy = observed_spy_ints[0]
    best_candidate, best_error = None, float("inf")

    for c in topo.graph.nodes:
        c_str = f"n{c}"
        if c_str in spy_ids:
            continue

        dist_from_c = dist_matrix.get(c, {})
        if any(sid not in dist_from_c for sid in observed_spy_ints):
            continue

        predicted_rel_ms = [dist_from_c[sid] - dist_from_c[ref_spy] for sid in observed_spy_ints]
        error = sum((o - p) ** 2 for o, p in zip(observed_rel_ms, predicted_rel_ms))

        if error < best_error:
            best_error = error
            best_candidate = c_str

    return best_candidate if best_candidate is not None else baseline_guess(sightings_for_packet, topo, spy_ids)


def make_multilateration_guess(topo):
    """Returns a ready-to-call multilateration_guess with its distance matrix precomputed once."""
    dist_matrix = build_distance_matrix(topo)
    return functools.partial(multilateration_guess, dist_matrix=dist_matrix)


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------
def plot_spy_selection(topo, spy_ids: Set[str], node_ids: Dict[int, str], path: str) -> None:
    """
    Saves a topology plot with the bribed (spy) nodes highlighted.
    """
    import matplotlib.pyplot as plt

    str_to_int = {v: k for k, v in node_ids.items()}
    spy_int_ids = {str_to_int[s] for s in spy_ids if s in str_to_int}

    fig, ax = plt.subplots(figsize=(8, 8))

    for u, v in topo.graph.edges():
        x1, y1 = topo.positions[u]
        x2, y2 = topo.positions[v]
        ax.plot([x1, x2], [y1, y2], color="lightgray", linewidth=0.8, zorder=1)

    colors = plt.cm.tab10.colors
    for n in topo.graph.nodes:
        c = topo.cluster_of[n]
        color = "black" if c == -1 else colors[c % len(colors)]
        x, y = topo.positions[n]
        ax.scatter(x, y, color=color, s=40, zorder=2)

    for n in spy_int_ids:
        x, y = topo.positions[n]
        ax.scatter(x, y, s=240, facecolors="none", edgecolors="red", linewidths=2.2, zorder=3)
        ax.annotate(node_ids[n], (x, y), textcoords="offset points", xytext=(6, 6),
                    fontsize=8, color="red", fontweight="bold")

    ax.scatter([], [], s=240, facecolors="none", edgecolors="red", linewidths=2.2, label="Bribed (spy) node")
    ax.set_xlim(0, PLANE_SIZE)
    ax.set_ylim(0, PLANE_SIZE)
    ax.set_title(f"Spy node selection ({len(spy_int_ids)} of {topo.graph.number_of_nodes()} nodes)")
    ax.legend(loc="upper right")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Evaluation: Accuracy and Score_adv
# ---------------------------------------------------------------------------
def evaluate_attack(log_path: str, spy_ids: Set[str], guess_fn, topo=None):
    """
    Evaluates the accuracy of an attack method with robust null/zero-division handling.
    """
    origins, sightings = _spy_sightings_by_packet(log_path, spy_ids)

    total = len(origins)
    observed = 0
    correct = 0

    for pid, o in origins.items():
        obs = sightings.get(pid)
        if not obs:
            continue

        observed += 1
        guess = guess_fn(obs, topo=topo, spy_ids=spy_ids)

        # Ensure guess exists before validating
        if guess is not None and guess == o["node_id"]:
            correct += 1

    accuracy = correct / total if total > 0 else 0.0
    n_spies = len(spy_ids)

    return {
        "total_packets": total,
        "observed_by_spies": observed,
        "accuracy": accuracy,
        "score_adv": accuracy / n_spies if n_spies > 0 else 0.0,
        "n_spies": n_spies,
    }

def evaluate_attack_all_methods(log_path: str, spy_ids: Set[str], topo=None,
                                 profile_data: Optional[Dict] = None):
    """
    Evaluates baseline, proposed (Hybrid Attack), common-ancestor, vector profiling, 
    and multilateration together for comprehensive benchmarking.
    """
    r_base = evaluate_attack(log_path, spy_ids, baseline_guess, topo)

    result = {
        "total_packets": r_base["total_packets"],
        "observed_by_spies": r_base["observed_by_spies"],
        "accuracy_baseline": r_base["accuracy"],
        "score_adv_baseline": r_base["score_adv"],
        "n_spies": r_base["n_spies"],
    }

    if topo is not None:
        r_prop = evaluate_attack(log_path, spy_ids, proposed_guess, topo)
        result["accuracy_proposed"] = r_prop["accuracy"]
        result["score_adv_proposed"] = r_prop["score_adv"]

        if profile_data is None:
            profile_data = build_reference_profiles(topo, spy_ids)
            
        vector_fn = functools.partial(vector_profiling_guess, profile_data=profile_data)
        r_vector = evaluate_attack(log_path, spy_ids, vector_fn, topo)
        result["accuracy_vector_profiling"] = r_vector["accuracy"]
        result["score_adv_vector_profiling"] = r_vector["score_adv"]

        r_common = evaluate_attack(log_path, spy_ids, common_ancestor_guess, topo)
        result["accuracy_common_ancestor"] = r_common["accuracy"]
        result["score_adv_common_ancestor"] = r_common["score_adv"]

        r_multi = evaluate_attack(log_path, spy_ids, make_multilateration_guess(topo), topo)
        result["accuracy_multilateration"] = r_multi["accuracy"]
        result["score_adv_multilateration"] = r_multi["score_adv"]
    else:
        result["accuracy_proposed"] = r_base["accuracy"]
        result["score_adv_proposed"] = r_base["score_adv"]

    return result