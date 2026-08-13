"""
Adversary Model and Attack Strategies.

This module implements:
  1) select_bribed_nodes: picks the Score_adv-OPTIMAL number/set of spies
     (not just "as many as the budget allows") — see its docstring.
  2) baseline_guess: one-hop backtrack (assumes the neighbor of the
     earliest-observing spy is the origin).
  3) common_ancestor_guess: the older Common-Ancestor / Graph-Intersection
     attack (kept available, but no longer the default "proposed" method).
  4) build_reference_profiles + vector_profiling_guess: the new two-phase
     Vector Profiling attack (offline analytical profiling + runtime
     nearest-reference-vector guessing). This is now `proposed_guess`.
  5) multilateration_guess: TDOA multilateration (kept for comparison;
     empirically weaker than the methods above in this setting, see its
     docstring).
  6) plot_spy_selection: visualizes which nodes were picked as spies.
"""

import functools
import random
import statistics
from typing import Dict, List, Optional, Set

import networkx as nx

from sim_log import read_log
from topology import JITTER_FRACTION, PLANE_SIZE


# ---------------------------------------------------------------------------
# Shared helper: all-pairs shortest-path delay (used by several methods below)
# ---------------------------------------------------------------------------
def build_distance_matrix(topo) -> Dict[int, Dict[int, float]]:
    """
    All-pairs shortest-path delay (ms), using each edge's delay_base_ms as
    weight. The graph is small (<= 30 nodes) so computing this once per
    topology is cheap, and it can be reused across every packet, every
    candidate spy set, and every run on that topology.
    """
    return dict(nx.all_pairs_dijkstra_path_length(topo.graph, weight="delay_base_ms"))


# ---------------------------------------------------------------------------
# 1) Spy Nodes Selection — now Score_adv-optimal, not just "fill the budget"
# ---------------------------------------------------------------------------
def _rank_candidates_by_structure(topo, max_budget: int) -> List[int]:
    """
    Produces a fixed candidate ORDER of up to `max_budget` node ids, ranked
    by the same structural score as before: degree + 2x(number of edges
    crossing a cluster boundary). Boundary/gateway nodes score higher
    because inter-cluster traffic is forced through them. Selection is
    round-robin across clusters so the ranking doesn't front-load every
    high-score node from a single cluster (keeps spies spread out).
    This part is unchanged from the earlier implementation and is purely
    topological — it does not depend on traffic or Score_adv.
    """
    g = topo.graph
    cluster_of = topo.cluster_of

    boundary_count = {n: 0 for n in g.nodes}
    for u, v in g.edges():
        if cluster_of[u] != cluster_of[v]:
            boundary_count[u] += 1
            boundary_count[v] += 1

    score = {n: g.degree(n) + 2 * boundary_count[n] for n in g.nodes}

    clusters: Dict[int, List[int]] = {}
    for n, c in cluster_of.items():
        clusters.setdefault(c, []).append(n)
    for c in clusters:
        clusters[c].sort(key=lambda n: score[n], reverse=True)

    cluster_ids = list(clusters.keys())
    pointers = {c: 0 for c in cluster_ids}
    ranked: List[int] = []

    while len(ranked) < max_budget:
        progressed = False
        for c in cluster_ids:
            if len(ranked) >= max_budget:
                break
            lst = clusters[c]
            p = pointers[c]
            if p < len(lst):
                ranked.append(lst[p])
                pointers[c] = p + 1
                progressed = True
        if not progressed:
            break  # ran out of nodes before hitting max_budget

    return ranked


def _synthetic_accuracy_estimate(topo, spy_ids: Set[str], trials_per_origin: int = 20,
                                  seed: int = 0) -> float:
    """
    Fast, simulation-free accuracy estimate for a candidate spy set, used
    only to decide how many spies are worth bribing in select_bribed_nodes().

    For every honest node h (a hypothetical true origin) we already have an
    analytical reference vector (its shortest-path delay to each spy, from
    build_reference_profiles). We synthesize several noisy "observed"
    timing vectors by adding jitter to that reference using the SAME noise
    model the real simulator uses (topology.JITTER_FRACTION, i.e. +/-20% of
    the delay), then check whether vector_profiling_guess recovers h from
    the noisy sample. The fraction of correct recoveries across all
    (honest node x trial) draws is the accuracy estimate.

    This never launches real UDP node processes — it's pure in-memory
    arithmetic — so evaluating every candidate spy-set size from 1 up to
    the budget ceiling stays cheap enough to do exhaustively.
    """
    rng = random.Random(seed)
    profile_data = build_reference_profiles(topo, spy_ids)
    profiles = profile_data["profiles"]
    if not profiles:
        return 0.0

    correct, total = 0, 0
    for h_str, ref_vector in profiles.items():
        for _ in range(trials_per_origin):
            synthetic_sightings = []
            for i, s in enumerate(profile_data["spy_order"]):
                rel_delay = ref_vector[i]
                if rel_delay == float("inf"):
                    continue
                jitter = rng.uniform(-JITTER_FRACTION, JITTER_FRACTION) * abs(rel_delay)
                synthetic_sightings.append({
                    "node_id": s,
                    # arbitrary common time origin (0.0) since only relative
                    # timing between spies matters to vector_profiling_guess
                    "wall_time": (rel_delay + jitter) / 1000.0,
                })
            if len(synthetic_sightings) < 2:
                continue
            guess = vector_profiling_guess(synthetic_sightings, topo=topo, spy_ids=spy_ids,
                                            profile_data=profile_data)
            total += 1
            if guess == h_str:
                correct += 1

    return correct / total if total else 0.0


def select_bribed_nodes(topo, budget_fraction: float = 0.3, trials_per_origin: int = 20,
                         verbose: bool = False) -> Set[int]:
    """
    Selects the Score_adv-OPTIMAL set of spies, instead of blindly bribing
    every node up to the budget ceiling.

    Score_adv = Accuracy / Number_of_Spies — so every extra spy makes the
    denominator strictly worse, and is only "worth it" if it raises
    Accuracy by more than that denominator penalty. This function checks
    that trade-off explicitly rather than assuming "more spies = better".

    Algorithm (greedy ranking + exhaustive pre-evaluation over its prefixes):
      1) Rank up to `budget_fraction * n` candidate nodes using the same
         structural score as before (degree + cluster-boundary edges,
         round-robin across clusters) — see _rank_candidates_by_structure.
         This ceiling (e.g. 9 of 30 nodes) remains the most the adversary
         is ever willing to bribe.
      2) For every prefix size k = 1..len(candidates), take the first k
         ranked candidates as a trial spy set and get a fast, simulation-
         free accuracy estimate for it (_synthetic_accuracy_estimate),
         then compute Score_adv = accuracy / k.
      3) Return the k (and its corresponding spy set) with the highest
         Score_adv — i.e. the point where adding one more spy stopped
         being worth the extra 1/k penalty.

    Trying all nested prefixes (rather than all 2^k subsets) is the
    "greedy" part: it trusts the structural ranking to order candidates
    from most to least individually valuable, then does a cheap exhaustive
    search over "how many of the best ones to actually use" — capturing
    exactly the trade-off asked for, in O(budget) analytical evaluations
    instead of an exponential search over subsets.
    """
    max_budget = max(1, int(topo.graph.number_of_nodes() * budget_fraction))
    ranked_candidates = _rank_candidates_by_structure(topo, max_budget)
    node_ids = {n: f"n{n}" for n in topo.graph.nodes}

    best_k, best_score_adv, best_subset = 1, -1.0, set(ranked_candidates[:1])
    for k in range(1, len(ranked_candidates) + 1):
        subset_int = ranked_candidates[:k]
        subset_str = {node_ids[i] for i in subset_int}
        accuracy = _synthetic_accuracy_estimate(topo, subset_str, trials_per_origin=trials_per_origin)
        score_adv = accuracy / k

        if verbose:
            print(f"  k={k}: estimated_accuracy={accuracy:.3f}  Score_adv={score_adv:.4f}")

        if score_adv > best_score_adv:
            best_score_adv = score_adv
            best_k = k
            best_subset = set(subset_int)

    if verbose:
        print(f"  -> selected k={best_k} spies (estimated Score_adv={best_score_adv:.4f})")

    return best_subset


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


def baseline_guess(sightings_for_packet: List[dict], topo=None, spy_ids=None) -> str:
    """
    Baseline method (Phase 2): one-hop backtrack.

    Assumes the neighbor who sent the packet to the earliest observing
    spy (sender_peer_id) is the true origin.
    """
    earliest = sightings_for_packet[0]
    sender = earliest.get("sender_peer_id")

    return sender if sender is not None else earliest["node_id"]


def common_ancestor_guess(sightings_for_packet: List[dict], topo, spy_ids: Set[str]) -> str:
    """
    Common-Ancestor / Graph-Intersection attack (previously the "proposed"
    method; kept here for comparison, no longer the default).

    Collects the immediate senders to the first few observing spies, and
    searches the topology for the honest node connected to the most of them.
    """
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

        connections = 0
        for suspect_int in suspects_int:
            if v_int == suspect_int or g.has_edge(v_int, suspect_int):
                connections += 1

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
    """
    Offline profiling phase of the Vector Profiling attack.

    For every HONEST (non-spy) node h, analytically computes the
    shortest-path delay (ms, via Dijkstra over delay_base_ms) from h to
    each spy — i.e. the noise-free arrival delay a packet from h would
    have at every spy. This models "simulating a packet from each honest
    node's origin once" without actually running the real UDP simulator:
    the base delays fully determine the shortest-path expectation, and
    that's all this phase needs.

    Each honest node's raw delay vector is re-expressed relative to its
    own MEDIAN delay (not its minimum/closest spy): the median is a more
    robust reference point, since a single unusually close or far spy
    can't skew the whole profile the way anchoring on "the nearest spy"
    would (this directly targets the noise-sensitivity problem found when
    evaluating multilateration_guess, which anchored on the single
    earliest observation).

    Call this once per (topology, spy_ids) pair — it depends on neither
    packet traffic nor the number of packets, only on the network's
    static structure, so it must run BEFORE any runtime guessing.

    Returns a dict:
      {
        "spy_order": [spy_id, ...],                  # fixed, deterministic order
        "profiles": {honest_node_id: [delay_ms - median_delay_ms, ...]},
        "abs_delays": {honest_node_id: [delay_ms, ...]},  # kept for variance weighting
      }
    """
    dist_matrix = build_distance_matrix(topo)
    str_to_int = {f"n{n}": n for n in topo.graph.nodes}
    spy_order = sorted(spy_ids)
    spy_ints = [str_to_int[s] for s in spy_order if s in str_to_int]

    profiles: Dict[str, List[float]] = {}
    abs_delays: Dict[str, List[float]] = {}

    for n in topo.graph.nodes:
        n_str = f"n{n}"
        if n_str in spy_ids:
            continue  # only honest nodes are candidate origins

        dist_from_n = dist_matrix.get(n, {})
        delays = [dist_from_n.get(s, float("inf")) for s in spy_ints]
        finite = [d for d in delays if d != float("inf")]
        median_delay = statistics.median(finite) if finite else 0.0

        abs_delays[n_str] = delays
        profiles[n_str] = [d - median_delay if d != float("inf") else float("inf") for d in delays]

    return {"spy_order": spy_order, "profiles": profiles, "abs_delays": abs_delays}


def vector_profiling_guess(sightings_for_packet: List[dict], topo=None, spy_ids=None,
                            profile_data: Optional[Dict] = None) -> str:
    """
    Runtime guessing phase of the Vector Profiling attack (this is now
    `proposed_guess` — see the alias below).

    Builds the OBSERVED timing vector from this packet's spy sightings,
    expressed relative to their own MEDIAN observed arrival time (matching
    the reference vectors' normalization in build_reference_profiles), then
    compares it against every honest node's precomputed reference vector
    using a variance-NORMALIZED squared error: each per-spy term is divided
    by that spy's expected jitter variance under the candidate hypothesis,
    (JITTER_FRACTION * expected_delay)^2 — i.e. the exact noise model
    topology.sample_delay() actually uses. This down-weights long, noisy
    hops relative to short, reliable ones automatically, instead of
    treating every observation as equally trustworthy (the flaw that made
    the earlier multilateration_guess underperform simple heuristics).

    The honest candidate node with the lowest total normalized error is
    the guess.
    """
    if topo is None or spy_ids is None or profile_data is None or len(sightings_for_packet) < 2:
        return baseline_guess(sightings_for_packet, topo, spy_ids)

    profiles = profile_data["profiles"]
    abs_delays = profile_data["abs_delays"]
    spy_order = profile_data["spy_order"]
    spy_index = {s: i for i, s in enumerate(spy_order)}

    # Keep only the first sighting per spy (a spy's SeenSet means it only
    # ever reports one arrival time per packet anyway).
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
                continue  # spy unreachable from this candidate (disconnected — shouldn't happen)

            expected_delay = node_abs_delays[idx]
            variance = max((JITTER_FRACTION * expected_delay) ** 2, 1.0)  # 1ms^2 floor
            total_error += ((obs_rel_ms - ref_rel_ms) ** 2) / variance
            terms += 1

        if terms == 0:
            continue
        avg_error = total_error / terms  # keeps packets with more/fewer sightings comparable

        if avg_error < best_error:
            best_error = avg_error
            best_node = node_str

    return best_node if best_node is not None else baseline_guess(sightings_for_packet, topo, spy_ids)


def make_vector_profiling_guess(topo, spy_ids: Set[str]):
    """
    Returns a ready-to-call vector_profiling_guess with its offline
    reference profiles precomputed once (the offline profiling phase).
    Use this instead of calling build_reference_profiles + binding
    profile_data by hand at every call site.
    """
    profile_data = build_reference_profiles(topo, spy_ids)
    return functools.partial(vector_profiling_guess, profile_data=profile_data)


# `proposed_guess` now names the Vector Profiling attack. Requires topo AND
# spy_ids AND profile_data (via make_vector_profiling_guess) to run at full
# strength; falls back to baseline_guess otherwise.
proposed_guess = vector_profiling_guess


def phase4_guess(sightings_for_packet: List[dict], topo=None, spy_ids=None,
                  profile_data: Optional[Dict] = None) -> str:
    """
    Phase 4 method: STEM-only backtrack. Filters out FLUFF observations and
    uses the earliest STEM sighting; falls back to vector_profiling_guess
    (now the strongest fallback available) if no spy saw the packet in STEM.
    """
    stem_sightings = [e for e in sightings_for_packet if e.get("state") == "STEM"]
    if stem_sightings:
        earliest_stem = stem_sightings[0]
        sender = earliest_stem.get("sender_peer_id")
        return sender if sender is not None else earliest_stem["node_id"]

    return vector_profiling_guess(sightings_for_packet, topo, spy_ids, profile_data)


def make_phase4_guess(topo, spy_ids: Set[str]):
    """Returns a ready-to-call phase4_guess with its Vector Profiling fallback profile precomputed once."""
    profile_data = build_reference_profiles(topo, spy_ids)
    return functools.partial(phase4_guess, profile_data=profile_data)


def multilateration_guess(sightings_for_packet: List[dict], topo=None, spy_ids=None,
                           dist_matrix: Optional[Dict[int, Dict[int, float]]] = None,
                           max_observations: int = 5) -> str:
    """
    Time-Difference-of-Arrival (TDOA) multilateration (kept for comparison).

    Uses the shortest-path delay from a candidate origin to each of the
    `max_observations` earliest-observing spies, compares predicted vs.
    observed RELATIVE arrival times (earliest spy as time reference), and
    picks the candidate with least squared error.

    Reported honestly: across 8 tested seeds this did NOT beat the simpler
    baseline_guess in this setting (small graph, high per-hop jitter) —
    unweighted squared error over an unweighted timestamp reference lets
    farther/noisier spies dominate the fit. vector_profiling_guess above
    fixes exactly this with its variance-normalized error and median
    reference; kept here only as a comparison point.
    """
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


def plot_spy_selection(topo, spy_ids: Set[str], node_ids: Dict[int, str], path: str) -> None:
    """
    Saves a topology plot with the bribed (spy) nodes highlighted, so the
    output of select_bribed_nodes() can be inspected visually — e.g. to
    confirm spies land on cluster-boundary/gateway nodes and are spread
    across clusters rather than clumped in one.
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

        if guess == o["node_id"]:
            correct += 1

    accuracy = correct / total if total else 0.0
    n_spies = len(spy_ids)

    return {
        "total_packets": total,
        "observed_by_spies": observed,
        "accuracy": accuracy,
        "score_adv": accuracy / n_spies if n_spies else 0.0,
        "n_spies": n_spies,
    }


def evaluate_attack_all_methods(log_path: str, spy_ids: Set[str], topo=None,
                                 profile_data: Optional[Dict] = None):
    """
    Evaluates baseline, proposed (Vector Profiling), common-ancestor, and
    multilateration together. `profile_data` should be the output of
    build_reference_profiles(topo, spy_ids) — pass it in if the caller
    already ran the offline profiling phase (recommended, see
    phase2_simulator.py) to avoid recomputing it here.
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
        if profile_data is None:
            profile_data = build_reference_profiles(topo, spy_ids)
        proposed_fn = functools.partial(vector_profiling_guess, profile_data=profile_data)
        r_prop = evaluate_attack(log_path, spy_ids, proposed_fn, topo)
        result["accuracy_proposed"] = r_prop["accuracy"]
        result["score_adv_proposed"] = r_prop["score_adv"]

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