"""
Designing the attack on public broadcast

This module has two independent parts:

  1) select_bribed_nodes(...)  — Spy selection algorithm (max 30%).
  2) baseline_guess / proposed_guess — Two methods for guessing the origin based on spy observations.

The logic for both is detailed in the respective docstrings below.
"""

import random
import statistics
from typing import Dict, List, Set

import networkx as nx

from sim_log import read_log


# ---------------------------------------------------------------------------
# 1) Spy Nodes Selection
# ---------------------------------------------------------------------------
def select_bribed_nodes(topo, budget_fraction: float = 0.3) -> Set[int]:
    """
    Node score = its degree + 2 * (number of its cluster's boundary edges 
    connected to this node). Cluster boundary nodes (bridge/gateway) receive 
    double points because any packet exchanged between two clusters must 
    pass through one of them — intercepting these nodes gives us the most 
    'visibility' with the fewest number of spies.

    To prevent concentrating all spies in one cluster (which would only see 
    packets near that cluster well), the selection is done round-robin among 
    clusters: from each cluster, the highest-scoring unselected node is picked, 
    and this cycle continues until the budget is reached.
    """
    g = topo.graph
    cluster_of = topo.cluster_of
    n_total = g.number_of_nodes()
    budget = max(1, int(n_total * budget_fraction))

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
    chosen: List[int] = []

    while len(chosen) < budget:
        progressed = False
        for c in cluster_ids:
            if len(chosen) >= budget:
                break
            lst = clusters[c]
            p = pointers[c]
            if p < len(lst):
                chosen.append(lst[p])
                pointers[c] = p + 1
                progressed = True
        if not progressed:
            break  # All nodes are exhausted

    return set(chosen)


# ---------------------------------------------------------------------------
# 2) Origin Guessing Methods
# ---------------------------------------------------------------------------
def _spy_sightings_by_packet(log_path: str, spy_ids: Set[str]):
    """
    For each packet_id, returns a list of 'receive' events where the node_id 
    belongs to the spies, sorted by wall_time. Each event includes the state 
    (STEM/FLUFF) and sender_peer_id (which neighbor the spy heard the packet 
    from and in what state).
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


def baseline_guess(sightings_for_packet: List[dict]) -> str:
    """
    Baseline method (Phase 2): One-hop backtrack.
    
    Since spies are no longer selected as packet origins, guessing the spy 
    itself yields 0% accuracy. The new baseline assumes the neighbor who 
    sent the packet to the first observing spy (sender_peer_id) is the origin.
    """
    earliest = sightings_for_packet[0]
    sender = earliest.get("sender_peer_id")
    
    # Fallback to the node itself if sender is None (should not happen in practice)
    return sender if sender is not None else earliest["node_id"]


def proposed_guess(sightings_for_packet: List[dict]) -> str:
    """
    Proposed method for Phase 2 — "One-hop backtrack" (ignoring STEM/FLUFF):
    Guess = the neighbor who sent the packet to the earliest-seeing spy 
    (sender_peer_id). Works well for Flood because dissemination is radial 
    and simultaneous. 
    Does not work well for Dandelion (Phase 3): If the earliest spy saw the 
    packet in FLUFF state, it means it saw it after the Stem->Fluff transition 
    point — which could be very far from the actual origin (since before the 
    transition, the packet traveled the point-to-point Stem path and no spy 
    saw it). "One step back" from this spy only gets us closer to the transition 
    point, not the origin.
    """
    earliest = sightings_for_packet[0]
    sender = earliest.get("sender_peer_id")
    return sender if sender is not None else earliest["node_id"]


def phase4_guess(sightings_for_packet: List[dict]) -> str:
    """
    Phase 4 method — "STEM-only Backtrack":

    Core idea: In Dandelion, only packets in the STEM state have truly 
    traversed the hidden path near the origin. Therefore, we **only** trust 
    observations where the spy saw the packet in the STEM state (not FLUFF).
    These observations are guaranteed to be on the point-to-point Stem path, 
    so the earlier they are, the closer they are to the origin.

    Algorithm:
      1) Among the observations of these spies, keep only those where 
         state == STEM (already sorted by wall_time).
      2) If at least one exists: pick the earliest one and go "one step back" 
         (its sender_peer_id — exactly like proposed_guess, but this time 
         on a STEM observation, not just any observation).
      3) If no spy saw the packet in the STEM state (meaning the Stem path 
         turned to Fluff before reaching any spy), we lack hidden path data; 
         in this case, we fallback to the best available option: the same 
         proposed_guess (one-hop backtrack from the earliest FLUFF observation) 
         — because it's still better than baseline_guess, though less accurate 
         than case (2).

    Why this is better than Phase 2's proposed_guess: proposed_guess always 
    uses the earliest observation overall (STEM or FLUFF); but a FLUFF observation 
    can temporally precede a STEM observation while being spatially much farther 
    from the origin (since Fluff means rapid radial broadcast has started and 
    can reach a spy near the transition point faster, even if that point is 
    several hops away from the origin). By strictly prioritizing STEM observations, 
    this error is eliminated.
    """
    stem_sightings = [e for e in sightings_for_packet if e["state"] == "STEM"]
    if stem_sightings:
        earliest_stem = stem_sightings[0]
        sender = earliest_stem.get("sender_peer_id")
        return sender if sender is not None else earliest_stem["node_id"]

    return proposed_guess(sightings_for_packet)


# ---------------------------------------------------------------------------
# Evaluation: Accuracy and Score_adv for a given guessing method
# ---------------------------------------------------------------------------
def evaluate_attack(log_path: str, spy_ids: Set[str], guess_fn=baseline_guess):
    origins, sightings = _spy_sightings_by_packet(log_path, spy_ids)

    total = len(origins)
    observed = 0
    correct = 0

    for pid, o in origins.items():
        obs = sightings.get(pid)
        if not obs:
            continue  # No spy saw this packet -> unguessable
        observed += 1
        if guess_fn(obs) == o["node_id"]:
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


def evaluate_attack_all_methods(log_path: str, spy_ids: Set[str]):
    """For compatibility with phase2_simulator: returns both Phase 2 methods' stats together."""
    r_base = evaluate_attack(log_path, spy_ids, baseline_guess)
    r_prop = evaluate_attack(log_path, spy_ids, proposed_guess)
    return {
        "total_packets": r_base["total_packets"],
        "observed_by_spies": r_base["observed_by_spies"],
        "accuracy_baseline": r_base["accuracy"],
        "accuracy_proposed": r_prop["accuracy"],
        "score_adv_baseline": r_base["score_adv"],
        "score_adv_proposed": r_prop["score_adv"],
        "n_spies": r_base["n_spies"],
    }