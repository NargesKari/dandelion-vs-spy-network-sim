"""
Adversary Model and Attack Strategies.

This module implements:
  1) select_bribed_nodes: Chooses spies based on graph topology (targeting gateways).
  2) baseline_guess: One-hop backtrack (assumes the neighbor of the first spy is the origin).
  3) proposed_guess: Common Ancestor (Graph Intersection) Attack.
"""

from typing import Dict, List, Set
import networkx as nx

from sim_log import read_log


# ---------------------------------------------------------------------------
# 1) Spy Nodes Selection
# ---------------------------------------------------------------------------
def select_bribed_nodes(topo, budget_fraction: float = 0.3) -> Set[int]:
    """
    Selects spy nodes based on their degree and boundary edges.
    Gateway nodes connecting different clusters are prioritized.
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
            break

    return set(chosen)


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
    Baseline method (Phase 2): One-hop backtrack.
    
    Assumes the neighbor who sent the packet to the earliest observing 
    spy (sender_peer_id) is the true origin.
    """
    earliest = sightings_for_packet[0]
    sender = earliest.get("sender_peer_id")
    
    return sender if sender is not None else earliest["node_id"]


def proposed_guess(sightings_for_packet: List[dict], topo, spy_ids: Set[str]) -> str:
    """
    Proposed method (Phase 2): Common Ancestor (Graph Intersection) Attack.
    
    Instead of relying on noisy timestamps, this exploits the deterministic 
    routing graph. It collects the immediate senders to the first few spies,
    and searches for a common neighbor (ancestor) that connects to all of them.
    """
    if topo is None or spy_ids is None or len(sightings_for_packet) < 2:
        return baseline_guess(sightings_for_packet, topo, spy_ids)
        
    # 1. Collect the 1-hop backtrack suspects from the first 3 observing spies.
    # Limiting to early spies reduces noise from long, winding packet paths.
    first_sightings = sightings_for_packet[:3]
    suspects = set()
    
    for s in first_sightings:
        sender = s.get("sender_peer_id")
        # Ensure the sender is known and is an honest node
        if sender and sender not in spy_ids:
            suspects.add(sender)
            
    if not suspects:
        return baseline_guess(sightings_for_packet, topo, spy_ids)
        
    g = topo.graph
    int_to_str = {n: f"n{n}" for n in g.nodes}
    str_to_int = {f"n{n}": n for n in g.nodes}
    
    suspects_int = [str_to_int[s] for s in suspects if s in str_to_int]
    
    # 2. Find nodes connected to the MAXIMUM number of these suspects
    best_candidates = []
    max_connections = -1
    
    for v_int in g.nodes:
        v_str = int_to_str[v_int]
        if v_str in spy_ids:
            continue
            
        connections = 0
        for suspect_int in suspects_int:
            # The candidate could be the suspect itself OR connected to it
            if v_int == suspect_int or g.has_edge(v_int, suspect_int):
                connections += 1
                
        if connections > max_connections:
            max_connections = connections
            best_candidates = [v_str]
        elif connections == max_connections:
            best_candidates.append(v_str)
            
    # 3. Resolve ties. If the baseline guess is among the top candidates, trust it.
    if best_candidates:
        base_guess = baseline_guess(sightings_for_packet, topo, spy_ids)
        if base_guess in best_candidates:
            return base_guess
        return best_candidates[0]
        
    return baseline_guess(sightings_for_packet, topo, spy_ids)


def phase4_guess(sightings_for_packet: List[dict], topo=None, spy_ids=None) -> str:
    """
    Phase 4 method: STEM-only Backtrack.
    Filters out FLUFF observations and uses the earliest STEM sighting.
    """
    stem_sightings = [e for e in sightings_for_packet if e.get("state") == "STEM"]
    if stem_sightings:
        earliest_stem = stem_sightings[0]
        sender = earliest_stem.get("sender_peer_id")
        return sender if sender is not None else earliest_stem["node_id"]

    return proposed_guess(sightings_for_packet, topo, spy_ids)


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


def evaluate_attack_all_methods(log_path: str, spy_ids: Set[str], topo=None):
    """Evaluates both Phase 2 methods simultaneously."""
    r_base = evaluate_attack(log_path, spy_ids, baseline_guess, topo)
    r_prop = evaluate_attack(log_path, spy_ids, proposed_guess, topo)
    
    return {
        "total_packets": r_base["total_packets"],
        "observed_by_spies": r_base["observed_by_spies"],
        "accuracy_baseline": r_base["accuracy"],
        "accuracy_proposed": r_prop["accuracy"],
        "score_adv_baseline": r_base["score_adv"],
        "score_adv_proposed": r_prop["score_adv"],
        "n_spies": r_base["n_spies"],
    }