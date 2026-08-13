"""
Phase 1 Orchestrator (Basic Public Broadcast).

This script:
  1) Generates the topology with a specific seed.
  2) Spawns an independent process (multiprocessing) with a separate UDP socket for each node.
  3) Executes a 200-packet scenario: The origin of each packet is randomly selected from
     honest nodes (spies excluded), with no fixed timing pattern (random injection intervals).
  4) Records the "origin" event in the ground-truth log for each packet (actual origin +
     actual time) — this information is never placed in the network packet.
  5) Sends a SHUTDOWN command to all nodes after the dissemination finishes.
  6) Calculates T_80% (time to reach 80% of nodes) for each packet from the log.
"""

import json
import multiprocessing as mp
import random
import socket
import statistics
import time
from pathlib import Path
from typing import Optional, Set

from node_process import run_node
from packet import new_packet_id
from sim_log import log_event, read_log
from topology import generate_topology

BASE_PORT = 20000


def build_node_configs(topo):
    node_ids = {n: f"n{n}" for n in topo.graph.nodes}
    addr_of = {n: ("127.0.0.1", BASE_PORT + n) for n in topo.graph.nodes}

    configs = {}
    for n in topo.graph.nodes:
        peers, peer_delay = {}, {}
        for nb in topo.graph.neighbors(n):
            peers[node_ids[nb]] = addr_of[nb]
            peer_delay[node_ids[nb]] = topo.graph[n][nb]["delay_base_ms"]
        configs[n] = {
            "node_id": node_ids[n],
            "self_addr": addr_of[n],
            "peers": peers,
            "peer_delay": peer_delay,
        }
    return configs, node_ids, addr_of


def run_phase1(seed: int, num_packets: int, log_path: str, settle_time_s: float = 3.0,
               stem_p=None, topology_seed=None, origin_seed=None, run_seed=None,
               spy_ids: Optional[Set[str]] = None, topo=None):
    """
    stem_p=None  -> Phase 1/2: Simple Flood (legacy behavior, unchanged).
    stem_p=<p>   -> Phase 3+: Dandelion with probability p to continue Stem.

    topology_seed / origin_seed / run_seed are separated for Phase 3+
    to keep the topology and origin sequences constant across multiple
    comparative runs (e.g., p=0.9 vs p=0.1). Only the random network behavior
    (link jitter + Stem/Fluff coin) changes between runs. If not provided,
    all default to `seed` (exact legacy behavior of Phase 1).

    spy_ids (Phase 2 & Phase 5 Fix): A set of node_ids representing spies.
    Spies are excluded from being packet origins (Phase 2 fix).
    If a node is in spy_ids and we are in Phase 5, it will also apply
    intentional delay behavior. If None, no node applies intentional delay.

    topo: an already-generated Topology object. When the caller already
    built the topology (e.g. to run select_bribed_nodes on it before
    starting the simulation), pass it here to avoid regenerating it a
    second time from topology_seed — generation is deterministic so the
    result would be identical, but recomputing it is wasted work and
    reads confusingly as if topology and spy selection were independent
    steps when spy selection must actually happen on this exact object.
    """
    topology_seed = seed if topology_seed is None else topology_seed
    origin_seed = seed if origin_seed is None else origin_seed
    run_seed = seed if run_seed is None else run_seed

    Path(log_path).write_text("")  # Clear previous run log

    if topo is None:
        topo = generate_topology(topology_seed)
    configs, node_ids, addr_of = build_node_configs(topo)
    log_lock = mp.Lock()
    procs = []
    
    for n, cfg in configs.items():
        is_spy = spy_ids is not None and cfg["node_id"] in spy_ids
        # Large offset separated from main rng seed space (run_seed*1000+n) so there is
        # no overlap with other nodes' seeds or the main rng of this node.
        spy_delay_seed = (run_seed * 1000 + n + 900_000_000) if is_spy else None
        
        p = mp.Process(
            target=run_node,
            kwargs=dict(
                node_id=cfg["node_id"], self_addr=cfg["self_addr"], peers=cfg["peers"],
                peer_delay=cfg["peer_delay"], log_path=log_path, seed=run_seed * 1000 + n,
                stem_p=stem_p,
                lock=log_lock,
                is_spy=is_spy,
                spy_delay_seed=spy_delay_seed,
            ),
            daemon=True,
        )
        p.start()
        procs.append(p)

    time.sleep(0.3)  # Allow time for UDP sockets to bind

    rng = random.Random(origin_seed)
    ctrl_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    
    # CRITICAL FIX: Filter out spy nodes so they cannot be chosen as packet origins
    if spy_ids is not None:
        origin_nodes = [n for n in topo.graph.nodes if node_ids[n] not in spy_ids]
    else:
        origin_nodes = list(topo.graph.nodes)

    for _ in range(num_packets):
        origin = rng.choice(origin_nodes)  # Now it strictly picks from honest nodes
        packet_id = new_packet_id()
        time.sleep(rng.uniform(0.004, 0.02))  # No pre-determined timing pattern
        
        log_event(log_path, {
            "event": "origin",
            "node_id": node_ids[origin],
            "packet_id": packet_id,
            "wall_time": time.time(),
        }, lock=log_lock)
        
        ctrl_sock.sendto(json.dumps({"type": "INJECT", "packet_id": packet_id}).encode(), addr_of[origin])

    time.sleep(settle_time_s)  # Allow time for the last packets to reach all nodes

    for cfg in configs.values():
        ctrl_sock.sendto(json.dumps({"type": "SHUTDOWN"}).encode(), cfg["self_addr"])

    for p in procs:
        p.join(timeout=3)
        if p.is_alive():
            p.terminate()

    return topo, node_ids


def compute_t80(log_path: str, total_nodes: int):
    events = read_log(log_path)
    origins, receives = {}, {}
    for e in events:
        if e["event"] == "origin":
            origins[e["packet_id"]] = e["wall_time"]
        elif e["event"] == "receive":
            receives.setdefault(e["packet_id"], []).append(e["wall_time"])

    target = max(1, round(0.8 * total_nodes))
    t80s = []
    for pid, t0 in origins.items():
        times = sorted(receives.get(pid, []))
        if len(times) >= target:
            t80s.append(times[target - 1] - t0)
    return t80s, len(origins)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from pathlib import Path
    import statistics

    seed_val = 42
    
    # 1. Create a master output directory with nested logs and phase folders
    base_out_dir = Path("outputs")
    log_dir = base_out_dir / "logs"
    phase1_out_dir = base_out_dir / "phase1"
    
    # Create the directories if they don't exist
    log_dir.mkdir(parents=True, exist_ok=True)
    phase1_out_dir.mkdir(parents=True, exist_ok=True)
    
    # Define the log path explicitly inside the logs directory
    log_path = log_dir / f"phase1_seed{seed_val}.jsonl"

    # 2. Run Phase 1
    # Pass the string representation of the log_path
    topo, node_ids = run_phase1(seed=seed_val, num_packets=200, log_path=str(log_path))

    # 3. Extract topology summary
    summary_text = topo.summary()
    
    # 4. Save JSON and plot for the topology INSIDE the phase1 directory
    topo.save(str(phase1_out_dir / f"topology_seed{seed_val}.json"))
    topo.plot(str(phase1_out_dir / f"topology_seed{seed_val}.png"))

    # 5. Compute T_80%
    t80s, n_origins = compute_t80(str(log_path), topo.graph.number_of_nodes())
    
    # Prepare the results text
    results_str = f"Nodes = {topo.graph.number_of_nodes()}\n"
    results_str += f"Packets injected = {n_origins}, reached 80% coverage = {len(t80s)}\n"
    
    if t80s:
        t80s_ms = [t * 1000 for t in t80s]
        mean_val = statistics.mean(t80s_ms)
        median_val = statistics.median(t80s_ms)
        std_val = statistics.pstdev(t80s_ms)
        
        results_str += f"T_80%: avg={mean_val:.1f}ms  median={median_val:.1f}ms  stdev={std_val:.1f}ms\n"
        
        # --- Create and save Histogram ---
        plt.figure(figsize=(8, 6))
        plt.hist(t80s_ms, bins=15, color='skyblue', edgecolor='black', alpha=0.7)
        plt.axvline(mean_val, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {mean_val:.1f}ms')
        plt.axvline(median_val, color='green', linestyle='dashed', linewidth=2, label=f'Median: {median_val:.1f}ms')
        
        plt.title(f"Distribution of T_80% (Time to reach 80% coverage) - Seed {seed_val}")
        plt.xlabel("Time (ms)")
        plt.ylabel("Frequency (Number of Packets)")
        plt.legend()
        plt.grid(axis='y', alpha=0.75)
        
        # Save the histogram INSIDE the phase1 directory
        plot_filename = phase1_out_dir / f"t80_histogram_seed{seed_val}.png"
        plt.savefig(str(plot_filename), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Histogram successfully saved to: {plot_filename}")

    # Print to console for immediate feedback
    print("\n" + summary_text)
    print("\n" + results_str)
    
    # 6. Save all textual outputs to a summary file
    summary_filename = phase1_out_dir / f"phase1_summary_seed{seed_val}.txt"
    with open(summary_filename, "w", encoding="utf-8") as f:
        f.write("=== TOPOLOGY SUMMARY ===\n")
        f.write(summary_text + "\n\n")
        f.write("=== SIMULATION RESULTS ===\n")
        f.write(results_str)
        
    print(f"Outputs successfully routed: Texts/Images -> '{phase1_out_dir}', Logs -> '{log_dir}'")