"""
Phase 1 Orchestrator (Basic Public Broadcast).

This script:
  1) Generates the topology with a specific seed and verifies its consistency.
  2) Spawns an independent process (multiprocessing) with a separate UDP socket for each node.
  3) Executes a 200-packet scenario: The origin of each packet is randomly selected from
     honest nodes (spies excluded), with no fixed timing pattern.
  4) Records the "origin" event in the ground-truth log for each packet.
  5) Sends a SHUTDOWN command to all nodes after dissemination finishes.
  6) Calculates T_80% (time to reach 80% of nodes) for each packet.
"""

import json
import multiprocessing as mp
import random
import socket
import statistics
import time
from pathlib import Path
from typing import Dict, Optional, Set

from node_process import run_node
from packet import new_packet_id
from sim_log import log_event, read_log
from topology import generate_topology
from config import seed_int, TOPOLOGY_SEED, ORIGIN_SEED, NUM_NODES

BASE_PORT = 20000


def build_node_configs(topo):
    """Build configuration dictionary for all nodes including addresses and delays."""
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
               spy_ids: Optional[Set[str]] = None, topo=None,
               spy_delay_enabled: bool = False):
    """
    stem_p=None  -> Phase 1/2: Simple Flood.
    stem_p=<p>   -> Phase 3+: Dandelion with probability p to continue Stem.

    topology_seed / origin_seed / run_seed are separated to keep the topology
    and origin sequences constant across multiple comparative runs.

    spy_ids, when given, is used for TWO independent things: (1) excluding
    those nodes from ever being chosen as a packet origin (per spec, origins
    are always honest nodes, in every phase), and (2) marking is_spy=True on
    those node processes for post-hoc "what would a spy see" log analysis.
    It does NOT by itself cause any timing change. spy_delay_enabled is the
    separate, Phase-5-only switch that turns on spies' intentional forwarding
    delay; every other phase must leave it False so that simply naming a spy
    set (Phase 2, or the origin-exclusion set reused in Phase 3/4) never
    silently perturbs propagation timing.
    """
    # Use globally configured seeds if not explicitly provided
    topology_seed = TOPOLOGY_SEED if topology_seed is None else topology_seed
    origin_seed = ORIGIN_SEED if origin_seed is None else origin_seed
    run_seed = seed_int("sim", "phase1", seed) if run_seed is None else run_seed

    Path(log_path).write_text("")  # Clear previous run log

    if topo is None:
        topo = generate_topology(topology_seed)
        
   # VERIFICATION: Ensure the topology is strictly consistent and within project limits (20-30)
    actual_nodes = topo.graph.number_of_nodes()
    assert 20 <= actual_nodes <= 30, f"Topology verification failed: {actual_nodes} nodes generated, expected between 20 and 30."


    configs, node_ids, addr_of = build_node_configs(topo)
    log_lock = mp.Lock()
    procs = []
    ready_events = {n: mp.Event() for n in configs}

    for n, cfg in configs.items():
        is_spy = spy_ids is not None and cfg["node_id"] in spy_ids

        # Use deterministic hash-based seeds to prevent overlap
        node_seed = seed_int("node_process", run_seed, n)
        spy_delay_seed = seed_int("spy_delay", run_seed, n) if is_spy else None

        p = mp.Process(
            target=run_node,
            kwargs=dict(
                node_id=cfg["node_id"], self_addr=cfg["self_addr"], peers=cfg["peers"],
                peer_delay=cfg["peer_delay"], log_path=log_path, seed=node_seed,
                stem_p=stem_p,
                lock=log_lock,
                is_spy=is_spy,
                spy_delay_seed=spy_delay_seed,
                spy_delay_enabled=spy_delay_enabled,
                ready_event=ready_events[n],
            ),
            daemon=True,
        )
        p.start()
        procs.append(p)

    # Wait for EVERY node's UDP socket to actually be bound before injecting any
    # packet. A fixed sleep here is not reliable: spawning 20-30 processes (each
    # re-importing networkx etc., since Windows uses the "spawn" start method)
    # can easily take longer than a short fixed delay under system load, and UDP
    # silently drops datagrams sent to a port nobody is listening on yet — with
    # no exception and no log trace, which would quietly lose packets.
    STARTUP_TIMEOUT_S = 60.0
    deadline = time.time() + STARTUP_TIMEOUT_S
    for n, ev in ready_events.items():
        remaining = max(0.0, deadline - time.time())
        if not ev.wait(timeout=remaining):
            raise RuntimeError(
                f"Node {node_ids[n]} did not finish binding its UDP socket within "
                f"{STARTUP_TIMEOUT_S}s; aborting before any packet is lost silently."
            )

    rng = random.Random(origin_seed)
    ctrl_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    
    # Filter out spy nodes so they cannot be chosen as packet origins
    if spy_ids is not None:
        origin_nodes = [n for n in topo.graph.nodes if node_ids[n] not in spy_ids]
    else:
        origin_nodes = list(topo.graph.nodes)

    for _ in range(num_packets):
        origin = rng.choice(origin_nodes)  # Strictly picks from honest nodes
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
    """Calculate the time it took for each packet to reach 80% of the network."""
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


def coverage_fractions(log_path: str, total_nodes: int):
    """
    Fraction of the network that ended up receiving each packet (shared by
    phase1/2/3/5 reporting, so coverage is computed identically everywhere).
    """
    events = read_log(log_path)
    receives: Dict[str, set] = {}
    for e in events:
        if e["event"] == "receive":
            receives.setdefault(e["packet_id"], set()).add(e["node_id"])
    return [len(nodes) / total_nodes for nodes in receives.values()]


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from pathlib import Path
    import statistics

    from config import TOPOLOGY_SEED, NUM_PACKETS, OUTPUTS_DIR

    seed_val = TOPOLOGY_SEED  

    # 1. Create a master output directory with nested logs and phase folders
    base_out_dir = Path(OUTPUTS_DIR)
    log_dir = base_out_dir / "logs"
    phase1_out_dir = base_out_dir / "phase1"
    
    log_dir.mkdir(parents=True, exist_ok=True)
    phase1_out_dir.mkdir(parents=True, exist_ok=True)
    
    log_path = log_dir / f"phase1_seed{seed_val}.jsonl"

    # 2. Run Phase 1
    topo, node_ids = run_phase1(seed=seed_val, num_packets=NUM_PACKETS, log_path=str(log_path))

    # 3. Extract topology summary
    summary_text = topo.summary()
    
    # 4. Save JSON and plot for the topology
    topo.save(str(phase1_out_dir / f"topology_seed{seed_val}.json"))
    topo.plot(str(phase1_out_dir / f"topology_seed{seed_val}.png"))

    # 5. Compute T_80%
    t80s, n_origins = compute_t80(str(log_path), topo.graph.number_of_nodes())
    
    results_str = f"Nodes = {topo.graph.number_of_nodes()}\n"
    results_str += f"Packets injected = {n_origins}, reached 80% coverage = {len(t80s)}\n"
    
    if t80s:
        t80s_ms = [t * 1000 for t in t80s]
        mean_val = statistics.mean(t80s_ms)
        median_val = statistics.median(t80s_ms)
        std_val = statistics.pstdev(t80s_ms)
        
        results_str += f"T_80%: avg={mean_val:.1f}ms  median={median_val:.1f}ms  stdev={std_val:.1f}ms\n"
        
        # Create and save Histogram
        plt.figure(figsize=(8, 6))
        plt.hist(t80s_ms, bins=15, color='skyblue', edgecolor='black', alpha=0.7)
        plt.axvline(mean_val, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {mean_val:.1f}ms')
        plt.axvline(median_val, color='green', linestyle='dashed', linewidth=2, label=f'Median: {median_val:.1f}ms')
        
        plt.title(f"Distribution of T_80% (Time to reach 80% coverage) - Seed {seed_val}")
        plt.xlabel("Time (ms)")
        plt.ylabel("Frequency (Number of Packets)")
        plt.legend()
        plt.grid(axis='y', alpha=0.75)
        
        plot_filename = phase1_out_dir / f"t80_histogram_seed{seed_val}.png"
        plt.savefig(str(plot_filename), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Histogram successfully saved to: {plot_filename}")

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