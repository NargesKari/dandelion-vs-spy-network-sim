"""
Phase 3 — Optimized Dandelion Protocol with Parameter p Sweep

Key Improvements:
1. Bounded Stem walks (no infinite loops) using MAX_STEM_HOPS.
2. Deterministic seeding imported from global config.py.
3. Streaming logs to reduce memory overhead during large runs.
4. Discrete Event Simulation (Priority Queue) for mathematically accurate TDOA timing.
"""

import json
import statistics
import heapq
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set

import networkx as nx
import matplotlib.pyplot as plt

from config import (
    TOPOLOGY_SEED, NUM_PACKETS, P_VALUES, RUNS_PER_P,
    OUTPUTS_DIR, ORIGIN_SEED, seed_int
)
from topology import generate_topology, sample_delay


class StreamingLogger:
    def __init__(self, path: str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.file = open(self.path, 'w', encoding='utf-8')
    
    def write_event(self, **kwargs):
        self.file.write(json.dumps(kwargs) + '\n')
    
    def close(self):
        if self.file and not self.file.closed:
            self.file.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

def read_events(log_path: str):
    with open(log_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def run_dandelion_stem(source_node: int, adj: Dict[int, Set[int]], p: float,
                       rng, max_stem_hops: Optional[int] = None) -> Tuple[List[int], str]:
    num_nodes = len(adj)
    if max_stem_hops is None:
        max_stem_hops = num_nodes * 2
    
    current = source_node
    prev = None
    path = [source_node]
    
    for hop in range(max_stem_hops):
        if rng.random() > p:
            return path, "FLUFF"
        
        neighbors = [n for n in adj[current] if n != prev]
        
        if not neighbors:
            return path, "FLUFF"
        
        prev = current
        current = rng.choice(neighbors)
        path.append(current)
    
    return path, "FLUFF"


def simulate_packet_dandelion(packet_id: str, source_node: int,
                              topo, p: float, rng,
                              seen_sets: Dict[str, Set[str]],
                              logger: StreamingLogger):
    adj = {node: set(topo.graph.neighbors(node)) for node in topo.graph.nodes()}
    
    # 1. Stem Phase
    stem_path, final_state = run_dandelion_stem(source_node, adj, p, rng)
    
    for node in stem_path:
        seen_sets.setdefault(f"n{node}", set()).add(packet_id)
    
    time_elapsed = 0.0
    for i, node in enumerate(stem_path):
        logger.write_event(
            wall_time=time_elapsed,
            packet_id=packet_id,
            node_id=f"n{node}",
            event="receive" if i > 0 else "origin",
            state="STEM",
            sender_peer_id=f"n{stem_path[i-1]}" if i > 0 else None
        )
        if i < len(stem_path) - 1:
            next_node = stem_path[i + 1]
            delay = sample_delay(topo.graph[node][next_node]["delay_base_ms"], rng)
            time_elapsed += delay / 1000.0  
            
    # 2. Fluff Phase using Discrete Event Simulation (Priority Queue)
    if final_state == "FLUFF":
        last_stem_node = stem_path[-1]
        prev_sender = stem_path[-2] if len(stem_path) > 1 else None
        
        # Priority queue structure: (arrival_time, receiving_node, sender_node)
        fluff_queue = []
        
        # Initialize queue with neighbors of the last stem node
        neighbors = [n for n in adj[last_stem_node] if n != prev_sender]
        for neighbor in neighbors:
            delay = sample_delay(topo.graph[last_stem_node][neighbor]["delay_base_ms"], rng)
            arrival_time = time_elapsed + delay / 1000.0
            heapq.heappush(fluff_queue, (arrival_time, neighbor, last_stem_node))
            
        while fluff_queue:
            t, current, prev = heapq.heappop(fluff_queue)
            current_str = f"n{current}"
            
            # If the node has already received this packet from a faster path, ignore
            if packet_id in seen_sets.get(current_str, set()):
                continue
                
            # Process receipt
            seen_sets.setdefault(current_str, set()).add(packet_id)
            
            logger.write_event(
                wall_time=t,
                packet_id=packet_id,
                node_id=current_str,
                event="receive",
                state="FLUFF",
                sender_peer_id=f"n{prev}"
            )
            
            # Broadcast to neighbors
            next_neighbors = [n for n in adj[current] if n != prev]
            for neighbor in next_neighbors:
                if packet_id not in seen_sets.get(f"n{neighbor}", set()):
                    delay = sample_delay(topo.graph[current][neighbor]["delay_base_ms"], rng)
                    arrival_time = t + delay / 1000.0
                    heapq.heappush(fluff_queue, (arrival_time, neighbor, current))


def run_phase3_one_run(p: float, run_idx: int) -> Dict:
    plan_seed = seed_int("plan", p, run_idx)
    sim_seed = seed_int("sim", p, run_idx)
    
    sim_rng = __import__('random').Random(sim_seed)
    
    topo = generate_topology(TOPOLOGY_SEED)
    num_nodes = topo.graph.number_of_nodes()
    
    origin_seed_rng = __import__('random').Random(ORIGIN_SEED)
    node_list = sorted(topo.graph.nodes())
    origins = [origin_seed_rng.choice(node_list) for _ in range(NUM_PACKETS)]
    
    log_dir = Path(OUTPUTS_DIR) / "logs" / "phase3"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"p{p:.1f}_run{run_idx}.jsonl"
    
    seen_sets = {}
    
    with StreamingLogger(str(log_path)) as logger:
        for packet_idx, origin in enumerate(origins):
            packet_id = f"P3-{p:.1f}-R{run_idx}-{packet_idx:04d}"
            simulate_packet_dandelion(
                packet_id, origin, topo, p, sim_rng,
                seen_sets, logger
            )
    
    coverage = []
    t80_times = []
    receives = {}
    timestamps = {}
    
    for event in read_events(str(log_path)):
        pid = event["packet_id"]
        if event["event"] == "origin":
            timestamps.setdefault(pid, {})["origin"] = event["wall_time"]
        elif event["event"] == "receive":
            node = event["node_id"]
            t = event["wall_time"]
            receives.setdefault(pid, set()).add(node)
            node_times = timestamps.setdefault(pid, {})
            if node not in node_times or t < node_times[node]:
                node_times[node] = t
                
    for pid, nodes in receives.items():
        cov_fraction = len(nodes) / num_nodes
        coverage.append(cov_fraction)
        
        if cov_fraction >= 0.8:
            node_times = timestamps[pid]
            origin_t = node_times.get("origin", 0.0)
            sorted_arrival_times = sorted([node_times[n] for n in nodes])
            target_count = max(1, int(0.8 * num_nodes))
            if len(sorted_arrival_times) >= target_count:
                t80 = sorted_arrival_times[target_count - 1] - origin_t
                t80_times.append(t80)
    
    return {
        "p": p,
        "run": run_idx,
        "avg_coverage": statistics.mean(coverage) if coverage else 0.0,
        "std_coverage": statistics.pstdev(coverage) if len(coverage) > 1 else 0.0,
        "median_coverage": statistics.median(coverage) if coverage else 0.0,
        "full_coverage_count": sum(1 for c in coverage if c >= 0.99),
        "t80_count": len(t80_times),
        "t80_mean_ms": statistics.mean(t80_times) * 1000 if t80_times else None,
        "t80_median_ms": statistics.median(t80_times) * 1000 if t80_times else None,
        "t80_stdev_ms": statistics.pstdev(t80_times) * 1000 if len(t80_times) > 1 else None,
    }

def run_phase3_parameter_sweep():
    all_results = {p: [] for p in P_VALUES}
    for p in P_VALUES:
        print(f"\n=== Phase 3: Dandelion with p={p} ===")
        for run_idx in range(RUNS_PER_P):
            result = run_phase3_one_run(p, run_idx)
            all_results[p].append(result)
            print(f"  Run {run_idx+1}: "
                  f"coverage={result['avg_coverage']:.3f} ± {result['std_coverage']:.3f}, "
                  f"T80%={result['t80_mean_ms']:.0f}ms")
    return all_results

def print_phase3_summary(all_results):
    print("\n" + "="*100)
    print("PHASE 3 SUMMARY: DANDELION PARAMETER SWEEP")
    print("="*100)
    
    header = f"{'p':>5} | {'Coverage (mean/med/std)':>30} | {'T80% (mean/med)':>25} | {'Full Coverage %':>15}"
    print(header)
    print("-" * len(header))
    
    for p in sorted(P_VALUES):
        runs = all_results[p]
        coverages = [r["avg_coverage"] for r in runs]
        t80s = [r["t80_mean_ms"] for r in runs if r["t80_mean_ms"] is not None]
        full_cov = [r["full_coverage_count"] for r in runs]
        
        cov_str = (f"{statistics.mean(coverages):.3f} / "
                   f"{statistics.median(coverages):.3f} / "
                   f"{statistics.pstdev(coverages):.3f}")
        
        t80_str = (f"{statistics.mean(t80s):.0f} / {statistics.median(t80s):.0f}"
                   if t80s else "N/A")
        full_pct = (sum(full_cov) / (RUNS_PER_P * NUM_PACKETS)) * 100
        print(f"{p:>5} | {cov_str:>30} | {t80_str:>25} | {full_pct:>14.1f}%")

if __name__ == "__main__":
    print("Starting Phase 3: Dandelion Parameter Sweep")
    print(f"Configuration:")
    print(f"  - TOPOLOGY_SEED: {TOPOLOGY_SEED}")
    print(f"  - NUM_PACKETS: {NUM_PACKETS}")
    print(f"  - P_VALUES: {P_VALUES}")
    print(f"  - RUNS_PER_P: {RUNS_PER_P}")
    
    results = run_phase3_parameter_sweep()
    print_phase3_summary(results)
    
    output_file = Path(OUTPUTS_DIR) / "phase3_results.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    phase3_out_dir = Path(OUTPUTS_DIR) / "phase3"
    phase3_out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 5))
    data_cov = [[r["avg_coverage"] for r in results[p]] for p in sorted(P_VALUES)]
    ax.boxplot(data_cov, tick_labels=[f"p={p}" for p in sorted(P_VALUES)])
    ax.set_ylabel("Average Fraction of Covered Nodes")
    ax.set_title("Phase 3: Network Coverage vs. p (Dandelion)")
    ax.grid(axis="y", alpha=0.4)
    fig.savefig(str(phase3_out_dir / "coverage_vs_p.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    data_t80 = [[r["t80_mean_ms"] for r in results[p] if r["t80_mean_ms"] is not None] for p in sorted(P_VALUES)]
    ax.boxplot(data_t80, tick_labels=[f"p={p}" for p in sorted(P_VALUES)])
    ax.set_ylabel("T_80% (milliseconds)")
    ax.set_title("Phase 3: Time to Reach 80% of Nodes vs. p")
    ax.grid(axis="y", alpha=0.4)
    fig.savefig(str(phase3_out_dir / "t80_vs_p.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"\nResults saved to: {output_file}")
    print(f"Plots successfully saved to: {phase3_out_dir}")