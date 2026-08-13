"""
Phase 3 — Parameter p sweep for Dandelion.

Experimental Design (and rationale):
  - `TOPOLOGY_SEED` and `ORIGIN_SEED` are kept constant for all runs (all three p values and all
    5 iterations). This means all runs are executed on the exact same topology and the exact same
    sequence of "which node generates a packet and when". This is necessary so that differences
    in results between p=0.9 and p=0.1 are truly due to p itself, not due to randomness in the
    topology or packet injection scheduling.
  - The only thing that changes among the 5 iterations of each p is the `run_seed`: This seed
    controls both the link delay jitter and the probabilistic coin flip for "continue Stem or
    convert to Fluff". Changing it introduces variance in the network behavior, which is exactly
    what we need to calculate the mean/median/standard deviation.

Two families of metrics are reported:
  - `avg_coverage`: The average fraction of nodes that received each packet
    (0 to 1). This metric is more important than T_80% because in a real Dandelion network, a
    packet might not reach everywhere at all (explanation below).
  - `T_80%`: The time to reach 80% of the nodes, only for packets that actually reached it.

Important and unexpected finding observed in tests: Because the Stem path in this implementation
is a *random walk* on the graph (not a fixed predetermined path like the original Dandelion),
the Stem path might return to a node that has already seen the same packet. According to the
SeenSet, that node ignores the packet and does not forward it at all — meaning the packet "dies"
without ever turning into Fluff and does not reach the rest of the network. The larger p is,
the longer the Stem path, and the higher the probability of this collision. Therefore, we expect
p=0.9 to have a lower average coverage compared to p=0.1 — this is a real and reportable finding,
not a bug.
"""

import statistics
from pathlib import Path

from phase1_simulator import compute_t80, run_phase1
from sim_log import read_log
from config import TOPOLOGY_SEED, ORIGIN_SEED, P_VALUES, RUNS_PER_P, NUM_PACKETS, OUTPUTS_DIR

SETTLE_TIME_S = 6.0
LOG_DIR = f"{OUTPUTS_DIR}/logs/phase3"


def coverage_fractions(log_path: str, total_nodes: int):
    events = read_log(log_path)
    receives = {}
    for e in events:
        if e["event"] == "receive":
            receives.setdefault(e["packet_id"], set()).add(e["node_id"])
    return [len(nodes) / total_nodes for nodes in receives.values()]


def run_one(p: float, run_idx: int):
    run_seed = int(p * 1000) * 100 + run_idx  # Each (p, run_idx) combination gets a unique, reproducible seed
    log_path = f"{LOG_DIR}/p{p}_run{run_idx}.jsonl"

    topo, _node_ids = run_phase1(
        seed=run_seed, num_packets=NUM_PACKETS, log_path=log_path,
        settle_time_s=SETTLE_TIME_S, stem_p=p,
        topology_seed=TOPOLOGY_SEED, origin_seed=ORIGIN_SEED, run_seed=run_seed,
    )
    total_nodes = topo.graph.number_of_nodes()
    cov = coverage_fractions(log_path, total_nodes)
    t80s, _ = compute_t80(log_path, total_nodes)

    return {
        "avg_coverage": statistics.mean(cov) if cov else 0.0,
        "full_coverage_frac": (sum(1 for c in cov if c >= 0.999) / len(cov)) if cov else 0.0,
        "reached80_frac": len(t80s) / NUM_PACKETS,
        "t80_ms": [t * 1000 for t in t80s],
    }


def run_sweep():
    Path(LOG_DIR).mkdir(parents=True, exist_ok=True)
    results = {}
    for p in P_VALUES:
        results[p] = [run_one(p, i) for i in range(RUNS_PER_P)]
    return results


def summarize(results) -> str:
    lines = [f"{'p':>5} | {'avg_coverage m/md/sd':>22} | {'full_cov_frac':>13} | {'reached80_frac':>14} | {'T_80% ms m/md/sd':>20}"]
    lines.append("-" * len(lines[0]))
    for p, runs in results.items():
        avg_cov = [r["avg_coverage"] for r in runs]
        full_cov = [r["full_coverage_frac"] for r in runs]
        r80 = [r["reached80_frac"] for r in runs]
        all_t80 = [t for r in runs for t in r["t80_ms"]]

        cov_str = f"{statistics.mean(avg_cov):.3f}/{statistics.median(avg_cov):.3f}/{statistics.pstdev(avg_cov):.3f}"
        t80_str = (
            f"{statistics.mean(all_t80):.0f}/{statistics.median(all_t80):.0f}/{statistics.pstdev(all_t80):.0f}"
            if all_t80 else "n/a"
        )
        lines.append(
            f"{p:>5} | {cov_str:>22} | {statistics.mean(full_cov):>13.3f} | "
            f"{statistics.mean(r80):>14.3f} | {t80_str:>20}"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from topology import generate_topology

    phase3_out_dir = Path(OUTPUTS_DIR) / "phase3"
    phase3_out_dir.mkdir(parents=True, exist_ok=True)

    # The same topology from Phase 1/2 (same TOPOLOGY_SEED from config.py) -- saved/plotted
    # to ensure the constraint "topology must remain constant across all 5 phases" is met.
    topo = generate_topology(TOPOLOGY_SEED)
    topo.plot(str(phase3_out_dir / f"topology_seed{TOPOLOGY_SEED}.png"))

    results = run_sweep()
    summary_text = summarize(results)
    print(summary_text)

    with open(phase3_out_dir / "phase3_summary.txt", "w", encoding="utf-8") as f:
        f.write(f"TOPOLOGY_SEED={TOPOLOGY_SEED}  ORIGIN_SEED={ORIGIN_SEED}  "
                f"NUM_PACKETS={NUM_PACKETS}  RUNS_PER_P={RUNS_PER_P}\n\n")
        f.write(summary_text + "\n")

    # --- Chart 1: Average coverage (avg_coverage) vs. p (boxplot over 5 runs) ---
    fig, ax = plt.subplots(figsize=(7, 5))
    data = [[r["avg_coverage"] for r in results[p]] for p in P_VALUES]
    ax.boxplot(data, tick_labels=[f"p={p}" for p in P_VALUES])
    ax.set_ylabel("Average Fraction of Covered Nodes")
    ax.set_title(f"Phase 3: Network Coverage vs. p (Dandelion), {RUNS_PER_P} Runs per p")
    ax.grid(axis="y", alpha=0.4)
    fig.savefig(str(phase3_out_dir / "coverage_vs_p.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # --- Chart 2: T_80% (ms) vs. p (boxplot over all packets in 5 runs) ---
    fig, ax = plt.subplots(figsize=(7, 5))
    data_t80 = [[t for r in results[p] for t in r["t80_ms"]] for p in P_VALUES]
    ax.boxplot(data_t80, tick_labels=[f"p={p}" for p in P_VALUES])
    ax.set_ylabel("T_80% (milliseconds)")
    ax.set_title("Phase 3: Time to Reach 80% of Nodes vs. p")
    ax.grid(axis="y", alpha=0.4)
    fig.savefig(str(phase3_out_dir / "t80_vs_p.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Outputs saved to: {phase3_out_dir}")