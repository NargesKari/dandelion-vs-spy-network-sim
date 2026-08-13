"""
Phase 3 — Dandelion Protocol with Parameter p Sweep.

IMPORTANT — architecture note: this phase runs on the exact same per-node
independent-process / UDP architecture as Phases 1, 2 and 5 (node_process.py,
orchestrated through phase1_simulator.run_phase1 with stem_p=p). An earlier
version of this file re-implemented Dandelion's Stem/Fluff logic from scratch
as a standalone in-process discrete-event simulation, bypassing Node/Packet/
node_process.py entirely. That violated Section 1 of the spec ("each node is
an independent process on localhost") and, worse, risked silently diverging
from the actual Stem/Fluff behavior implemented in node_process.py (which is
also what Phase 5 uses) — so the Phase-3-vs-Phase-5 "no delay" comparison
would not have actually been comparing the same protocol implementation. This
version removes that duplicate engine: Phase 3 IS the same node_process.py
code path, just swept over p.

Spy handling: Phase 3 itself has no adversary (the spec doesn't introduce one
until Phase 2 for Flood / Phase 4 for Dandelion) — but Phase 4 reuses these
exact logs to evaluate the Dandelion-aware attack, and the spec requires that
packet origins always be honest nodes in every phase. So Phase 3 still picks
the same fixed 30%-budget spy set used by Phases 2/4/5 (deterministic given
the topology, so it's identical across phases without any extra bookkeeping)
purely to exclude those nodes from being origins here; spy_delay_enabled is
left at its default False, so this has zero effect on Phase 3's own
protocol-level results (coverage / T80%) — only Phase 4's later attack
analysis depends on it.
"""

import json
import statistics
from pathlib import Path

import matplotlib.pyplot as plt

from adversary import select_bribed_nodes
from config import (
    TOPOLOGY_SEED, NUM_PACKETS, P_VALUES, RUNS_PER_P, BUDGET_FRACTION,
    OUTPUTS_DIR, ORIGIN_SEED, seed_int
)
from phase1_simulator import build_node_configs, coverage_fractions, compute_t80, run_phase1
from topology import generate_topology

LOG_DIR = Path(OUTPUTS_DIR) / "logs" / "phase3"

# A high p means long single-path Stem chains (average length 1/(1-p), so ~10
# hops at p=0.9), and each hop adds its own real link delay before the packet
# even starts flooding — settling faster than Flood-only Phase 1/2 needs is
# expected, so those phases' shorter default is not reused here.
SETTLE_TIME_S = 8.0


def run_phase3_one_run(p: float, run_idx: int, topo, spy_ids) -> dict:
    """
    Runs one (p, run_idx) scenario through the real multiprocess node
    architecture and reports coverage / T80% statistics.

    run_seed follows the same seed_int("sim", p, run_idx) convention that
    Phase 5 relies on for its NO-DELAY condition to be the exact same
    execution as the corresponding Phase-3 run (down to every node's RNG
    stream) — see phase5_simulator.py.
    """
    run_seed = seed_int("sim", p, run_idx)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"p{p:.1f}_run{run_idx}.jsonl"

    run_phase1(
        seed=run_seed, num_packets=NUM_PACKETS, log_path=str(log_path),
        settle_time_s=SETTLE_TIME_S,
        stem_p=p, topology_seed=TOPOLOGY_SEED, origin_seed=ORIGIN_SEED, run_seed=run_seed,
        spy_ids=spy_ids, topo=topo,
    )

    total_nodes = topo.graph.number_of_nodes()
    cov = coverage_fractions(str(log_path), total_nodes)
    t80s, n_origins = compute_t80(str(log_path), total_nodes)

    return {
        "p": p,
        "run": run_idx,
        "avg_coverage": statistics.mean(cov) if cov else 0.0,
        "std_coverage": statistics.pstdev(cov) if len(cov) > 1 else 0.0,
        "median_coverage": statistics.median(cov) if cov else 0.0,
        "full_coverage_count": sum(1 for c in cov if c >= 0.999),
        "n_origins": n_origins,
        "t80_count": len(t80s),
        "t80_mean_ms": statistics.mean(t80s) * 1000 if t80s else None,
        "t80_median_ms": statistics.median(t80s) * 1000 if t80s else None,
        "t80_stdev_ms": statistics.pstdev(t80s) * 1000 if len(t80s) > 1 else None,
    }


def run_phase3_parameter_sweep(budget_fraction: float = BUDGET_FRACTION):
    topo = generate_topology(TOPOLOGY_SEED)
    actual_nodes = topo.graph.number_of_nodes()
    assert 20 <= actual_nodes <= 30, f"Topology verification failed: {actual_nodes} nodes generated, expected between 20 and 30."

    _configs, node_ids, _addr_of = build_node_configs(topo)
    spy_indices = select_bribed_nodes(topo, budget_fraction=budget_fraction)
    spy_ids = {node_ids[i] for i in spy_indices}

    all_results = {p: [] for p in P_VALUES}
    for p in P_VALUES:
        print(f"\n=== Phase 3: Dandelion with p={p} ===")
        for run_idx in range(RUNS_PER_P):
            result = run_phase3_one_run(p, run_idx, topo, spy_ids)
            all_results[p].append(result)
            t80_str = f"{result['t80_mean_ms']:.0f}ms" if result['t80_mean_ms'] is not None else "N/A"
            print(f"  Run {run_idx+1}: "
                  f"coverage={result['avg_coverage']:.3f} +/- {result['std_coverage']:.3f}, "
                  f"T80%={t80_str}")
    return topo, spy_ids, all_results


def print_phase3_summary(all_results):
    print("\n" + "=" * 100)
    print("PHASE 3 SUMMARY: DANDELION PARAMETER SWEEP")
    print("=" * 100)

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

    topo, spy_ids, results = run_phase3_parameter_sweep()
    print_phase3_summary(results)

    output_file = Path(OUTPUTS_DIR) / "phase3_results.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    phase3_out_dir = Path(OUTPUTS_DIR) / "phase3"
    phase3_out_dir.mkdir(parents=True, exist_ok=True)
    topo.plot(str(phase3_out_dir / f"topology_seed{TOPOLOGY_SEED}.png"))

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
