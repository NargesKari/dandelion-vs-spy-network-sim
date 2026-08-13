"""
Phase 4 — Advanced Attack on Dandelion.

This script reuses the same logs from Phase 3 to evaluate the advanced attack.
Since the reference log already contains all necessary state and sender information,
determining which node is a spy is merely a filter applied post-simulation.
There is no need to re-run the network simulation.

For each probability p, we analyze all 5 runs using three methods:
  - baseline_guess   (Phase 2, unchanged)
  - proposed_guess   (Phase 2, 1-hop backtrack regardless of STEM/FLUFF)
  - phase4_guess     (New, 1-hop backtrack limited exclusively to STEM observations)

We then report the mean, median, and standard deviation of accuracy, Score_adv,
and Score_honest (Section 3-1 of the spec).

Note: Phase 3 now runs on the real per-node-process architecture (see
phase3_simulator.py) and already excludes the fixed 30%-budget spy set from
ever being chosen as a packet origin, so these logs are unbiased for attack
evaluation (a packet's true origin is never itself one of the spy nodes).
"""

import statistics
from pathlib import Path
import matplotlib.pyplot as plt

from adversary import (baseline_guess, compute_score_honest, evaluate_attack,
                        phase4_guess, proposed_guess, select_bribed_nodes)
from phase1_simulator import build_node_configs, compute_t80
from topology import generate_topology
from config import TOPOLOGY_SEED, BUDGET_FRACTION, OUTPUTS_DIR, P_VALUES, RUNS_PER_P

# Point to Phase 3 logs explicitly
PHASE3_LOG_DIR = Path(OUTPUTS_DIR) / "logs" / "phase3"

METHODS = {
    "baseline (phase2)": baseline_guess,
    "proposed (phase2)": proposed_guess,
    "phase4 (STEM-only backtrack)": phase4_guess,
}


def run_phase4_analysis(budget_fraction: float = BUDGET_FRACTION):
    """Run analysis for Phase 4 using existing Phase 3 logs."""
    topo = generate_topology(TOPOLOGY_SEED)
    configs, node_ids, addr_of = build_node_configs(topo)

    spy_indices = select_bribed_nodes(topo, budget_fraction=budget_fraction)
    spy_ids = {node_ids[i] for i in spy_indices}
    total_nodes = topo.graph.number_of_nodes()

    results = {}
    for p in P_VALUES:
        results[p] = {name: [] for name in METHODS}
        results[p]["t80_mean_ms"] = []
        for run_idx in range(RUNS_PER_P):
            log_path = PHASE3_LOG_DIR / f"p{p:.1f}_run{run_idx}.jsonl"

            # Ensure the log file exists before attempting to analyze
            if not log_path.exists():
                print(f"Warning: Log file {log_path} not found. Ensure Phase 3 has been executed.")
                continue

            t80s, _n_origins = compute_t80(str(log_path), total_nodes)
            t80_mean_s = statistics.mean(t80s) if t80s else None
            if t80_mean_s is not None:
                results[p]["t80_mean_ms"].append(t80_mean_s * 1000)

            for name, guess_fn in METHODS.items():
                r = evaluate_attack(str(log_path), spy_ids, guess_fn, topo=topo)
                r["score_honest"] = compute_score_honest(t80_mean_s, r["accuracy"])
                results[p][name].append(r)

    return topo, spy_ids, results


def summarize(results) -> str:
    """Generate a formatted summary string of the attack performance."""
    lines = []
    for p, methods in results.items():
        lines.append(f"--- p = {p} ---")
        t80s = methods.get("t80_mean_ms", [])
        if t80s:
            lines.append(f"  T_80% (mean over runs) = {statistics.mean(t80s):.1f} ms")
        for name in METHODS:
            runs = methods[name]
            if not runs:
                continue

            acc = [r["accuracy"] for r in runs]
            score = [r["score_adv"] for r in runs]
            honest = [r["score_honest"] for r in runs if r["score_honest"] is not None]
            line = (
                f"  {name:<30} acc(m/md/sd)={statistics.mean(acc):.3f}/"
                f"{statistics.median(acc):.3f}/{statistics.pstdev(acc):.3f}   "
                f"Score_adv(m/md/sd)={statistics.mean(score):.4f}/"
                f"{statistics.median(score):.4f}/{statistics.pstdev(score):.4f}"
            )
            if honest:
                line += f"   Score_honest(mean)={statistics.mean(honest):.4f}"
            lines.append(line)
    return "\n".join(lines)


if __name__ == "__main__":
    phase4_out_dir = Path(OUTPUTS_DIR) / "phase4"
    phase4_out_dir.mkdir(parents=True, exist_ok=True)

    topo, spy_ids, results = run_phase4_analysis(budget_fraction=BUDGET_FRACTION)
    topo.plot(str(phase4_out_dir / f"topology_seed{TOPOLOGY_SEED}.png"))

    header = f"spies ({len(spy_ids)}): {', '.join(sorted(spy_ids))}\n"
    summary_text = header + summarize(results)
    print(summary_text)

    with open(phase4_out_dir / "phase4_summary.txt", "w", encoding="utf-8") as f:
        f.write(summary_text + "\n")

    # Chart 1: Accuracy Comparison
    method_names = list(METHODS.keys())
    x = list(range(len(P_VALUES)))
    width = 0.25
    fig, ax = plt.subplots(figsize=(8, 6))

    for i, name in enumerate(method_names):
        # Calculate statistics, ensuring we have data
        means = [statistics.mean([r["accuracy"] for r in results[p][name]]) if results[p][name] else 0 for p in P_VALUES]
        stdevs = [statistics.pstdev([r["accuracy"] for r in results[p][name]]) if results[p][name] else 0 for p in P_VALUES]
        offsets = [xi + (i - 1) * width for xi in x]
        ax.bar(offsets, means, width, yerr=stdevs, capsize=4, label=name)

    ax.set_xticks(x)
    ax.set_xticklabels([f"p={p}" for p in P_VALUES])
    ax.set_ylabel("Source Estimation Accuracy")
    ax.set_title("Phase 4: Attack Accuracy Comparison on Dandelion vs. p")
    ax.legend()
    ax.grid(axis="y", alpha=0.4)
    fig.savefig(str(phase4_out_dir / "accuracy_comparison_vs_p.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Chart 2: Score_adv Comparison
    fig, ax = plt.subplots(figsize=(8, 6))

    for i, name in enumerate(method_names):
        means = [statistics.mean([r["score_adv"] for r in results[p][name]]) if results[p][name] else 0 for p in P_VALUES]
        stdevs = [statistics.pstdev([r["score_adv"] for r in results[p][name]]) if results[p][name] else 0 for p in P_VALUES]
        offsets = [xi + (i - 1) * width for xi in x]
        ax.bar(offsets, means, width, yerr=stdevs, capsize=4, label=name)

    ax.set_xticks(x)
    ax.set_xticklabels([f"p={p}" for p in P_VALUES])
    ax.set_ylabel("Score_adv")
    ax.set_title("Phase 4: Attack Score_adv Comparison on Dandelion vs. p")
    ax.legend()
    ax.grid(axis="y", alpha=0.4)
    fig.savefig(str(phase4_out_dir / "score_adv_comparison_vs_p.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Chart 3: Score_honest Comparison (Section 3-1-b)
    fig, ax = plt.subplots(figsize=(8, 6))

    for i, name in enumerate(method_names):
        vals_by_p = [[r["score_honest"] for r in results[p][name] if r["score_honest"] is not None] for p in P_VALUES]
        means = [statistics.mean(v) if v else 0 for v in vals_by_p]
        stdevs = [statistics.pstdev(v) if len(v) > 1 else 0 for v in vals_by_p]
        offsets = [xi + (i - 1) * width for xi in x]
        ax.bar(offsets, means, width, yerr=stdevs, capsize=4, label=name)

    ax.set_xticks(x)
    ax.set_xticklabels([f"p={p}" for p in P_VALUES])
    ax.set_ylabel("Score_honest")
    ax.set_title("Phase 4: Network's Score_honest vs. p (per attack method faced)")
    ax.legend()
    ax.grid(axis="y", alpha=0.4)
    fig.savefig(str(phase4_out_dir / "score_honest_comparison_vs_p.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Outputs saved to: {phase4_out_dir}")
