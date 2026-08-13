"""
Phase 4 — Advanced Attack on Dandelion.

This script reuses the same logs from Phase 3 (phase3_logs/p{p}_run{i}.jsonl) —
because the reference log already contains all the necessary information (state,
sender_peer_id for each receiver), and determining "which node is a spy"
is just a filter applied to the same data after running the simulation.
Therefore, there is no need to run the network simulation again.

For each p, we analyze all 5 runs using three methods:
  - baseline_guess   (Phase 2, unchanged)
  - proposed_guess   (Phase 2, 1-hop backtrack regardless of STEM/FLUFF)
  - phase4_guess     (New, 1-hop backtrack limited to STEM observations)

We then report the mean/median/standard deviation of accuracy and Score_adv over the 5 runs.
"""

import statistics

from adversary import baseline_guess, evaluate_attack, phase4_guess, proposed_guess, select_bribed_nodes
from phase3_simulator import LOG_DIR, P_VALUES, RUNS_PER_P
from phase1_simulator import build_node_configs
from topology import generate_topology
from config import TOPOLOGY_SEED, BUDGET_FRACTION, OUTPUTS_DIR

METHODS = {
    "baseline (phase2)": baseline_guess,
    "proposed (phase2)": proposed_guess,
    "phase4 (STEM-only backtrack)": phase4_guess,
}


def run_phase4_analysis(budget_fraction: float = BUDGET_FRACTION):
    topo = generate_topology(TOPOLOGY_SEED)  # The same topology used in Phase 3 (config.py)
    node_ids, _addr_of = None, None
    configs, node_ids, addr_of = build_node_configs(topo)

    spy_indices = select_bribed_nodes(topo, budget_fraction=budget_fraction)
    spy_ids = {node_ids[i] for i in spy_indices}

    results = {}  # p -> method -> list of per-run dicts
    for p in P_VALUES:
        results[p] = {name: [] for name in METHODS}
        for run_idx in range(RUNS_PER_P):
            log_path = f"{LOG_DIR}/p{p}_run{run_idx}.jsonl"
            for name, guess_fn in METHODS.items():
                # Important note (bug fix): topo=topo must always be passed, otherwise
                # proposed_guess and phase4_guess fall back to baseline_guess due to topo=None
                # and their advanced method is practically never executed --
                # making the Phase 4 comparison meaningless.
                r = evaluate_attack(log_path, spy_ids, guess_fn, topo=topo)
                results[p][name].append(r)

    return topo, spy_ids, results


def summarize(results) -> str:
    lines = []
    for p, methods in results.items():
        lines.append(f"--- p = {p} ---")
        for name, runs in methods.items():
            acc = [r["accuracy"] for r in runs]
            score = [r["score_adv"] for r in runs]
            lines.append(
                f"  {name:<30} acc(m/md/sd)={statistics.mean(acc):.3f}/"
                f"{statistics.median(acc):.3f}/{statistics.pstdev(acc):.3f}   "
                f"Score_adv(m/md/sd)={statistics.mean(score):.4f}/"
                f"{statistics.median(score):.4f}/{statistics.pstdev(score):.4f}"
            )
    return "\n".join(lines)


if __name__ == "__main__":
    from pathlib import Path
    import matplotlib.pyplot as plt

    phase4_out_dir = Path(OUTPUTS_DIR) / "phase4"
    phase4_out_dir.mkdir(parents=True, exist_ok=True)

    topo, spy_ids, results = run_phase4_analysis(budget_fraction=BUDGET_FRACTION)
    topo.plot(str(phase4_out_dir / f"topology_seed{TOPOLOGY_SEED}.png"))

    header = f"spies ({len(spy_ids)}): {', '.join(sorted(spy_ids))}\n"
    summary_text = header + summarize(results)
    print(summary_text)

    with open(phase4_out_dir / "phase4_summary.txt", "w", encoding="utf-8") as f:
        f.write(summary_text + "\n")

    # Chart: Accuracy of all three methods vs. p (mean over 5 runs)
    method_names = list(METHODS.keys())
    x = list(range(len(P_VALUES)))
    width = 0.25
    fig, ax = plt.subplots(figsize=(8, 6))
    for i, name in enumerate(method_names):
        means = [statistics.mean([r["accuracy"] for r in results[p][name]]) for p in P_VALUES]
        stdevs = [statistics.pstdev([r["accuracy"] for r in results[p][name]]) for p in P_VALUES]
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

    # Chart: Score_adv of all three methods vs. p
    fig, ax = plt.subplots(figsize=(8, 6))
    for i, name in enumerate(method_names):
        means = [statistics.mean([r["score_adv"] for r in results[p][name]]) for p in P_VALUES]
        stdevs = [statistics.pstdev([r["score_adv"] for r in results[p][name]]) for p in P_VALUES]
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

    print(f"Outputs saved to: {phase4_out_dir}")