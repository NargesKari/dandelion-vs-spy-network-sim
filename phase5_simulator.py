"""
Phase 5 — Applying intentional delay restrictions by spy nodes.

Phase 5 Question (based on the project description):
  "Does this behavior [intentional delay by spies] increase or decrease the attacker's
  joint estimation accuracy? And what is its effect on T_80%?"

Experimental Design:
  - The exact same topology and the same 9 spies (30%) from Phases 2 to 4 are used.
  - For each p in {0.9, 0.5, 0.1} and each of the 5 independent runs, we execute the 
    network TWICE with perfectly identical initial deterministic seeds:
      1) NO-DELAY  : spy_ids=None  -> Spies do not apply any intentional delay.
      2) WITH-DELAY: spy_ids=<spies> -> Spies apply an intentional delay.
    Since the intentional delay is drawn from an RNG completely independent of the 
    node's main rng, the only difference between the two runs is the intentional delay itself.
"""

import statistics
from pathlib import Path
import matplotlib.pyplot as plt

from adversary import baseline_guess, evaluate_attack, phase4_guess, proposed_guess, select_bribed_nodes
from phase1_simulator import build_node_configs, compute_t80, run_phase1
from sim_log import read_log
from topology import generate_topology
from config import (TOPOLOGY_SEED, ORIGIN_SEED, NUM_PACKETS, P_VALUES, RUNS_PER_P,
                    BUDGET_FRACTION, OUTPUTS_DIR, seed_int)

LOG_DIR = Path(OUTPUTS_DIR) / "logs" / "phase5"
SETTLE_TIME_S = 10.0  # Greater than phase 3 because intentional delay slows down propagation

METHODS = {
    "baseline (phase2)": baseline_guess,
    "proposed (phase2)": proposed_guess,
    "phase4 (STEM-only)": phase4_guess,
}

CONDITIONS = ("nodelay", "delay")


def coverage_fractions(log_path: str, total_nodes: int):
    """Standalone helper to calculate coverage from a log file."""
    events = read_log(log_path)
    receives = {}
    for e in events:
        if e["event"] in ("receive", "forward"):
            receives.setdefault(e["packet_id"], set()).add(e["node_id"])
    return [len(nodes) / total_nodes for nodes in receives.values()]


def spy_delay_stats(log_path: str):
    """Descriptive statistics on the applied intentional delays themselves."""
    events = [e for e in read_log(log_path) if e["event"] == "spy_delay"]
    vals = [e["intentional_delay_ms"] for e in events]
    if not vals:
        return {"n": 0, "mean_ms": 0.0, "max_ms": 0.0}
    return {"n": len(vals), "mean_ms": statistics.mean(vals), "max_ms": max(vals)}


def run_one(p: float, run_idx: int, spy_ids, condition: str):
    """Execute a single scenario for a specific probability, run index, and delay condition."""
    assert condition in CONDITIONS
    
    # Use deterministic hash-based seed to exactly match Phase 3 network execution
    run_seed = seed_int("sim", p, run_idx)
    log_path = LOG_DIR / f"p{p:.1f}_run{run_idx}_{condition}.jsonl"

    topo, node_ids = run_phase1(
        seed=run_seed, num_packets=NUM_PACKETS, log_path=str(log_path),
        settle_time_s=SETTLE_TIME_S, stem_p=p,
        topology_seed=TOPOLOGY_SEED, origin_seed=ORIGIN_SEED, run_seed=run_seed,
        spy_ids=(spy_ids if condition == "delay" else None),
    )
    
    total_nodes = topo.graph.number_of_nodes()
    cov = coverage_fractions(str(log_path), total_nodes)
    t80s, _n_origins = compute_t80(str(log_path), total_nodes)

    attack = {name: evaluate_attack(str(log_path), spy_ids, fn, topo=topo) for name, fn in METHODS.items()}

    return {
        "avg_coverage": statistics.mean(cov) if cov else 0.0,
        "reached80_frac": len(t80s) / NUM_PACKETS,
        "t80_ms": [t * 1000 for t in t80s],
        "attack": attack,
        "delay_stats": spy_delay_stats(str(log_path)),
    }


def run_sweep(budget_fraction: float = BUDGET_FRACTION):
    """Execute the parameter sweep for Phase 5."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    topo = generate_topology(TOPOLOGY_SEED)
    _configs, node_ids, _addr_of = build_node_configs(topo)
    spy_indices = select_bribed_nodes(topo, budget_fraction=budget_fraction)
    spy_ids = {node_ids[i] for i in spy_indices}

    results = {p: {c: [] for c in CONDITIONS} for p in P_VALUES}
    for p in P_VALUES:
        for run_idx in range(RUNS_PER_P):
            for condition in CONDITIONS:
                results[p][condition].append(run_one(p, run_idx, spy_ids, condition))

    return spy_ids, results


def _fmt(vals):
    """Format statistical values."""
    if not vals:
        return "n/a"
    return f"{statistics.mean(vals):.3f}/{statistics.median(vals):.3f}/{statistics.pstdev(vals):.3f}"


def summarize(results) -> str:
    """Format a textual summary of the delay analysis."""
    lines = []
    for p, by_cond in results.items():
        lines.append(f"=== p = {p} ===")
        for condition in CONDITIONS:
            runs = by_cond[condition]
            all_t80 = [t for r in runs for t in r["t80_ms"]]
            avg_cov = [r["avg_coverage"] for r in runs]
            r80 = [r["reached80_frac"] for r in runs]
            label = "WITH intentional delay" if condition == "delay" else "NO delay (baseline)"
            
            lines.append(f"  --- {label} ---")
            lines.append(
                f"    avg_coverage(m/md/sd)={_fmt(avg_cov)}   reached80_frac(avg)={statistics.mean(r80):.3f}   "
                f"T_80%ms(m/md/sd)={_fmt(all_t80) if all_t80 else 'n/a'}"
            )
            for name in METHODS:
                acc = [r["attack"][name]["accuracy"] for r in runs]
                score = [r["attack"][name]["score_adv"] for r in runs]
                lines.append(
                    f"    {name:<20} acc(m/md/sd)={_fmt(acc)}   Score_adv(m/md/sd)={_fmt(score)}"
                )
            if condition == "delay":
                dstats = [r["delay_stats"] for r in runs]
                mean_delays = [d["mean_ms"] for d in dstats if d["n"] > 0]
                if mean_delays:
                    lines.append(
                        f"    intentional_delay applied: avg={statistics.mean(mean_delays):.1f}ms "
                        f"over {sum(d['n'] for d in dstats)} sends across {RUNS_PER_P} runs"
                    )
        lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    phase5_out_dir = Path(OUTPUTS_DIR) / "phase5"
    phase5_out_dir.mkdir(parents=True, exist_ok=True)

    spy_ids, results = run_sweep(budget_fraction=BUDGET_FRACTION)

    topo = generate_topology(TOPOLOGY_SEED)
    topo.plot(str(phase5_out_dir / f"topology_seed{TOPOLOGY_SEED}.png"))

    header = f"spies ({len(spy_ids)}): {', '.join(sorted(spy_ids))}\n\n"
    summary_text = header + summarize(results)
    print(summary_text)

    with open(phase5_out_dir / "phase5_summary.txt", "w", encoding="utf-8") as f:
        f.write(summary_text + "\n")

    # Chart 1: T_80% with/without intentional delay
    x = list(range(len(P_VALUES)))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 6))
    
    for i, condition in enumerate(CONDITIONS):
        means, stdevs = [], []
        for p in P_VALUES:
            all_t80 = [t for r in results[p][condition] for t in r["t80_ms"]]
            means.append(statistics.mean(all_t80) if all_t80 else 0.0)
            stdevs.append(statistics.pstdev(all_t80) if all_t80 else 0.0)
        offsets = [xi + (i - 0.5) * width for xi in x]
        label = "With Intentional Delay" if condition == "delay" else "Without Intentional Delay"
        ax.bar(offsets, means, width, yerr=stdevs, capsize=4, label=label)
        
    ax.set_xticks(x)
    ax.set_xticklabels([f"p={p}" for p in P_VALUES])
    ax.set_ylabel("T_80% (milliseconds)")
    ax.set_title("Phase 5: Effect of Spy Intentional Delay on T_80%")
    ax.legend()
    ax.grid(axis="y", alpha=0.4)
    fig.savefig(str(phase5_out_dir / "t80_delay_vs_nodelay.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Chart 2: Score_adv of proposed method with/without intentional delay
    method_of_interest = "proposed (phase2)"
    fig, ax = plt.subplots(figsize=(8, 6))
    
    for i, condition in enumerate(CONDITIONS):
        means, stdevs = [], []
        for p in P_VALUES:
            scores = [r["attack"][method_of_interest]["score_adv"] for r in results[p][condition]]
            means.append(statistics.mean(scores))
            stdevs.append(statistics.pstdev(scores))
        offsets = [xi + (i - 0.5) * width for xi in x]
        label = "With Intentional Delay" if condition == "delay" else "Without Intentional Delay"
        ax.bar(offsets, means, width, yerr=stdevs, capsize=4, label=label)
        
    ax.set_xticks(x)
    ax.set_xticklabels([f"p={p}" for p in P_VALUES])
    ax.set_ylabel(f"Score_adv ({method_of_interest})")
    ax.set_title("Phase 5: Effect of Spy Intentional Delay on Attacker's Joint Accuracy")
    ax.legend()
    ax.grid(axis="y", alpha=0.4)
    fig.savefig(str(phase5_out_dir / "score_adv_delay_vs_nodelay.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Outputs saved to: {phase5_out_dir}")