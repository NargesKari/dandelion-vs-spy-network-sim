"""
Phase 2 Orchestrator.

Important design note: In phase 2, the packet dissemination protocol is
still the public broadcast (Flood) from phase 1. A subset of nodes is considered
"spies" and their observations are analyzed after the run to guess the origin.

Order of operations (topology -> spy selection -> simulation):
select_bribed_nodes() needs the topology to decide which nodes are worth bribing.
The topology is generated exactly once and verified before running the simulation.

Section 2-1-b of the spec also requires determining the OPTIMAL number of
bribed nodes according to Score_adv = Accuracy / N_spies (more spies is not
automatically better, since the denominator grows too). We do this without
any extra simulation runs: the network is simulated once with the full
30%-budget spy set "listening", and every candidate spy-set size k <= max_k
is then scored by replaying that single log (see adversary.evaluate_spy_count_curve).
The full-budget (30%) result stays the headline "Score_adv table" so it
remains directly comparable with Phases 4 and 5, which reuse the same fixed
spy set; the optimal-k analysis is reported alongside it as required.
"""

import statistics
from pathlib import Path

from adversary import (build_reference_profiles, compute_score_honest,
                        evaluate_attack_all_methods, evaluate_spy_count_curve,
                        plot_spy_selection, rank_spy_candidates, select_bribed_nodes)
from phase1_simulator import build_node_configs, compute_t80, run_phase1
from topology import generate_topology
from config import seed_int, TOPOLOGY_SEED, NUM_PACKETS, BUDGET_FRACTION, OUTPUTS_DIR


def run_phase2(seed: int, num_packets: int = 200, budget_fraction: float = 0.3,
               log_path: str = "phase2_log.jsonl", verbose_spy_selection: bool = False):
    Path(log_path).write_text("")

    # 1. Generate the topology exactly once.
    topo = generate_topology(seed)

    # VERIFICATION: Ensure the topology is consistent and within project limits (20-30)
    actual_nodes = topo.graph.number_of_nodes()
    assert 20 <= actual_nodes <= 30, f"Topology verification failed: {actual_nodes} nodes generated, expected between 20 and 30."

    configs, node_ids, addr_of = build_node_configs(topo)

    # 2. Select the (fixed, 30%-budget) spy set before running the simulation.
    #    This is the set reused as-is by Phases 4 and 5 for cross-phase comparability.
    spy_indices = select_bribed_nodes(topo, budget_fraction=budget_fraction,
                                       verbose=verbose_spy_selection)
    spy_ids = {node_ids[i] for i in spy_indices}

    # 3. Offline profiling phase (Vector Profiling attack)
    profile_data = build_reference_profiles(topo, spy_ids)

    # 4. Run the simulation using deterministic integration seed.
    #    spy_delay_enabled is intentionally left at its default (False): intentional
    #    delay is a Phase-5-only behavior, never applied here even though a spy set exists.
    run_seed = seed_int("sim", "phase2", seed)
    run_phase1(
        seed=seed,
        num_packets=num_packets,
        log_path=log_path,
        topo=topo,
        spy_ids=spy_ids,
        run_seed=run_seed,
    )

    # 5. Runtime evaluation of every method, at the fixed 30% budget.
    result = evaluate_attack_all_methods(log_path, spy_ids, topo=topo, profile_data=profile_data)

    # 6. Determine the optimal number of bribed nodes per Score_adv (Section 2-1-b),
    #    by replaying the SAME log restricted to increasing spy-set sizes — no re-run.
    ranked_int = rank_spy_candidates(topo, max_k=len(spy_indices))
    ranked_ids = [node_ids[n] for n in ranked_int]
    spy_count_curve = evaluate_spy_count_curve(log_path, ranked_ids, topo)
    best = max(spy_count_curve, key=lambda c: c["score_adv_proposed"])
    result["spy_count_curve"] = spy_count_curve
    result["optimal_k"] = best["k"]
    result["optimal_k_spy_ids"] = best["spy_ids"]
    result["optimal_k_accuracy_proposed"] = best["accuracy_proposed"]
    result["optimal_k_score_adv_proposed"] = best["score_adv_proposed"]

    # 7. Score_honest (Section 3-1-b): 1/T80% * (1 - Adversary Detection Rate).
    #    The network protocol here is still Flood, so T80% is scenario-wide (not
    #    method-specific); Adversary Detection Rate is each method's accuracy.
    t80s, _n_origins = compute_t80(log_path, actual_nodes)
    t80_mean_s = statistics.mean(t80s) if t80s else None
    result["t80_mean_s"] = t80_mean_s
    result["score_honest_baseline"] = compute_score_honest(t80_mean_s, result["accuracy_baseline"])
    result["score_honest_proposed"] = compute_score_honest(t80_mean_s, result["accuracy_proposed"])

    return topo, spy_ids, result


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from pathlib import Path

    seed_val = TOPOLOGY_SEED

    # 1. Create master output directories for Phase 2
    base_out_dir = Path(OUTPUTS_DIR)
    log_dir = base_out_dir / "logs"
    phase2_out_dir = base_out_dir / "phase2"

    log_dir.mkdir(parents=True, exist_ok=True)
    phase2_out_dir.mkdir(parents=True, exist_ok=True)

    log_path = log_dir / f"phase2_seed{seed_val}.jsonl"

    # 2. Run Phase 2
    topo, spy_ids, result = run_phase2(
        seed=seed_val,
        num_packets=NUM_PACKETS,
        budget_fraction=BUDGET_FRACTION,
        log_path=str(log_path),
        verbose_spy_selection=True,
    )

    # 3. Format the results string
    results_str = f"=== PHASE 2 SIMULATION RESULTS (Seed: {seed_val}) ===\n"
    results_str += f"Nodes = {topo.graph.number_of_nodes()}, Bribed spies (30% budget) = {result['n_spies']} "
    results_str += f"({', '.join(sorted(spy_ids))})\n"
    results_str += f"Total packets = {result['total_packets']}, Observed by >=1 spy = {result['observed_by_spies']}\n"
    if result["t80_mean_s"] is not None:
        results_str += f"T_80% (mean) = {result['t80_mean_s'] * 1000:.1f} ms\n"
    results_str += "\n"

    results_str += "--- ACCURACY (at 30% budget) ---\n"
    results_str += f"Baseline (one-hop backtrack)         : {result['accuracy_baseline']:.3f}\n"
    results_str += f"Proposed (Hybrid Attack, new)        : {result['accuracy_proposed']:.3f}\n"
    if "accuracy_common_ancestor" in result:
        results_str += f"Common-ancestor (older method)       : {result['accuracy_common_ancestor']:.3f}\n"
    if "accuracy_multilateration" in result:
        results_str += f"Multilateration (TDOA)                : {result['accuracy_multilateration']:.3f}\n"
    results_str += "\n"

    results_str += "--- ADVERSARY SCORE (Score_adv, at 30% budget) ---\n"
    results_str += f"Baseline (one-hop backtrack)         : {result['score_adv_baseline']:.4f}\n"
    results_str += f"Proposed (Hybrid Attack, new)        : {result['score_adv_proposed']:.4f}\n"
    if "score_adv_common_ancestor" in result:
        results_str += f"Common-ancestor (older method)       : {result['score_adv_common_ancestor']:.4f}\n"
    if "score_adv_multilateration" in result:
        results_str += f"Multilateration (TDOA)                : {result['score_adv_multilateration']:.4f}\n"
    results_str += "\n"

    results_str += "--- SCORE_HONEST (Section 3-1-b) ---\n"
    if result["score_honest_baseline"] is not None:
        results_str += f"vs Baseline guess                    : {result['score_honest_baseline']:.4f}\n"
    if result["score_honest_proposed"] is not None:
        results_str += f"vs Proposed guess                    : {result['score_honest_proposed']:.4f}\n"
    results_str += "\n"

    results_str += "--- OPTIMAL SPY COUNT (Section 2-1-b) ---\n"
    results_str += (
        f"Score_adv-optimal k = {result['optimal_k']} (out of {result['n_spies']} allowed by 30% budget)\n"
        f"  accuracy_proposed @ optimal k = {result['optimal_k_accuracy_proposed']:.3f}\n"
        f"  score_adv_proposed @ optimal k = {result['optimal_k_score_adv_proposed']:.4f}\n"
    )

    print(results_str)

    # 4. Save textual outputs to a summary file
    summary_filename = phase2_out_dir / f"phase2_summary_seed{seed_val}.txt"
    with open(summary_filename, "w", encoding="utf-8") as f:
        f.write(results_str)

    # 5. Visualize which nodes were picked as spies
    node_ids_by_int = {n: f"n{n}" for n in topo.graph.nodes}
    spy_plot_filename = phase2_out_dir / f"spy_selection_seed{seed_val}.png"
    plot_spy_selection(topo, spy_ids, node_ids_by_int, str(spy_plot_filename))

    # 6. Comparative bar chart across all available methods (at 30% budget)
    labels = ['Baseline', 'Proposed\n(Hybrid Attack)']
    accuracies = [result['accuracy_baseline'], result['accuracy_proposed']]
    scores = [result['score_adv_baseline'], result['score_adv_proposed']]

    if "accuracy_common_ancestor" in result:
        labels.append('Common-\nAncestor')
        accuracies.append(result['accuracy_common_ancestor'])
        scores.append(result['score_adv_common_ancestor'])
    if "accuracy_multilateration" in result:
        labels.append('Multi-\nlateration')
        accuracies.append(result['accuracy_multilateration'])
        scores.append(result['score_adv_multilateration'])

    x = list(range(len(labels)))
    width = 0.35
    x_acc = [i - width / 2 for i in x]
    x_score = [i + width / 2 for i in x]

    fig, ax1 = plt.subplots(figsize=(9, 6))

    rects1 = ax1.bar(x_acc, accuracies, width, label='Accuracy', color='royalblue', edgecolor='black')
    ax1.set_ylabel('Accuracy (Rate)', color='royalblue', fontweight='bold')
    ax1.tick_params(axis='y', labelcolor='royalblue')
    ax1.set_ylim(0, max(accuracies) * 1.3)

    ax2 = ax1.twinx()
    rects2 = ax2.bar(x_score, scores, width, label='Score_adv', color='darkorange', edgecolor='black')
    ax2.set_ylabel('Score_adv (Score / Spy)', color='darkorange', fontweight='bold')
    ax2.tick_params(axis='y', labelcolor='darkorange')
    ax2.set_ylim(0, max(scores) * 1.3)

    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontweight='bold', fontsize=11)
    plt.title(f"Phase 2 Attack Performance Comparison (Seed {seed_val})", fontweight='bold')

    def autolabel(rects, ax, is_score=False):
        for rect in rects:
            height = rect.get_height()
            format_str = f'{height:.4f}' if is_score else f'{height:.3f}'
            ax.annotate(format_str,
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 5),
                        textcoords="offset points",
                        ha='center', va='bottom', fontweight='bold')

    autolabel(rects1, ax1, is_score=False)
    autolabel(rects2, ax2, is_score=True)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    plot_filename = phase2_out_dir / f"attack_performance_seed{seed_val}.png"
    plt.savefig(str(plot_filename), dpi=150, bbox_inches='tight')
    plt.close()

    # 7. Score_adv vs. spy count k — shows WHY the optimal k is chosen (Section 2-1-b)
    curve = result["spy_count_curve"]
    ks = [c["k"] for c in curve]
    score_adv_k = [c["score_adv_proposed"] for c in curve]
    accuracy_k = [c["accuracy_proposed"] for c in curve]

    fig, ax1 = plt.subplots(figsize=(8, 6))
    ax1.plot(ks, score_adv_k, marker='o', color='darkorange', label='Score_adv (proposed)')
    ax1.axvline(result['optimal_k'], color='darkorange', linestyle='dashed', linewidth=1.5,
                label=f"optimal k = {result['optimal_k']}")
    ax1.set_xlabel('Number of bribed (spy) nodes, k')
    ax1.set_ylabel('Score_adv', color='darkorange', fontweight='bold')
    ax1.tick_params(axis='y', labelcolor='darkorange')

    ax2 = ax1.twinx()
    ax2.plot(ks, accuracy_k, marker='s', color='royalblue', label='Accuracy (proposed)')
    ax2.set_ylabel('Accuracy', color='royalblue', fontweight='bold')
    ax2.tick_params(axis='y', labelcolor='royalblue')

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='lower right')
    plt.title(f"Phase 2: Score_adv vs. Spy Count (Seed {seed_val})", fontweight='bold')
    ax1.grid(axis='y', alpha=0.4)

    plot_filename = phase2_out_dir / f"score_adv_vs_k_seed{seed_val}.png"
    plt.savefig(str(plot_filename), dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Outputs successfully routed: Texts/Images -> '{phase2_out_dir}', Logs -> '{log_dir}'")
