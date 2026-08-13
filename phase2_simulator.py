"""
Phase 2 Orchestrator.

Important design note: In phase 2, the packet dissemination protocol is
still the public broadcast (Flood) from phase 1 (Dandelion is not yet
introduced). The only difference is that a subset of nodes is considered
"spies" and their observations are analyzed after the run to guess the origin.
The spies exhibit no different behavior during the simulation (they receive
and forward packets normally) — we simply read the reference log and filter
which events belonged to the spy nodes.

For this reason, phase2 reuses phase1_simulator.run_phase1; the only
additional tasks are selecting the spy nodes (adversary.select_bribed_nodes)
and then evaluating the attack (adversary.evaluate_attack_all_methods) on
the same log.

Order of operations (topology -> spy selection -> simulation), and why it
can't be reversed: select_bribed_nodes() needs the topology (node degrees,
cluster-boundary edges) to decide which nodes are worth bribing, so the
topology must exist first. What *was* wrong before is that run_phase1() was
regenerating the topology a second time from topology_seed instead of
reusing the exact object spy selection ran on. Even though generation is
deterministic (same seed -> identical graph) this was still redundant and
confusing to read, as if spy selection and simulation depended on two
independently-generated topologies. Fixed by generating the topology
exactly once here and passing that same object into run_phase1(topo=topo).
"""

from pathlib import Path

from adversary import (build_reference_profiles, evaluate_attack_all_methods,
                        plot_spy_selection, select_bribed_nodes)
from phase1_simulator import build_node_configs, run_phase1
from topology import generate_topology


def run_phase2(seed: int, num_packets: int = 200, budget_fraction: float = 0.3,
               log_path: str = "phase2_log.jsonl", verbose_spy_selection: bool = False):
    Path(log_path).write_text("")

    # 1. Generate the topology exactly once.
    topo = generate_topology(seed)
    configs, node_ids, addr_of = build_node_configs(topo)

    # 2. Select the Score_adv-optimal spy set on that topology, before
    #    running the simulation (spy identity must be fixed before packet
    #    injection so honest-only origin selection can use it).
    spy_indices = select_bribed_nodes(topo, budget_fraction=budget_fraction,
                                       verbose=verbose_spy_selection)
    spy_ids = {node_ids[i] for i in spy_indices}

    # 3. Offline profiling phase (Vector Profiling attack): build every
    #    HONEST node's reference timing vector against this exact spy set.
    #    This must happen before runtime evaluation and depends only on
    #    the topology + spy_ids (no packet traffic needed yet), which is
    #    exactly why it's a separate, explicit step here rather than being
    #    hidden inside evaluate_attack_all_methods.
    profile_data = build_reference_profiles(topo, spy_ids)

    # 4. Run the simulation, reusing the SAME topology object (no
    #    regeneration) and passing spy_ids so spies are excluded from
    #    being origins.
    _, _ = run_phase1(
        seed=seed,
        num_packets=num_packets,
        log_path=log_path,
        topo=topo,
        spy_ids=spy_ids,
    )

    # 5. Runtime evaluation of every method, reusing the precomputed profile.
    result = evaluate_attack_all_methods(log_path, spy_ids, topo=topo, profile_data=profile_data)
    return topo, spy_ids, result


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from pathlib import Path

    seed_val = 42

    # 1. Create master output directories for Phase 2
    base_out_dir = Path("outputs")
    log_dir = base_out_dir / "logs"
    phase2_out_dir = base_out_dir / "phase2"

    log_dir.mkdir(parents=True, exist_ok=True)
    phase2_out_dir.mkdir(parents=True, exist_ok=True)

    log_path = log_dir / f"phase2_seed{seed_val}.jsonl"

    # 2. Run Phase 2 (budget_fraction is now a CEILING - select_bribed_nodes
    #    decides the Score_adv-optimal number of spies up to this cap;
    #    verbose_spy_selection prints the k-sweep table to the console)
    topo, spy_ids, result = run_phase2(
        seed=seed_val,
        num_packets=200,
        budget_fraction=0.3,
        log_path=str(log_path),
        verbose_spy_selection=True,
    )

    # 3. Format the results string
    results_str = f"=== PHASE 2 SIMULATION RESULTS (Seed: {seed_val}) ===\n"
    results_str += f"Nodes = {topo.graph.number_of_nodes()}, Bribed spies = {result['n_spies']} "
    results_str += f"({', '.join(sorted(spy_ids))})\n"
    results_str += f"Total packets = {result['total_packets']}, Observed by >=1 spy = {result['observed_by_spies']}\n\n"

    results_str += "--- ACCURACY ---\n"
    results_str += f"Baseline (one-hop backtrack)         : {result['accuracy_baseline']:.3f}\n"
    results_str += f"Proposed (Vector Profiling, new)     : {result['accuracy_proposed']:.3f}\n"
    if "accuracy_common_ancestor" in result:
        results_str += f"Common-ancestor (older method)       : {result['accuracy_common_ancestor']:.3f}\n"
    if "accuracy_multilateration" in result:
        results_str += f"Multilateration (TDOA)                : {result['accuracy_multilateration']:.3f}\n"
    results_str += "\n"

    results_str += "--- ADVERSARY SCORE (Score_adv) ---\n"
    results_str += f"Baseline (one-hop backtrack)         : {result['score_adv_baseline']:.4f}\n"
    results_str += f"Proposed (Vector Profiling, new)     : {result['score_adv_proposed']:.4f}\n"
    if "score_adv_common_ancestor" in result:
        results_str += f"Common-ancestor (older method)       : {result['score_adv_common_ancestor']:.4f}\n"
    if "score_adv_multilateration" in result:
        results_str += f"Multilateration (TDOA)                : {result['score_adv_multilateration']:.4f}\n"

    print(results_str)

    # 4. Save textual outputs to a summary file
    summary_filename = phase2_out_dir / f"phase2_summary_seed{seed_val}.txt"
    with open(summary_filename, "w", encoding="utf-8") as f:
        f.write(results_str)

    # 5. Visualize which nodes were picked as spies (answers "which nodes did
    #    we bribe" — useful to sanity-check select_bribed_nodes() visually).
    node_ids_by_int = {n: f"n{n}" for n in topo.graph.nodes}
    spy_plot_filename = phase2_out_dir / f"spy_selection_seed{seed_val}.png"
    plot_spy_selection(topo, spy_ids, node_ids_by_int, str(spy_plot_filename))

    # 6. Comparative bar chart across all available methods
    labels = ['Baseline', 'Proposed\n(Vector Profiling)']
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

    print(f"Outputs successfully routed: Texts/Images -> '{phase2_out_dir}', Logs -> '{log_dir}'")