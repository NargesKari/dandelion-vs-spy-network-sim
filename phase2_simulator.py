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
and then evaluating the attack (adversary.evaluate_attack) on the same log.
"""

from pathlib import Path

from adversary import evaluate_attack_all_methods, select_bribed_nodes
from phase1_simulator import build_node_configs, run_phase1
from topology import generate_topology


def run_phase2(seed: int, num_packets: int = 200, budget_fraction: float = 0.3,
               log_path: str = "phase2_log.jsonl"):
    Path(log_path).write_text("")

    # 1. Generate the topology first using the given seed
    topo = generate_topology(seed)
    configs, node_ids, addr_of = build_node_configs(topo)
    
    # 2. Select the spy nodes before running the simulation
    spy_indices = select_bribed_nodes(topo, budget_fraction=budget_fraction)
    spy_ids = {node_ids[i] for i in spy_indices}

    # 3. Run the simulation and pass the spy_ids so they are excluded from being origins.
    # Note: run_phase1 generates the topology again with the same seed, resulting in the exact same graph.
    _, _ = run_phase1(
        seed=seed, 
        num_packets=num_packets, 
        log_path=log_path, 
        topology_seed=seed, 
        spy_ids=spy_ids 
    )

    result = evaluate_attack_all_methods(log_path, spy_ids, topo=topo)
    return topo, spy_ids, result


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from pathlib import Path

    seed_val = 42
    
    # 1. Create master output directories for Phase 2
    base_out_dir = Path("outputs")
    log_dir = base_out_dir / "logs"
    phase2_out_dir = base_out_dir / "phase2"
    
    # Create the directories if they do not exist
    log_dir.mkdir(parents=True, exist_ok=True)
    phase2_out_dir.mkdir(parents=True, exist_ok=True)
    
    # Define the log path explicitly inside the logs directory
    log_path = log_dir / f"phase2_seed{seed_val}.jsonl"

    # 2. Run Phase 2
    topo, spy_ids, result = run_phase2(
        seed=seed_val, 
        num_packets=200, 
        budget_fraction=0.3,
        log_path=str(log_path)
    )
    
    # 3. Format the results string
    results_str = f"=== PHASE 2 SIMULATION RESULTS (Seed: {seed_val}) ===\n"
    results_str += f"Nodes = {topo.graph.number_of_nodes()}, Bribed spies = {result['n_spies']} "
    results_str += f"({', '.join(sorted(spy_ids))})\n"
    results_str += f"Total packets = {result['total_packets']}, Observed by >=1 spy = {result['observed_by_spies']}\n\n"
    
    results_str += "--- ACCURACY ---\n"
    results_str += f"Baseline : {result['accuracy_baseline']:.3f}\n"
    results_str += f"Proposed : {result['accuracy_proposed']:.3f}\n\n"
    
    results_str += "--- ADVERSARY SCORE (Score_adv) ---\n"
    results_str += f"Baseline : {result['score_adv_baseline']:.4f}\n"
    results_str += f"Proposed : {result['score_adv_proposed']:.4f}\n"

    # Print to console for immediate feedback
    print(results_str)
    
    # 4. Save textual outputs to a summary file
    summary_filename = phase2_out_dir / f"phase2_summary_seed{seed_val}.txt"
    with open(summary_filename, "w", encoding="utf-8") as f:
        f.write(results_str)
        
    # 5. Create and save a comparative Bar Chart for the report
    labels = ['Baseline', 'Proposed']
    accuracies = [result['accuracy_baseline'], result['accuracy_proposed']]
    scores = [result['score_adv_baseline'], result['score_adv_proposed']]
    
    # X locations for the groups
    x = [0, 1]
    width = 0.35
    x_acc = [i - width/2 for i in x]
    x_score = [i + width/2 for i in x]

    fig, ax1 = plt.subplots(figsize=(8, 6))

    # Plot Accuracy on the primary Y-axis
    rects1 = ax1.bar(x_acc, accuracies, width, label='Accuracy', color='royalblue', edgecolor='black')
    ax1.set_ylabel('Accuracy (Rate)', color='royalblue', fontweight='bold')
    ax1.tick_params(axis='y', labelcolor='royalblue')
    ax1.set_ylim(0, max(accuracies) * 1.3)  # Add some headroom for labels

    # Create a secondary Y-axis for Score_adv
    ax2 = ax1.twinx()
    rects2 = ax2.bar(x_score, scores, width, label='Score_adv', color='darkorange', edgecolor='black')
    ax2.set_ylabel('Score_adv (Score / Spy)', color='darkorange', fontweight='bold')
    ax2.tick_params(axis='y', labelcolor='darkorange')
    ax2.set_ylim(0, max(scores) * 1.3)

    # Add labels, title, and formatting
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontweight='bold', fontsize=11)
    plt.title(f"Phase 2 Attack Performance: Baseline vs. Proposed (Seed {seed_val})", fontweight='bold')

    # Attach a text label above each bar, displaying its height
    def autolabel(rects, ax, is_score=False):
        for rect in rects:
            height = rect.get_height()
            format_str = f'{height:.4f}' if is_score else f'{height:.3f}'
            ax.annotate(format_str,
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 5),  # 5 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontweight='bold')

    autolabel(rects1, ax1, is_score=False)
    autolabel(rects2, ax2, is_score=True)

    # Combine legends from both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    # Save the chart INSIDE the phase2 directory
    plot_filename = phase2_out_dir / f"attack_performance_seed{seed_val}.png"
    plt.savefig(str(plot_filename), dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Outputs successfully routed: Texts/Images -> '{phase2_out_dir}', Logs -> '{log_dir}'")