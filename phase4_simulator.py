"""
فاز ۴ — حمله پیشرفته به Dandelion.

این اسکریپت از همان لاگ‌های فاز ۳ (phase3_logs/p{p}_run{i}.jsonl) دوباره
استفاده می‌کند — چون لاگ مرجع از قبل شامل همه‌ی اطلاعات لازم (state،
sender_peer_id برای هر دریافت‌کننده) است، و اینکه «کدام گره جاسوس است»
فقط یک فیلتر است که بعد از اجرای شبیه‌سازی روی همان داده اعمال می‌شود.
پس نیازی به اجرای دوباره‌ی شبکه نیست.

برای هر p، هر ۵ اجرا را با سه روش تحلیل می‌کنیم:
  - baseline_guess   (فاز ۲، بدون تغییر)
  - proposed_guess   (فاز ۲، بازگشت یک‌هاپه‌ی بدون توجه به STEM/FLUFF)
  - phase4_guess     (جدید، بازگشت یک‌هاپه‌ی محدود به مشاهدات STEM)

و میانگین/میانه/انحراف معیار دقت و Score_adv را روی ۵ اجرا گزارش می‌کنیم.
"""

import statistics

from adversary import baseline_guess, evaluate_attack, phase4_guess, proposed_guess, select_bribed_nodes
from phase3_simulator import LOG_DIR, P_VALUES, RUNS_PER_P, TOPOLOGY_SEED
from phase1_simulator import build_node_configs
from topology import generate_topology

METHODS = {
    "baseline (phase2)": baseline_guess,
    "proposed (phase2)": proposed_guess,
    "phase4 (STEM-only backtrack)": phase4_guess,
}


def run_phase4_analysis(budget_fraction: float = 0.3):
    topo = generate_topology(TOPOLOGY_SEED)  # همان توپولوژی استفاده‌شده در فاز ۳
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
                r = evaluate_attack(log_path, spy_ids, guess_fn)
                results[p][name].append(r)

    return spy_ids, results


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
    spy_ids, results = run_phase4_analysis(budget_fraction=0.3)
    print(f"spies ({len(spy_ids)}): {', '.join(sorted(spy_ids))}")
    print(summarize(results))
