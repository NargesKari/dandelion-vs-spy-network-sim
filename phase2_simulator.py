"""
ارکستریتور فاز ۲.

نکته مهم طراحی: در فاز ۲ هنوز پروتکل انتشار بسته‌ها همان پخش عمومی
فاز ۱ است (Dandelion هنوز وارد نشده). تنها تفاوت این است که زیرمجموعه‌ای
از گره‌ها «جاسوس» تلقی می‌شوند و مشاهداتشان بعد از اجرا برای حدس مبدأ
تحلیل می‌شود. جاسوس‌ها در جریان شبیه‌سازی هیچ رفتار متفاوتی ندارند (بسته
را عادی دریافت و فوروارد می‌کنند) — فقط لاگ مرجع را می‌خوانیم و فیلتر
می‌کنیم که کدام رویدادها متعلق به گره‌های جاسوس بوده‌اند.

به همین دلیل phase2 دوباره از phase1_simulator.run_phase1 استفاده می‌کند؛
تنها کار اضافه، انتخاب گره‌های جاسوس (adversary.select_bribed_nodes) و
سپس ارزیابی حمله (adversary.evaluate_attack) روی همان لاگ است.
"""

from pathlib import Path

from adversary import evaluate_attack_all_methods, select_bribed_nodes
from phase1_simulator import build_node_configs, run_phase1
from topology import generate_topology


def run_phase2(seed: int, num_packets: int = 200, budget_fraction: float = 0.3,
               log_path: str = "phase2_log.jsonl"):
    Path(log_path).write_text("")

    topo, node_ids = run_phase1(seed=seed, num_packets=num_packets, log_path=log_path)

    spy_indices = select_bribed_nodes(topo, budget_fraction=budget_fraction)
    spy_ids = {node_ids[i] for i in spy_indices}

    result = evaluate_attack_all_methods(log_path, spy_ids)
    return topo, spy_ids, result


if __name__ == "__main__":
    topo, spy_ids, result = run_phase2(seed=42, num_packets=200, budget_fraction=0.3)

    print(f"nodes = {topo.graph.number_of_nodes()}, bribed spies = {result['n_spies']} "
          f"({', '.join(sorted(spy_ids))})")
    print(f"packets total = {result['total_packets']}, observed by >=1 spy = {result['observed_by_spies']}")
    print(f"accuracy  baseline={result['accuracy_baseline']:.3f}   proposed={result['accuracy_proposed']:.3f}")
    print(f"Score_adv baseline={result['score_adv_baseline']:.4f}  proposed={result['score_adv_proposed']:.4f}")
