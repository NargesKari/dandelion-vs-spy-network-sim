"""
فاز ۵ — اعمال محدودیت تأخیر عمدی توسط گره‌های جاسوس.

سؤال فاز ۵ (طبق صورت پروژه):
  «آیا این رفتار [تأخیر عمدی جاسوس‌ها] دقت تخمین مشترک مهاجم را افزایش
  می‌دهد یا کاهش؟ و چه تأثیری بر T_80% دارد؟»

طراحی آزمایش:
  - همان توپولوژی و همان ۹ جاسوس (۳۰٪) فازهای ۲ تا ۴ استفاده می‌شود
    (select_bribed_nodes روی همان TOPOLOGY_SEED).
  - برای هر p در {0.9, 0.5, 0.1} و هر یک از ۵ اجرای مستقل (همان run_seed
    فاز ۳/۴)، شبکه را **دوبار** با seedهای اصلیِ کاملاً یکسان اجرا می‌کنیم:
      1) NO-DELAY  : spy_ids=None  -> جاسوس‌ها هیچ تأخیر عمدی‌ای اعمال
         نمی‌کنند (دقیقاً معادل فاز ۳/۴).
      2) WITH-DELAY: spy_ids=<9 جاسوس> -> همان جاسوس‌ها پیش از هر
         بازپخش، تأخیر عمدی U(0, delay_base همان لینک) اعمال می‌کنند
         (node_process.py، بخش فاز ۵).
    چون تأخیر عمدی از یک RNG کاملاً جدا از rng اصلی گره کشیده می‌شود
    (نگاه کنید به phase1_simulator.run_phase1)، تنها تفاوت بین دو اجرا
    خودِ تأخیر عمدی است — جیتر لینک‌ها و سکه‌ی Stem/Fluff هر گره در هر دو
    حالت کاملاً یکسان می‌مانند. این یعنی می‌توانیم اثر خالص تأخیر عمدی را
    جدا از نوسان تصادفی معمول شبکه اندازه بگیریم.
  - روی هر دو لاگ، هر سه روش حدس (baseline / proposed / phase4) دوباره
    ارزیابی می‌شوند، و T_80% دوباره محاسبه می‌شود.

نکته درباره‌ی SETTLE_TIME_S: چون تأخیر عمدی می‌تواند مسیر انتشار را کند
کند (هر هاپی که از یک جاسوس عبور کند تا delay_base همان لینک دیرتر
می‌رسد)، برای حالت WITH-DELAY مهلت پایان (settle time) بلندتری نسبت به
فاز ۳/۴ در نظر گرفته‌ایم تا آخرین بسته‌ها فرصت رسیدن داشته باشند.
"""

import statistics
from pathlib import Path

from adversary import baseline_guess, evaluate_attack, phase4_guess, proposed_guess, select_bribed_nodes
from phase1_simulator import build_node_configs, compute_t80, run_phase1
from phase3_simulator import P_VALUES, RUNS_PER_P, TOPOLOGY_SEED, ORIGIN_SEED, NUM_PACKETS, coverage_fractions
from sim_log import read_log
from topology import generate_topology

LOG_DIR = "phase5_logs"
SETTLE_TIME_S = 10.0  # از ۶ ثانیه‌ی فاز ۳ بیشتر است چون تأخیر عمدی انتشار را کند می‌کند
BUDGET_FRACTION = 0.3

METHODS = {
    "baseline (phase2)": baseline_guess,
    "proposed (phase2)": proposed_guess,
    "phase4 (STEM-only)": phase4_guess,
}

CONDITIONS = ("nodelay", "delay")


def spy_delay_stats(log_path: str):
    """آمار توصیفی روی خودِ تأخیرهای عمدی اعمال‌شده (فقط برای حالت delay معنی دارد)."""
    events = [e for e in read_log(log_path) if e["event"] == "spy_delay"]
    vals = [e["intentional_delay_ms"] for e in events]
    if not vals:
        return {"n": 0, "mean_ms": 0.0, "max_ms": 0.0}
    return {"n": len(vals), "mean_ms": statistics.mean(vals), "max_ms": max(vals)}


def run_one(p: float, run_idx: int, spy_ids, condition: str):
    assert condition in CONDITIONS
    run_seed = int(p * 1000) * 100 + run_idx  # همان run_seed فاز ۳/۴ -> کاملاً قابل‌مقایسه
    log_path = f"{LOG_DIR}/p{p}_run{run_idx}_{condition}.jsonl"

    topo, node_ids = run_phase1(
        seed=run_seed, num_packets=NUM_PACKETS, log_path=log_path,
        settle_time_s=SETTLE_TIME_S, stem_p=p,
        topology_seed=TOPOLOGY_SEED, origin_seed=ORIGIN_SEED, run_seed=run_seed,
        spy_ids=(spy_ids if condition == "delay" else None),
    )
    total_nodes = topo.graph.number_of_nodes()
    cov = coverage_fractions(log_path, total_nodes)
    t80s, _n_origins = compute_t80(log_path, total_nodes)

    attack = {name: evaluate_attack(log_path, spy_ids, fn) for name, fn in METHODS.items()}

    return {
        "avg_coverage": statistics.mean(cov) if cov else 0.0,
        "reached80_frac": len(t80s) / NUM_PACKETS,
        "t80_ms": [t * 1000 for t in t80s],
        "attack": attack,
        "delay_stats": spy_delay_stats(log_path),
    }


def run_sweep(budget_fraction: float = BUDGET_FRACTION):
    Path(LOG_DIR).mkdir(parents=True, exist_ok=True)

    topo = generate_topology(TOPOLOGY_SEED)
    _configs, node_ids, _addr_of = build_node_configs(topo)
    spy_indices = select_bribed_nodes(topo, budget_fraction=budget_fraction)
    spy_ids = {node_ids[i] for i in spy_indices}

    # results[p][condition] = list of per-run dicts (طول = RUNS_PER_P)
    results = {p: {c: [] for c in CONDITIONS} for p in P_VALUES}
    for p in P_VALUES:
        for run_idx in range(RUNS_PER_P):
            for condition in CONDITIONS:
                results[p][condition].append(run_one(p, run_idx, spy_ids, condition))

    return spy_ids, results


def _fmt(vals):
    if not vals:
        return "n/a"
    return f"{statistics.mean(vals):.3f}/{statistics.median(vals):.3f}/{statistics.pstdev(vals):.3f}"


def summarize(results) -> str:
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
    spy_ids, results = run_sweep(budget_fraction=BUDGET_FRACTION)
    print(f"spies ({len(spy_ids)}): {', '.join(sorted(spy_ids))}\n")
    print(summarize(results))