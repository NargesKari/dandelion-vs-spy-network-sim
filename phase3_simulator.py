"""
فاز ۳ — پویش پارامتر p برای Dandelion.

طراحی آزمایش (و چرایی آن):
  - `TOPOLOGY_SEED` و `ORIGIN_SEED` برای همه‌ی اجراها (هر سه مقدار p و هر
    ۵ تکرار) ثابت نگه داشته می‌شوند. یعنی همه‌ی اجراها دقیقاً روی همان
    توپولوژی و همان توالی «کدام گره کِی بسته تولید می‌کند» اجرا می‌شوند.
    این کار لازم است تا تفاوت نتایج بین p=0.9 و p=0.1 واقعاً ناشی از خودِ
    p باشد، نه ناشی از تصادفی بودن توپولوژی یا زمان‌بندی تزریق بسته‌ها.
  - چیزی که بین ۵ تکرارِ هر p عوض می‌شود فقط `run_seed` است: این seed هم
    جیتر تأخیر لینک‌ها و هم سکه‌ی احتمالاتی «ادامه‌ی Stem یا تبدیل به
    Fluff» را کنترل می‌کند. تغییر آن باعث تنوع در رفتار شبکه می‌شود که
    دقیقاً همان چیزی است که برای محاسبه‌ی میانگین/میانه/انحراف معیار
    لازم داریم.

دو خانواده معیار گزارش می‌شود:
  - `avg_coverage`: میانگین نسبت گره‌هایی که هر بسته را دریافت کرده‌اند
    (۰ تا ۱). این معیار مهم‌تر از T_80% است چون در Dandelion واقعی، یک
    بسته ممکن است اصلاً به همه‌جا نرسد (توضیح زیر).
  - `T_80%`: زمان رسیدن به ۸۰٪ گره‌ها، فقط برای بسته‌هایی که به آن رسیدند.

نکته‌ی مهم و غیرمنتظره‌ای که در تست‌ها دیده شد: چون در این پیاده‌سازی
مسیر Stem یک *گشت تصادفی* روی گراف است (نه یک مسیر ثابت از پیش تعیین‌شده
مثل Dandelion اصلی)، ممکن است مسیر Stem به یک گره‌ای برگردد که قبلاً
همان بسته را دیده است. آن گره طبق SeenSet بسته را نادیده می‌گیرد و اصلاً
آن را فوروارد نمی‌کند — یعنی بسته بدون اینکه هرگز به Fluff تبدیل شود
"می‌میرد" و به بقیه‌ی شبکه نمی‌رسد. هرچه p بزرگ‌تر باشد، مسیر Stem
طولانی‌تر است و احتمال این برخورد (collision) بیشتر می‌شود. به همین
دلیل انتظار داریم p=0.9 میانگین پوشش پایین‌تری نسبت به p=0.1 داشته باشد
— این یک یافته‌ی واقعی و قابل گزارش است، نه یک باگ.
"""

import statistics
from pathlib import Path

from phase1_simulator import compute_t80, run_phase1
from sim_log import read_log

TOPOLOGY_SEED = 42
ORIGIN_SEED = 42
P_VALUES = (0.9, 0.5, 0.1)
RUNS_PER_P = 5
NUM_PACKETS = 200
SETTLE_TIME_S = 6.0
LOG_DIR = "phase3_logs"


def coverage_fractions(log_path: str, total_nodes: int):
    events = read_log(log_path)
    receives = {}
    for e in events:
        if e["event"] == "receive":
            receives.setdefault(e["packet_id"], set()).add(e["node_id"])
    return [len(nodes) / total_nodes for nodes in receives.values()]


def run_one(p: float, run_idx: int):
    run_seed = int(p * 1000) * 100 + run_idx  # هر ترکیب (p, run_idx) یک seed یکتا و قابل بازتولید
    log_path = f"{LOG_DIR}/p{p}_run{run_idx}.jsonl"

    topo, _node_ids = run_phase1(
        seed=run_seed, num_packets=NUM_PACKETS, log_path=log_path,
        settle_time_s=SETTLE_TIME_S, stem_p=p,
        topology_seed=TOPOLOGY_SEED, origin_seed=ORIGIN_SEED, run_seed=run_seed,
    )
    total_nodes = topo.graph.number_of_nodes()
    cov = coverage_fractions(log_path, total_nodes)
    t80s, _ = compute_t80(log_path, total_nodes)

    return {
        "avg_coverage": statistics.mean(cov) if cov else 0.0,
        "full_coverage_frac": (sum(1 for c in cov if c >= 0.999) / len(cov)) if cov else 0.0,
        "reached80_frac": len(t80s) / NUM_PACKETS,
        "t80_ms": [t * 1000 for t in t80s],
    }


def run_sweep():
    Path(LOG_DIR).mkdir(parents=True, exist_ok=True)
    results = {}
    for p in P_VALUES:
        results[p] = [run_one(p, i) for i in range(RUNS_PER_P)]
    return results


def summarize(results) -> str:
    lines = [f"{'p':>5} | {'avg_coverage m/md/sd':>22} | {'full_cov_frac':>13} | {'reached80_frac':>14} | {'T_80% ms m/md/sd':>20}"]
    lines.append("-" * len(lines[0]))
    for p, runs in results.items():
        avg_cov = [r["avg_coverage"] for r in runs]
        full_cov = [r["full_coverage_frac"] for r in runs]
        r80 = [r["reached80_frac"] for r in runs]
        all_t80 = [t for r in runs for t in r["t80_ms"]]

        cov_str = f"{statistics.mean(avg_cov):.3f}/{statistics.median(avg_cov):.3f}/{statistics.pstdev(avg_cov):.3f}"
        t80_str = (
            f"{statistics.mean(all_t80):.0f}/{statistics.median(all_t80):.0f}/{statistics.pstdev(all_t80):.0f}"
            if all_t80 else "n/a"
        )
        lines.append(
            f"{p:>5} | {cov_str:>22} | {statistics.mean(full_cov):>13.3f} | "
            f"{statistics.mean(r80):>14.3f} | {t80_str:>20}"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    results = run_sweep()
    print(summarize(results))
