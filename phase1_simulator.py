"""
ارکستریتور فاز ۱ (پخش عمومی پایه).

این اسکریپت:
  1) توپولوژی را با seed مشخص می‌سازد.
  2) برای هر گره یک پردازه مستقل (multiprocessing) با سوکت UDP جدا راه می‌اندازد.
  3) سناریوی ۲۰۰ بسته‌ای را اجرا می‌کند: مبدأ هر بسته تصادفی از میان همه
     گره‌ها، بدون الگوی زمانی ثابت (فاصله بین تزریق‌ها تصادفی است).
  4) برای هر بسته رویداد "origin" را در لاگ مرجع ثبت می‌کند (مبدأ واقعی +
     زمان واقعی) — این اطلاعات هرگز در بسته شبکه قرار نمی‌گیرد.
  5) پس از پایان انتشار، به همه گره‌ها SHUTDOWN می‌فرستد.
  6) از لاگ، T_80% (زمان رسیدن به ۸۰٪ گره‌ها) را برای هر بسته حساب می‌کند.
"""

import json
import multiprocessing as mp
import random
import socket
import statistics
import time
from pathlib import Path
from typing import Optional, Set

from node_process import run_node
from packet import new_packet_id
from sim_log import log_event, read_log
from topology import generate_topology

BASE_PORT = 20000


def build_node_configs(topo):
    node_ids = {n: f"n{n}" for n in topo.graph.nodes}
    addr_of = {n: ("127.0.0.1", BASE_PORT + n) for n in topo.graph.nodes}

    configs = {}
    for n in topo.graph.nodes:
        peers, peer_delay = {}, {}
        for nb in topo.graph.neighbors(n):
            peers[node_ids[nb]] = addr_of[nb]
            peer_delay[node_ids[nb]] = topo.graph[n][nb]["delay_base_ms"]
        configs[n] = {
            "node_id": node_ids[n],
            "self_addr": addr_of[n],
            "peers": peers,
            "peer_delay": peer_delay,
        }
    return configs, node_ids, addr_of


def run_phase1(seed: int, num_packets: int, log_path: str, settle_time_s: float = 3.0,
               stem_p=None, topology_seed=None, origin_seed=None, run_seed=None,
               spy_ids: Optional[Set[str]] = None):
    """
    stem_p=None  -> فاز ۱/۲: Flood ساده (رفتار قبلی، بدون تغییر).
    stem_p=<p>   -> فاز ۳+: Dandelion با احتمال ادامه‌ی Stem برابر p.

    topology_seed / origin_seed / run_seed برای فاز ۳ به بعد جدا شده‌اند
    تا بتوان توپولوژی و توالی مبدأها را بین چند اجرای مقایسه‌ای (مثلاً
    p=0.9 در برابر p=0.1) ثابت نگه داشت و فقط رفتار تصادفیِ شبکه (جیتر
    تأخیر + سکه‌ی Stem/Fluff) را بین اجراها تغییر داد. اگر مقداردهی
    نشوند، هرسه برابر seed می‌شوند (دقیقاً رفتار قبلی فاز ۱).

    spy_ids (فاز ۵): مجموعه node_id هایی که علاوه بر نقش «مشاهده‌گر»
    (که همیشه از روی لاگ مرجع قابل استخراج است)، عملاً هم در حین اجرای
    شبکه رفتار جاسوس فاز ۵ (تأخیر عمدی پیش از بازپخش) را نشان می‌دهند.
    اگر None باشد (پیش‌فرض، رفتار فازهای ۱ تا ۴) هیچ گره‌ای تأخیر عمدی
    اعمال نمی‌کند — even اگر بعداً برای تحلیل حمله همان گره‌ها را به‌عنوان
    جاسوس در نظر بگیریم (evaluate_attack مستقل از این پرچم است).
    """
    topology_seed = seed if topology_seed is None else topology_seed
    origin_seed = seed if origin_seed is None else origin_seed
    run_seed = seed if run_seed is None else run_seed

    Path(log_path).write_text("")  # پاک‌سازی لاگ اجرای قبلی

    topo = generate_topology(topology_seed)
    configs, node_ids, addr_of = build_node_configs(topo)
    log_lock = mp.Lock()
    procs = []
    for n, cfg in configs.items():
        is_spy = spy_ids is not None and cfg["node_id"] in spy_ids
        # offset بزرگ و جدا از فضای seedهای rng اصلی (run_seed*1000+n) تا هیچ
        # هم‌پوشانی‌ای با seed گره‌های دیگر یا با rng اصلی همین گره نداشته باشد.
        spy_delay_seed = (run_seed * 1000 + n + 900_000_000) if is_spy else None
        p = mp.Process(
            target=run_node,
            kwargs=dict(
                node_id=cfg["node_id"], self_addr=cfg["self_addr"], peers=cfg["peers"],
                peer_delay=cfg["peer_delay"], log_path=log_path, seed=run_seed * 1000 + n,
                stem_p=stem_p,
                lock=log_lock,
                is_spy=is_spy,
                spy_delay_seed=spy_delay_seed,
            ),
            daemon=True,
        )
        p.start()
        procs.append(p)

    time.sleep(0.3)  # مهلت برای bind شدن سوکت‌های UDP

    rng = random.Random(origin_seed)
    ctrl_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    origin_nodes = list(topo.graph.nodes)

    for _ in range(num_packets):
        origin = rng.choice(origin_nodes)
        packet_id = new_packet_id()
        time.sleep(rng.uniform(0.004, 0.02))  # بدون الگوی زمانی از پیش تعیین‌شده
        log_event(log_path, {
            "event": "origin",
            "node_id": node_ids[origin],
            "packet_id": packet_id,
            "wall_time": time.time(),
        }, lock=log_lock)
        ctrl_sock.sendto(json.dumps({"type": "INJECT", "packet_id": packet_id}).encode(), addr_of[origin])

    time.sleep(settle_time_s)  # مهلت برای رسیدن آخرین بسته‌ها به همه گره‌ها

    for cfg in configs.values():
        ctrl_sock.sendto(json.dumps({"type": "SHUTDOWN"}).encode(), cfg["self_addr"])

    for p in procs:
        p.join(timeout=3)
        if p.is_alive():
            p.terminate()

    return topo, node_ids


def compute_t80(log_path: str, total_nodes: int):
    events = read_log(log_path)
    origins, receives = {}, {}
    for e in events:
        if e["event"] == "origin":
            origins[e["packet_id"]] = e["wall_time"]
        elif e["event"] == "receive":
            receives.setdefault(e["packet_id"], []).append(e["wall_time"])

    target = max(1, round(0.8 * total_nodes))
    t80s = []
    for pid, t0 in origins.items():
        times = sorted(receives.get(pid, []))
        if len(times) >= target:
            t80s.append(times[target - 1] - t0)
    return t80s, len(origins)


if __name__ == "__main__":
    LOG = "phase1_log.jsonl"
    topo, node_ids = run_phase1(seed=42, num_packets=200, log_path=LOG)

    t80s, n_origins = compute_t80(LOG, topo.graph.number_of_nodes())
    print(f"nodes = {topo.graph.number_of_nodes()}")
    print(f"packets injected = {n_origins}, reached 80% coverage = {len(t80s)}")
    if t80s:
        print(
            f"T_80%: avg={statistics.mean(t80s)*1000:.1f}ms  "
            f"median={statistics.median(t80s)*1000:.1f}ms  "
            f"stdev={statistics.pstdev(t80s)*1000:.1f}ms"
        )