"""
فاز ۲ — طراحی حمله به پخش عمومی.

این ماژول دو بخش مستقل دارد:

  1) select_bribed_nodes(...)  — الگوریتم انتخاب گره‌های جاسوس (حداکثر ۳۰٪).
  2) baseline_guess / proposed_guess — دو روش حدسِ مبدأ از روی مشاهدات جاسوس‌ها.

هر دو بخش را در پاسخ متنی مربوطه با جزئیات توضیح داده‌ام؛ خلاصه منطق هرکدام
هم به‌صورت docstring در همین فایل آمده.
"""

import random
import statistics
from typing import Dict, List, Set

import networkx as nx

from sim_log import read_log


# ---------------------------------------------------------------------------
# ۱) انتخاب گره‌های جاسوس
# ---------------------------------------------------------------------------
def select_bribed_nodes(topo, budget_fraction: float = 0.3) -> Set[int]:
    """
    امتیاز هر گره = درجه‌اش + ۲ × (تعداد یال‌های مرزی خوشه‌اش که به این گره
    وصل‌اند). گره‌های مرزی خوشه (bridge/gateway) امتیاز مضاعف می‌گیرند چون
    هر بسته‌ای که بین دو خوشه رد و بدل شود اجباراً از یکی از همین گره‌ها
    عبور می‌کند — بریدن این گره‌ها بیشترین «دید» را با کمترین تعداد جاسوس
    به ما می‌دهد.

    برای جلوگیری از تمرکز همه جاسوس‌ها در یک خوشه (که فقط بسته‌های نزدیک
    به آن خوشه را خوب می‌بیند)، انتخاب به‌صورت round-robin بین خوشه‌ها انجام
    می‌شود: از هر خوشه، پرامتیازترین گره‌ی هنوز انتخاب‌نشده برداشته می‌شود،
    و این چرخه تا رسیدن به بودجه ادامه پیدا می‌کند.
    """
    g = topo.graph
    cluster_of = topo.cluster_of
    n_total = g.number_of_nodes()
    budget = max(1, int(n_total * budget_fraction))

    boundary_count = {n: 0 for n in g.nodes}
    for u, v in g.edges():
        if cluster_of[u] != cluster_of[v]:
            boundary_count[u] += 1
            boundary_count[v] += 1

    score = {n: g.degree(n) + 2 * boundary_count[n] for n in g.nodes}

    clusters: Dict[int, List[int]] = {}
    for n, c in cluster_of.items():
        clusters.setdefault(c, []).append(n)
    for c in clusters:
        clusters[c].sort(key=lambda n: score[n], reverse=True)

    cluster_ids = list(clusters.keys())
    pointers = {c: 0 for c in cluster_ids}
    chosen: List[int] = []

    while len(chosen) < budget:
        progressed = False
        for c in cluster_ids:
            if len(chosen) >= budget:
                break
            lst = clusters[c]
            p = pointers[c]
            if p < len(lst):
                chosen.append(lst[p])
                pointers[c] = p + 1
                progressed = True
        if not progressed:
            break  # همه گره‌ها تمام شدند

    return set(chosen)


# ---------------------------------------------------------------------------
# ۲) روش‌های حدسِ مبدأ
# ---------------------------------------------------------------------------
def _spy_sightings_by_packet(log_path: str, spy_ids: Set[str]):
    """
    برای هر packet_id، لیست رویدادهای 'receive' که node_id آن‌ها جزو جاسوس‌هاست
    را بر اساس wall_time مرتب برمی‌گرداند. هر رویداد شامل state (STEM/FLUFF)
    و sender_peer_id هم هست (اینکه جاسوس بسته را از کدام همسایه و در چه
    وضعیتی شنیده).
    """
    events = read_log(log_path)
    origins: Dict[str, dict] = {}
    sightings: Dict[str, List[dict]] = {}

    for e in events:
        if e["event"] == "origin":
            origins[e["packet_id"]] = e
        elif e["event"] == "receive" and e["node_id"] in spy_ids:
            sightings.setdefault(e["packet_id"], []).append(e)

    for pid in sightings:
        sightings[pid].sort(key=lambda e: e["wall_time"])

    return origins, sightings


def baseline_guess(sightings_for_packet: List[dict]) -> str:
    """
    روش پایه (فاز ۲): مبدأ حدس‌زده‌شده = خودِ جاسوسی که زودتر از همه بسته را
    دیده است — چه در وضعیت STEM چه FLUFF. روی Flood ساده معقول بود؛ روی
    Dandelion (فاز ۳) این روش گمراه‌کننده می‌شود (توضیح در proposed_guess).
    """
    return sightings_for_packet[0]["node_id"]


def proposed_guess(sightings_for_packet: List[dict]) -> str:
    """
    روش پیشنهادی فاز ۲ — «بازگشت یک‌هاپه» (بدون توجه به STEM/FLUFF):
    حدس = همسایه‌ای که بسته را برای زودبین‌ترین جاسوس فرستاده
    (sender_peer_id). روی Flood خوب کار می‌کند چون انتشار شعاعی و هم‌زمان
    است. روی Dandelion (فاز ۳) دیگر خوب کار نمی‌کند: اگر زودبین‌ترین جاسوس
    بسته را در وضعیت FLUFF دیده باشد، یعنی بعد از نقطه‌ی تبدیل Stem->Fluff
    آن را دیده — که می‌تواند از مبدأ واقعی خیلی دور باشد (چون قبل از تبدیل،
    بسته از مسیر تک‌به‌تکِ Stem رد شده و هیچ جاسوسی آن را ندیده). «یک قدم
    عقب» از این جاسوس فقط ما را به نقطه‌ی تبدیل نزدیک می‌کند، نه به مبدأ.
    """
    earliest = sightings_for_packet[0]
    sender = earliest.get("sender_peer_id")
    return sender if sender is not None else earliest["node_id"]


def phase4_guess(sightings_for_packet: List[dict]) -> str:
    """
    روش فاز ۴ — «بازگشت یک‌هاپه‌ی محدود به STEM» (STEM-only Backtrack):

    ایده‌ی اصلی: چون در Dandelion فقط بسته‌هایی که هنوز در وضعیت STEM
    هستند واقعاً از مسیر پنهانِ نزدیک به مبدأ عبور کرده‌اند، **فقط** به
    مشاهداتی که جاسوس بسته را در وضعیت STEM دیده اعتماد می‌کنیم (نه FLUFF).
    این مشاهدات به‌طور تضمینی روی همان مسیر تک‌به‌تکِ Stem قرار دارند،
    پس هرچه زودتر باشند به مبدأ نزدیک‌ترند.

    الگوریتم:
      ۱) از بین مشاهدات این جاسوس‌ها، فقط آن‌هایی که state == STEM را
         نگه دار (بر اساس wall_time already مرتب‌اند).
      ۲) اگر حداقل یکی وجود داشت: زودترینِ آن‌ها را بردار و «یک قدم
         عقب» برو (sender_peer_id آن — دقیقاً مثل proposed_guess، اما این
         بار روی مشاهده‌ی STEM، نه هر مشاهده‌ای).
      ۳) اگر هیچ جاسوسی بسته را در وضعیت STEM ندید (یعنی مسیر Stem قبل
         از رسیدن به هر جاسوسی به Fluff تبدیل شده)، اطلاعات مسیر پنهان
         را نداریم؛ در این حالت به بهترین گزینه‌ی موجود عقب‌نشینی می‌کنیم:
         همان proposed_guess (بازگشت یک‌هاپه از زودبین‌ترین مشاهده‌ی
         FLUFF) — چون از baseline_guess باز هم بهتر است، هرچند مثل حالت
         (۲) دقیق نیست.

    چرا این از proposed_guess فاز ۲ بهتر است: proposed_guess همیشه از
    زودبین‌ترین مشاهده (STEM یا FLUFF، هرکدام زودتر بود) استفاده می‌کند؛
    اما یک مشاهده‌ی FLUFF می‌تواند از نظر زمانی زودتر از یک مشاهده‌ی STEM
    باشد در حالی که از نظر مکانی خیلی دورتر از مبدأ است (چون Fluff یعنی
    بعد از تبدیل، پخش شعاعی و سریع شروع شده و می‌تواند زودتر به یک جاسوسِ
    نزدیکِ نقطه‌ی تبدیل برسد، حتی اگر آن نقطه چند هاپ با مبدأ فاصله داشته
    باشد). با اولویت‌دادن قطعی به مشاهدات STEM، این خطا حذف می‌شود.
    """
    stem_sightings = [e for e in sightings_for_packet if e["state"] == "STEM"]
    if stem_sightings:
        earliest_stem = stem_sightings[0]
        sender = earliest_stem.get("sender_peer_id")
        return sender if sender is not None else earliest_stem["node_id"]

    return proposed_guess(sightings_for_packet)


# ---------------------------------------------------------------------------
# ارزیابی: دقت و Score_adv برای یک روش حدسِ دلخواه
# ---------------------------------------------------------------------------
def evaluate_attack(log_path: str, spy_ids: Set[str], guess_fn=baseline_guess):
    origins, sightings = _spy_sightings_by_packet(log_path, spy_ids)

    total = len(origins)
    observed = 0
    correct = 0

    for pid, o in origins.items():
        obs = sightings.get(pid)
        if not obs:
            continue  # هیچ جاسوسی این بسته را ندید -> قابل حدس نیست
        observed += 1
        if guess_fn(obs) == o["node_id"]:
            correct += 1

    accuracy = correct / total if total else 0.0
    n_spies = len(spy_ids)

    return {
        "total_packets": total,
        "observed_by_spies": observed,
        "accuracy": accuracy,
        "score_adv": accuracy / n_spies if n_spies else 0.0,
        "n_spies": n_spies,
    }


def evaluate_attack_all_methods(log_path: str, spy_ids: Set[str]):
    """برای سازگاری با phase2_simulator: دقت/Score_adv هر دو روش فاز ۲ با هم."""
    r_base = evaluate_attack(log_path, spy_ids, baseline_guess)
    r_prop = evaluate_attack(log_path, spy_ids, proposed_guess)
    return {
        "total_packets": r_base["total_packets"],
        "observed_by_spies": r_base["observed_by_spies"],
        "accuracy_baseline": r_base["accuracy"],
        "accuracy_proposed": r_prop["accuracy"],
        "score_adv_baseline": r_base["score_adv"],
        "score_adv_proposed": r_prop["score_adv"],
        "n_spies": r_base["n_spies"],
    }
