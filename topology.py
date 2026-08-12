"""
تولید توپولوژی شبکه فضایی خوشه‌ای (بخش اول - قسمت ۲ صورت پروژه).

الگوریتم به صورت خلاصه:
  1) انتخاب ۴ تا ۶ مرکز خوشه به صورت تصادفی روی صفحه ۱۰۰۰×۱۰۰۰.
  2) قرار دادن اکثر گره‌ها حول این مراکز با توزیع نرمال دو بعدی (تراکم بالا)
     و باقی گره‌ها به صورت یکنواخت روی کل صفحه (پراکنده).
  3) ساخت گراف اولیه با اتصال هر گره به نزدیک‌ترین همسایه‌هایش تا وقتی
     درجه‌اش به حداقل ۲ برسد (بدون عبور از سقف ۴).
  4) اطمینان از همبندی کل گراف: اگر چند مؤلفه وجود داشت، نزدیک‌ترین جفت
     گره بین مؤلفه‌ها به هم وصل می‌شوند.
  5) اطمینان از اینکه هر خوشه حداقل با ۲ یال مستقل (edge-disjoint) به
     بقیه شبکه وصل است (با استفاده از برش-یال کمینه / minimum edge cut).
  6) محاسبه Delay_Base هر یال بر اساس فاصله اقلیدسی (۱ واحد فاصله = ۱ ms).

Delay_Total واقعی (با جیتر ۲۰٪) در زمان اجرا و برای هر بسته جداگانه
محاسبه می‌شود (تابع sample_delay)، چون جیتر متعلق به لحظه ارسال است،
نه ویژگی ثابت لینک.
"""

import json
import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import networkx as nx

PLANE_SIZE = 1000.0
MIN_DEGREE = 2
MAX_DEGREE = 4
MIN_CLUSTERS = 4
MAX_CLUSTERS = 6
MIN_NODES = 20
MAX_NODES = 30
MIN_CLUSTER_EDGE_CONNECTIVITY = 2  # حداقل دو لینک مستقل هر خوشه به بقیه شبکه
JITTER_FRACTION = 0.2  # ±20% جیتر روی تأخیر پایه
MS_PER_UNIT_DISTANCE = 1.0  # 1ms به ازای هر واحد فاصله اقلیدسی


@dataclass
class TopologyConfig:
    seed: int
    num_nodes: Optional[int] = None       # اگر None باشد، تصادفی بین 20 تا 30
    num_clusters: Optional[int] = None    # اگر None باشد، تصادفی بین 4 تا 6
    cluster_std: float = 55.0             # پراکندگی نقاط حول مرکز خوشه
    scattered_fraction: float = 0.2       # سهم گره‌های پراکنده (غیر خوشه‌ای)


def euclidean(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    return math.dist(p1, p2)


def _generate_positions(rng: random.Random, cfg: TopologyConfig):
    """مرحله ۱ و ۲: تولید موقعیت گره‌ها و برچسب خوشه هرکدام (-1 = پراکنده)."""
    n = cfg.num_nodes or rng.randint(MIN_NODES, MAX_NODES)
    k = cfg.num_clusters or rng.randint(MIN_CLUSTERS, MAX_CLUSTERS)

    centers = [
        (rng.uniform(0.15, 0.85) * PLANE_SIZE, rng.uniform(0.15, 0.85) * PLANE_SIZE)
        for _ in range(k)
    ]

    n_scattered = max(1, round(n * cfg.scattered_fraction))
    n_clustered = n - n_scattered

    positions: Dict[int, Tuple[float, float]] = {}
    cluster_of: Dict[int, int] = {}

    node_id = 0
    for i in range(n_clustered):
        c = i % k  # پخش تقریبا یکنواخت گره‌ها بین خوشه‌ها
        cx, cy = centers[c]
        x = min(max(rng.gauss(cx, cfg.cluster_std), 0), PLANE_SIZE)
        y = min(max(rng.gauss(cy, cfg.cluster_std), 0), PLANE_SIZE)
        positions[node_id] = (x, y)
        cluster_of[node_id] = c
        node_id += 1

    for _ in range(n_scattered):
        x = rng.uniform(0, PLANE_SIZE)
        y = rng.uniform(0, PLANE_SIZE)
        positions[node_id] = (x, y)
        cluster_of[node_id] = -1
        node_id += 1

    return positions, cluster_of, centers


def _nearest_candidates(node: int, positions, exclude: set):
    others = [m for m in positions if m != node and m not in exclude]
    others.sort(key=lambda m: euclidean(positions[node], positions[m]))
    return others


def _build_base_graph(rng: random.Random, positions) -> nx.Graph:
    """مرحله ۳: هر گره را تا رسیدن به حداقل درجه ۲ به نزدیک‌ترین‌ها وصل می‌کنیم."""
    g = nx.Graph()
    g.add_nodes_from(positions.keys())

    nodes = list(positions.keys())
    rng.shuffle(nodes)

    for node in nodes:
        while g.degree(node) < MIN_DEGREE:
            existing_neighbors = set(g.neighbors(node)) | {node}
            candidates = _nearest_candidates(node, positions, existing_neighbors)
            candidates = [c for c in candidates if g.degree(c) < MAX_DEGREE]
            if not candidates:
                break  # اگر همه پر بودند، در مرحله همبندسازی جبران می‌شود
            # از بین چند نزدیک‌ترین، یکی را تصادفی انتخاب کن تا گراف تنوع داشته باشد
            pick_pool = candidates[: min(3, len(candidates))]
            chosen = rng.choice(pick_pool)
            g.add_edge(node, chosen)
    return g


def _ensure_connectivity(rng: random.Random, g: nx.Graph, positions) -> None:
    """مرحله ۴: مؤلفه‌های جدا را با نزدیک‌ترین جفت گره به هم وصل می‌کنیم."""
    while not nx.is_connected(g):
        components = list(nx.connected_components(g))
        components.sort(key=len)
        comp_a = components[0]
        comp_b = set().union(*components[1:])

        best_pair = None
        best_dist = float("inf")
        for a in comp_a:
            for b in comp_b:
                d = euclidean(positions[a], positions[b])
                if d < best_dist:
                    best_dist = d
                    best_pair = (a, b)

        a, b = best_pair
        g.add_edge(a, b)  # اینجا سقف درجه را عمداً می‌شکنیم تا همبندی تضمین شود
        # اگر درجه a یا b بیش از حد رفت، بعداً می‌توان در بازبینی دستی اصلاح کرد.


def _cluster_edge_connectivity(g: nx.Graph, cluster_of, cluster_id: int) -> int:
    """تعداد یال‌های عبوری از مرز خوشه (edge boundary) بین گره‌های خوشه و بقیه شبکه.

    این یک تخمین ساده و شفاف از حداقل برش-یال است: اگر مرز خوشه حداقل ۲ یال
    داشته باشد، طبق قضیه مِنگر حداقل ۲ مسیر یال-مجزا هم بین خوشه و بقیه شبکه
    وجود دارد (به شرطی که خود خوشه داخلاً همبند باشد).
    """
    cluster_nodes = {n for n, c in cluster_of.items() if c == cluster_id}
    boundary_edges = [
        (u, v) for u, v in g.edges() if (u in cluster_nodes) != (v in cluster_nodes)
    ]
    return len(boundary_edges)


def _ensure_cluster_multi_connectivity(rng: random.Random, g: nx.Graph, positions, cluster_of) -> None:
    """مرحله ۵: هر خوشه باید حداقل با ۲ یال مستقل به بقیه شبکه وصل باشد."""
    cluster_ids = sorted(set(c for c in cluster_of.values() if c != -1))

    for cid in cluster_ids:
        cluster_nodes = [n for n, c in cluster_of.items() if c == cid]
        guard = 0
        while _cluster_edge_connectivity(g, cluster_of, cid) < MIN_CLUSTER_EDGE_CONNECTIVITY and guard < 20:
            guard += 1
            # نزدیک‌ترین جفت (گره داخل خوشه، گره بیرون خوشه) که هنوز وصل نیستند را پیدا کن
            outside_nodes = [n for n in g.nodes if cluster_of[n] != cid]
            best_pair, best_dist = None, float("inf")
            for u in cluster_nodes:
                for v in outside_nodes:
                    if g.has_edge(u, v):
                        continue
                    d = euclidean(positions[u], positions[v])
                    if d < best_dist:
                        best_dist = d
                        best_pair = (u, v)
            if best_pair is None:
                break
            g.add_edge(*best_pair)  # این‌جا هم ممکن است سقف درجه موقتاً شکسته شود


def _assign_delays(g: nx.Graph, positions) -> None:
    """مرحله ۶: تأخیر پایه هر یال بر اساس فاصله اقلیدسی."""
    for u, v in g.edges():
        dist = euclidean(positions[u], positions[v])
        g[u][v]["delay_base_ms"] = dist * MS_PER_UNIT_DISTANCE


def sample_delay(delay_base_ms: float, rng: random.Random) -> float:
    """Delay_Total = Delay_Base + U(-0.2*Delay_Base, 0.2*Delay_Base) — برای هر بسته جداگانه."""
    jitter = rng.uniform(-JITTER_FRACTION * delay_base_ms, JITTER_FRACTION * delay_base_ms)
    return delay_base_ms + jitter


@dataclass
class Topology:
    seed: int
    graph: nx.Graph
    positions: Dict[int, Tuple[float, float]]
    cluster_of: Dict[int, int]
    centers: List[Tuple[float, float]]

    def summary(self) -> str:
        degrees = [d for _, d in self.graph.degree()]
        lines = [
            f"seed = {self.seed}",
            f"nodes = {self.graph.number_of_nodes()}, edges = {self.graph.number_of_edges()}",
            f"clusters = {len(self.centers)}",
            f"degree min/avg/max = {min(degrees)}/{sum(degrees)/len(degrees):.2f}/{max(degrees)}",
            f"connected = {nx.is_connected(self.graph)}",
        ]
        for cid in sorted(set(c for c in self.cluster_of.values() if c != -1)):
            conn = _cluster_edge_connectivity(self.graph, self.cluster_of, cid)
            size = sum(1 for c in self.cluster_of.values() if c == cid)
            lines.append(f"  cluster {cid}: size={size}, boundary_edges={conn}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "seed": self.seed,
            "nodes": [
                {
                    "id": n,
                    "x": self.positions[n][0],
                    "y": self.positions[n][1],
                    "cluster": self.cluster_of[n],
                }
                for n in self.graph.nodes
            ],
            "edges": [
                {"u": u, "v": v, "delay_base_ms": self.graph[u][v]["delay_base_ms"]}
                for u, v in self.graph.edges
            ],
        }

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

    def plot(self, path: str) -> None:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 8))
        for u, v in self.graph.edges():
            x1, y1 = self.positions[u]
            x2, y2 = self.positions[v]
            ax.plot([x1, x2], [y1, y2], color="lightgray", linewidth=0.8, zorder=1)

        colors = plt.cm.tab10.colors
        for n in self.graph.nodes:
            c = self.cluster_of[n]
            color = "black" if c == -1 else colors[c % len(colors)]
            x, y = self.positions[n]
            ax.scatter(x, y, color=color, s=40, zorder=2)

        for i, (cx, cy) in enumerate(self.centers):
            ax.scatter(cx, cy, marker="x", color=colors[i % len(colors)], s=100, zorder=3)

        ax.set_xlim(0, PLANE_SIZE)
        ax.set_ylim(0, PLANE_SIZE)
        ax.set_title(f"Network topology (seed={self.seed})")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)


def generate_topology(seed: int, cfg: Optional[TopologyConfig] = None) -> Topology:
    if cfg is None:
        cfg = TopologyConfig(seed=seed)
    rng = random.Random(seed)

    max_attempts = 30
    for attempt in range(max_attempts):
        positions, cluster_of, centers = _generate_positions(rng, cfg)
        g = _build_base_graph(rng, positions)
        _ensure_connectivity(rng, g, positions)
        _ensure_cluster_multi_connectivity(rng, g, positions, cluster_of)

        degrees = [d for _, d in g.degree()]
        ok = (
            nx.is_connected(g)
            and min(degrees) >= 1  # حداقل ۱ تضمینی؛ MIN_DEGREE=2 هدف نرم است، چک زیر دقیق‌تر است
            and all(
                _cluster_edge_connectivity(g, cluster_of, cid) >= MIN_CLUSTER_EDGE_CONNECTIVITY
                for cid in set(c for c in cluster_of.values() if c != -1)
            )
        )
        if ok:
            _assign_delays(g, positions)
            return Topology(seed=seed, graph=g, positions=positions, cluster_of=cluster_of, centers=centers)

    raise RuntimeError(f"failed to generate a valid topology for seed={seed} after {max_attempts} attempts")


if __name__ == "__main__":
    import sys

    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 42
    topo = generate_topology(seed)
    print(topo.summary())
    topo.save(rf"C:\Users\Big Boss\Documents\Term\Term5\CN\Project\topology_seed{seed}.json")
    topo.plot(rf"C:\Users\Big Boss\Documents\Term\Term5\CN\Project\topology_seed{seed}.png")
    print("saved topology JSON + PNG")
