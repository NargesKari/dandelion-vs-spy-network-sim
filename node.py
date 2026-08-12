"""
Node's internal state, independent of network transport.

هر گره باید حداقل این وضعیت‌ها را نگه دارد:
- NodeID: شناسه یکتا (UUID)، مستقل از آدرس شبکه
- SelfAddr: آدرس ip:port روی localhost
- Peer List: همسایه‌های مستقیم به همراه آخرین زمان مشاهده/پاسخ
- Seen Set: شناسه بسته‌های دیده‌شده (برای جلوگیری از پردازش/ارسال مجدد)

این کلاس عمداً از لایه شبکه (UDP socket) جدا نگه داشته شده تا هم برای
تست واحد ساده باشد، هم بعداً بتوان لایه async I/O روی آن سوار کرد.
"""

import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, Optional, Set, Tuple

Addr = Tuple[str, int]  # (ip, port)


@dataclass
class PeerInfo:
    node_id: str
    addr: Addr
    last_seen: Optional[float] = None  # epoch seconds of last contact

    def touch(self, when: Optional[float] = None) -> None:
        self.last_seen = when if when is not None else time.time()


@dataclass
class Node:
    self_addr: Addr
    node_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    is_honest: bool = True  # False = spy/adversary-controlled node
    peers: Dict[str, PeerInfo] = field(default_factory=dict)
    seen_set: Set[str] = field(default_factory=set)

    # ---- peer management ----
    def add_peer(self, peer_id: str, addr: Addr) -> None:
        if peer_id == self.node_id:
            raise ValueError("a node cannot be its own peer")
        self.peers[peer_id] = PeerInfo(node_id=peer_id, addr=addr)

    def remove_peer(self, peer_id: str) -> None:
        self.peers.pop(peer_id, None)

    def peer_ids(self):
        return list(self.peers.keys())

    def degree(self) -> int:
        return len(self.peers)

    def mark_peer_seen(self, peer_id: str, when: Optional[float] = None) -> None:
        if peer_id in self.peers:
            self.peers[peer_id].touch(when)

    # ---- seen-set / dedup management ----
    def has_seen(self, packet_id: str) -> bool:
        return packet_id in self.seen_set

    def mark_seen(self, packet_id: str) -> None:
        self.seen_set.add(packet_id)

    def __repr__(self) -> str:
        role = "honest" if self.is_honest else "SPY"
        return (
            f"Node(id={self.node_id[:8]}…, addr={self.self_addr}, "
            f"role={role}, degree={self.degree()})"
        )
