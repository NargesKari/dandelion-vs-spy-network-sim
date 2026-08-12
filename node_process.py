"""
لایه شبکه گره: هر گره یک پردازه مستقل با یک سوکت UDP روی localhost است
(طبق بخش ۱-۱ صورت پروژه)، با asyncio.DatagramProtocol برای I/O ناهمزمان.

این ماژول حالا دو حالت فوروارد را پشتیبانی می‌کند (با پارامتر stem_p):

  - stem_p=None  -> حالت فاز ۱/۲: Flood ساده (به همه همسایه‌ها بجز فرستنده)
  - stem_p=<0..1> -> حالت فاز ۳ به بعد: Dandelion (Stem تک‌مسیره + Fluff = Flood)

منطق Dandelion (طبق صورت پروژه):
  - مبدأ فقط به یک همسایه‌ی تصادفی می‌فرستد (STEM)، بدون قید و شرط.
  - هر گره‌ی بعدی که یک بسته STEM دریافت می‌کند: با احتمال p در حالت
    Stem می‌ماند و به یک همسایه‌ی تصادفیِ دیگر (غیر از فرستنده) می‌فرستد؛
    با احتمال 1-p (یا اگر همسایه واجد شرایطی نماند)، بسته به FLUFF تبدیل
    می‌شود و از همان‌جا مثل فاز ۱ به همه‌ی همسایه‌ها (به‌جز فرستنده) پخش
    می‌شود.
  - بسته‌ی FLUFF از همان لحظه‌ی تبدیل، مثل Flood ساده رفتار می‌کند.

پروتکل پیام روی UDP (JSON):
  {"type": "PKT", "packet_id": "...", "state": "STEM"|"FLUFF"}   -> بسته شبکه
  {"type": "INJECT", "packet_id": "..."}                          -> دستور کنترلی: این گره مبدأ بسته است
  {"type": "SHUTDOWN"}                                            -> دستور کنترلی: پایان شبیه‌سازی

فاز ۵ — تأخیر عمدی گره‌های جاسوس:
  اگر is_spy=True باشد، این گره پیش از هر ارسال (برای هر همسایه، جدا از
  بقیه) یک تأخیر عمدی اضافی U(0, delay_base_ms همان لینک) اعمال می‌کند؛
  یعنی سقف تأخیر عمدی دقیقاً برابر تأخیر پایه‌ی همان لینک است (طبق بخش
  ۲-۲-ج صورت پروژه). این تأخیر:
    - هرگز باعث حذف یا نگهداری نامحدود بسته نمی‌شود (فقط دیرتر ارسال
      می‌شود، ارسال هنوز تضمینی است).
    - از یک RNG کاملاً جدا (spy_rng، با seed مستقل از rng اصلی گره)
      گرفته می‌شود؛ به این ترتیب فعال/غیرفعال بودن تأخیر جاسوسی هیچ
      تأثیری روی توالی جیتر لینک یا سکه‌ی Stem/Fluff گره‌ها نمی‌گذارد و
      مقایسه‌ی «با تأخیر» در برابر «بدون تأخیر» با seed یکسان کاملاً
      کنترل‌شده باقی می‌ماند (تنها متغیر مستقل، خودِ تأخیر عمدی است).
    - در لاگ مرجع با رویداد "spy_delay" ثبت می‌شود تا در گزارش قابل
      تحلیل باشد (میانگین تأخیر اعمالی، تأثیر بر ترتیب مشاهدات و غیره).
"""

import asyncio
import json
import random
from typing import Dict, Optional, Tuple

from node import Node
from packet import FLUFF, STEM, Packet
from sim_log import log_event
from topology import sample_delay

Addr = Tuple[str, int]


class NodeUDPProtocol(asyncio.DatagramProtocol):
    def __init__(
        self,
        node: Node,
        peer_delay: Dict[str, float],
        addr_to_peer: Dict[Addr, str],
        log_path: str,
        rng: random.Random,
        loop: asyncio.AbstractEventLoop,
        stem_p: Optional[float] = None,
        lock=None,
        is_spy: bool = False,
        spy_rng: Optional[random.Random] = None,
    ):
        self.node = node
        self.peer_delay = peer_delay
        self.addr_to_peer = addr_to_peer
        self.log_path = log_path
        self.rng = rng
        self.loop = loop
        self.stem_p = stem_p  # None => فاز ۱/۲ (flood)؛ float => فاز ۳+ (dandelion)
        self.lock = lock
        self.transport = None
        # فاز ۵: آیا این گره جاسوس است و مجاز به اعمال تأخیر عمدی هست؟
        self.is_spy = is_spy
        self.spy_rng = spy_rng  # RNG مستقل، فقط برای کشیدن مقدار تأخیر عمدی

    def connection_made(self, transport):
        self.transport = transport

    def datagram_received(self, data, addr):
        try:
            msg = json.loads(data.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            return

        mtype = msg.get("type")
        if mtype == "PKT":
            self._handle_packet(msg, addr)
        elif mtype == "INJECT":
            self._handle_inject(msg)
        elif mtype == "SHUTDOWN":
            self.loop.call_later(0.05, self.loop.stop)

    # ---- ابزارهای ارسال ----
    def _send_packet(self, packet: Packet, addr: Addr):
        if self.transport is None:
            return
        payload = json.dumps({"type": "PKT", **packet.to_dict()}).encode("utf-8")
        self.transport.sendto(payload, addr)

    def _schedule_send(self, packet: Packet, peer_id: str):
        peer_info = self.node.peers[peer_id]
        delay_base = self.peer_delay.get(peer_id, 0.0)
        delay_ms = sample_delay(delay_base, self.rng)

        intentional_ms = 0.0
        if self.is_spy and self.spy_rng is not None:
            # سقف تأخیر عمدی = delay_base همین لینک (نه بیشتر)؛ هرگز drop نمی‌شود.
            intentional_ms = self.spy_rng.uniform(0.0, delay_base)
            log_event(self.log_path, {
                "event": "spy_delay",
                "node_id": self.node.node_id,
                "packet_id": packet.packet_id,
                "peer_id": peer_id,
                "delay_base_ms": delay_base,
                "intentional_delay_ms": intentional_ms,
            }, lock=self.lock)

        total_ms = max(delay_ms, 0.0) + intentional_ms
        self.loop.call_later(total_ms / 1000.0, self._send_packet, packet, peer_info.addr)

    def _forward_flood(self, packet: Packet, exclude_peer_id: str = None):
        """پخش به همه همسایه‌ها به‌جز فرستنده (فاز ۱ / حالت Fluff)."""
        for peer_id in self.node.peers:
            if peer_id == exclude_peer_id:
                continue
            self._schedule_send(packet, peer_id)

    def _forward_stem_single(self, packet: Packet, exclude_peer_id: str = None) -> Optional[str]:
        """ارسال به دقیقاً یک همسایه‌ی تصادفی (غیر از فرستنده). خروجی: peer_id انتخاب‌شده یا None."""
        candidates = [pid for pid in self.node.peers if pid != exclude_peer_id]
        if not candidates:
            return None
        chosen = self.rng.choice(candidates)
        self._schedule_send(packet, chosen)
        return chosen

    # ---- پردازش بسته‌ی دریافتی از یک همسایه ----
    def _handle_packet(self, msg, addr: Addr):
        packet = Packet(packet_id=msg["packet_id"], state=msg["state"])
        sender_peer_id = self.addr_to_peer.get(addr)

        if self.node.has_seen(packet.packet_id):
            return  # طبق Seen Set، بسته تکراری پردازش/ارسال مجدد نمی‌شود

        self.node.mark_seen(packet.packet_id)
        log_event(self.log_path, {
            "event": "receive",
            "node_id": self.node.node_id,
            "packet_id": packet.packet_id,
            "state": packet.state,
            # sender_peer_id فقط در لاگ مرجع (ground-truth) ثبت می‌شود، نه در بسته
            # شبکه — از فاز ۲ به بعد لازم است (تحلیل حمله، بازسازی مسیر Stem).
            "sender_peer_id": sender_peer_id,
        }, lock=self.lock)

        if self.stem_p is None:
            # حالت فاز ۱/۲: Flood ساده
            self._forward_flood(packet, exclude_peer_id=sender_peer_id)
            return

        # حالت فاز ۳+: Dandelion
        if packet.state == FLUFF:
            self._forward_flood(packet, exclude_peer_id=sender_peer_id)
            return

        # packet.state == STEM
        continue_stem = self.rng.random() < self.stem_p
        next_hop = None
        if continue_stem:
            next_hop = self._forward_stem_single(packet, exclude_peer_id=sender_peer_id)
        if next_hop is None:
            # یا با احتمال 1-p تصمیم به Fluff گرفتیم، یا همسایه‌ی واجد شرایطی نبود
            self._forward_flood(packet.as_fluff(), exclude_peer_id=sender_peer_id)

    # ---- شروع بسته توسط مبدأ ----
    def _handle_inject(self, msg):
        origin_state = STEM if self.stem_p is not None else FLUFF
        packet = Packet(packet_id=msg["packet_id"], state=origin_state)
        self.node.mark_seen(packet.packet_id)
        log_event(self.log_path, {
            "event": "receive",
            "node_id": self.node.node_id,
            "packet_id": packet.packet_id,
            "state": packet.state,
            "sender_peer_id": None,
            "is_origin": True,
        }, lock=self.lock)

        if self.stem_p is None:
            # فاز ۱/۲: مبدأ مستقیم به همه پخش می‌کند
            self._forward_flood(packet, exclude_peer_id=None)
            return

        # فاز ۳+: مبدأ بدون قید و شرط فقط به یک همسایه‌ی تصادفی می‌فرستد (STEM)
        next_hop = self._forward_stem_single(packet, exclude_peer_id=None)
        if next_hop is None:
            # اگر مبدأ هیچ همسایه‌ای نداشت (نباید پیش بیاید چون حداقل درجه تضمین شده)
            self._forward_flood(packet.as_fluff(), exclude_peer_id=None)


def run_node(node_id: str, self_addr, peers: Dict[str, Addr], peer_delay: Dict[str, float],
             log_path: str, seed: int, stem_p: Optional[float] = None, lock=None,
             is_spy: bool = False, spy_delay_seed: Optional[int] = None):
    """Entry point اجرا شده در یک پردازه مستقل (از طریق multiprocessing).

    is_spy / spy_delay_seed فقط برای فاز ۵ معنی دارند: اگر is_spy=True باشد
    یک RNG کاملاً مستقل (seed=spy_delay_seed) فقط برای کشیدن تأخیر عمدی
    ساخته می‌شود؛ در فازهای ۱ تا ۴ (is_spy=False، حالت پیش‌فرض) هیچ تأخیر
    عمدی‌ای اعمال نمی‌شود و رفتار دقیقاً مثل قبل باقی می‌ماند.
    """
    node = Node(self_addr=tuple(self_addr), node_id=node_id, is_honest=not is_spy)
    addr_to_peer: Dict[Addr, str] = {}
    for pid, addr in peers.items():
        addr = tuple(addr)
        node.add_peer(pid, addr)
        addr_to_peer[addr] = pid

    rng = random.Random(seed)
    spy_rng = random.Random(spy_delay_seed) if is_spy else None

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    listen = loop.create_datagram_endpoint(
        lambda: NodeUDPProtocol(
            node, peer_delay, addr_to_peer, log_path, rng, loop, stem_p=stem_p, lock=lock,
            is_spy=is_spy, spy_rng=spy_rng,
        ),
        local_addr=tuple(self_addr),
    )
    transport, _protocol = loop.run_until_complete(listen)
    try:
        loop.run_forever()
    finally:
        transport.close()
        loop.close()