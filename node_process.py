"""
Node network layer: Each node is an independent process with a UDP socket 
on localhost (as per project phase 1-1), using asyncio.DatagramProtocol 
for asynchronous I/O.

This module now supports two forwarding modes (via the stem_p parameter):

  - stem_p=None  -> Phase 1/2 mode: Simple Flood (to all neighbors except sender)
  - stem_p=<0..1> -> Phase 3+ mode: Dandelion (Single-path Stem + Fluff = Flood)

Dandelion logic (as per project specifications):
  - The origin sends only to one random neighbor (STEM), unconditionally.
  - Any subsequent node that receives a STEM packet: with probability p, it 
    stays in the Stem state and forwards to another random neighbor (excluding 
    the sender); with probability 1-p (or if no eligible neighbor remains), 
    the packet transitions to FLUFF and is broadcast from there to all 
    neighbors (excluding the sender) just like Phase 1.
  - A FLUFF packet behaves exactly like a simple Flood from the moment of transition.

Message protocol over UDP (JSON):
  {"type": "PKT", "packet_id": "...", "state": "STEM"|"FLUFF"}   -> Network packet
  {"type": "INJECT", "packet_id": "..."}                          -> Control cmd: this node is the origin
  {"type": "SHUTDOWN"}                                            -> Control cmd: end simulation

Phase 5 — Intentional delay by spy nodes:
  If is_spy=True, this node applies an additional intentional delay of 
  U(0, delay_base_ms of that specific link) before any transmission (for 
  each neighbor independently); meaning the maximum intentional delay is 
  exactly equal to the base delay of that link. This delay:
    - Never causes packets to be dropped or held indefinitely (it is only 
      delayed, delivery is still guaranteed).
    - Is drawn from a completely isolated RNG (spy_rng, with an independent 
      seed from the node's main rng); thus, enabling/disabling the spy delay 
      has absolutely no effect on the link jitter sequence or the node's 
      Stem/Fluff coin tosses. Comparing "with delay" vs "without delay" 
      using the same seed remains perfectly controlled.
    - Is recorded in the reference log with a "spy_delay" event for analysis 
      in the report (average applied delay, effect on observation order, etc.).
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
        self.stem_p = stem_p  # None => phase 1/2 (flood); float => phase 3+ (dandelion)
        self.lock = lock
        self.transport = None
        # Phase 5: Is this node a spy and permitted to apply intentional delay?
        self.is_spy = is_spy
        self.spy_rng = spy_rng  # Independent RNG, exclusively for drawing intentional delay

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

    # ---- Sending utilities ----
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
            # Intentional delay cap = delay_base of this link; never dropped.
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
        """Broadcast to all neighbors except the sender (Phase 1 / Fluff mode)."""
        for peer_id in self.node.peers:
            if peer_id == exclude_peer_id:
                continue
            self._schedule_send(packet, peer_id)

    def _forward_stem_single(self, packet: Packet, exclude_peer_id: str = None) -> Optional[str]:
        """Send to exactly one random neighbor (excluding the sender). Returns chosen peer_id or None."""
        candidates = [pid for pid in self.node.peers if pid != exclude_peer_id]
        if not candidates:
            return None
        chosen = self.rng.choice(candidates)
        self._schedule_send(packet, chosen)
        return chosen

    # ---- Processing a packet received from a neighbor ----
    def _handle_packet(self, msg, addr: Addr):
        packet = Packet(packet_id=msg["packet_id"], state=msg["state"])
        sender_peer_id = self.addr_to_peer.get(addr)

        if self.node.has_seen(packet.packet_id):
            return  # According to Seen Set, duplicate packets are neither processed nor forwarded again

        self.node.mark_seen(packet.packet_id)
        log_event(self.log_path, {
            "event": "receive",
            "node_id": self.node.node_id,
            "packet_id": packet.packet_id,
            "state": packet.state,
            # sender_peer_id is only recorded in the reference ground-truth log, not in the 
            # network packet — this is required from Phase 2 onwards (attack analysis, Stem path reconstruction).
            "sender_peer_id": sender_peer_id,
        }, lock=self.lock)

        if self.stem_p is None:
            # Phase 1/2 mode: Simple Flood
            self._forward_flood(packet, exclude_peer_id=sender_peer_id)
            return

        # Phase 3+ mode: Dandelion
        if packet.state == FLUFF:
            self._forward_flood(packet, exclude_peer_id=sender_peer_id)
            return

        # packet.state == STEM
        continue_stem = self.rng.random() < self.stem_p
        next_hop = None
        if continue_stem:
            next_hop = self._forward_stem_single(packet, exclude_peer_id=sender_peer_id)
        if next_hop is None:
            # Either decided to Fluff with probability 1-p, or no eligible neighbor was available
            self._forward_flood(packet.as_fluff(), exclude_peer_id=sender_peer_id)

    # ---- Packet initiation by the origin ----
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
            # Phase 1/2: Origin broadcasts directly to everyone
            self._forward_flood(packet, exclude_peer_id=None)
            return

        # Phase 3+: Origin unconditionally sends to just one random neighbor (STEM)
        next_hop = self._forward_stem_single(packet, exclude_peer_id=None)
        if next_hop is None:
            # If origin has no neighbors (shouldn't happen since minimum degree is guaranteed)
            self._forward_flood(packet.as_fluff(), exclude_peer_id=None)


def run_node(node_id: str, self_addr, peers: Dict[str, Addr], peer_delay: Dict[str, float],
             log_path: str, seed: int, stem_p: Optional[float] = None, lock=None,
             is_spy: bool = False, spy_delay_seed: Optional[int] = None):
    """Entry point executed in an independent process (via multiprocessing).

    is_spy / spy_delay_seed are only meaningful for Phase 5: if is_spy=True, 
    a completely independent RNG (seed=spy_delay_seed) is created solely for 
    drawing intentional delays; in phases 1 to 4 (is_spy=False, default mode), 
    no intentional delay is applied and the behavior remains exactly as before.
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