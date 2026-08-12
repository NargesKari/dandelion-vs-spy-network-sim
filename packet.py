"""
Packet structure for the network layer.

طبق صورت پروژه: بسته‌ای که روی شبکه رد و بدل می‌شود فقط شامل PacketID
و وضعیت آن (Stem / Fluff) است. شناسه واقعی گره مبدأ و زمان ایجاد بسته
هرگز در این ساختار قرار نمی‌گیرند — آن‌ها فقط در لاگ مرجع شبیه‌ساز
(simulator_log.py) ثبت می‌شوند.
"""

import uuid
from dataclasses import dataclass


STEM = "STEM"
FLUFF = "FLUFF"


def new_packet_id() -> str:
    return str(uuid.uuid4())


@dataclass
class Packet:
    packet_id: str
    state: str = STEM

    def __post_init__(self):
        if self.state not in (STEM, FLUFF):
            raise ValueError(f"invalid packet state: {self.state}")

    def to_dict(self) -> dict:
        return {"packet_id": self.packet_id, "state": self.state}

    @staticmethod
    def from_dict(d: dict) -> "Packet":
        return Packet(packet_id=d["packet_id"], state=d["state"])

    def as_fluff(self) -> "Packet":
        """Return a copy of this packet flipped to FLUFF state."""
        return Packet(packet_id=self.packet_id, state=FLUFF)
