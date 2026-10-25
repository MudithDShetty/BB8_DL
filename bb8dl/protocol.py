"""UDP wire formats shared by the companion app, the mock tools and the ESP32 sketches.

All little endian. Mirrors firmware/bb8_receiver and firmware/bb8_mics; keep in sync.

  command   (app -> drive ESP32, port 5005) 20 B:
            uint16 magic 0xB8D1, uint16 seq, uint32 t_ms, float uL, float uR, uint8 flags, 3 pad
            flags bit0 = e-stop (latched on the ESP32; cleared only by a non-e-stop packet with uL = uR = 0)
  telemetry (drive ESP32 -> app) 32 B:
            uint16 magic 0xB8D2, uint16 last_seq, uint32 t_ms, float uL, float uR,
            float wL, float wR (wheel rad/s, 0 without encoders), float battery_V, uint32 rx_count
  subscribe (app -> mic ESP32, port 5006) 4 B: b"SUB1", resent every second; stream stops 3 s after the last
  audio     (mic ESP32 -> app) 8 B header + N x 4 int16:
            uint16 magic 0xB8A1, uint16 seq, uint32 first_sample_index, then interleaved frames
            (mic0, mic1, mic2, mic3) at 48 kHz. Channel order = config.MIC_ANGLES order.
"""

from __future__ import annotations

import struct
import threading

import numpy as np

CMD_MAGIC, TEL_MAGIC, AUDIO_MAGIC = 0xB8D1, 0xB8D2, 0xB8A1
CMD = struct.Struct("<HHIffB3x")
TEL = struct.Struct("<HHIfffffI")
AUDIO_HDR = struct.Struct("<HHI")
FLAG_ESTOP = 0x01
CMD_PORT, MIC_PORT = 5005, 5006
SUBSCRIBE = b"SUB1"
FRAMES_PER_AUDIO_PACKET = 180  # 8 + 180*8 = 1448 B, below a 1472 B UDP payload

assert CMD.size == 20 and TEL.size == 32


def pack_cmd(seq: int, t_ms: int, uL: float, uR: float, estop: bool = False) -> bytes:
    return CMD.pack(CMD_MAGIC, seq & 0xFFFF, t_ms & 0xFFFFFFFF, float(uL), float(uR), FLAG_ESTOP if estop else 0)


def unpack_cmd(b: bytes) -> dict | None:
    if len(b) != CMD.size:
        return None
    m, seq, t, uL, uR, fl = CMD.unpack(b)
    if m != CMD_MAGIC:
        return None
    return dict(seq=seq, t_ms=t, uL=uL, uR=uR, estop=bool(fl & FLAG_ESTOP))


def pack_tel(last_seq, t_ms, uL, uR, wL, wR, vbat, rx_count) -> bytes:
    return TEL.pack(TEL_MAGIC, last_seq & 0xFFFF, t_ms & 0xFFFFFFFF, uL, uR, wL, wR, vbat, rx_count)


def unpack_tel(b: bytes) -> dict | None:
    if len(b) != TEL.size:
        return None
    m, seq, t, uL, uR, wL, wR, vb, rx = TEL.unpack(b)
    if m != TEL_MAGIC:
        return None
    return dict(last_seq=seq, t_ms=t, uL=uL, uR=uR, wL=wL, wR=wR, vbat=vb, rx_count=rx)


def pack_audio(seq: int, first_sample: int, frames: np.ndarray) -> bytes:
    """frames: (N, 4) int16."""
    return AUDIO_HDR.pack(AUDIO_MAGIC, seq & 0xFFFF, first_sample & 0xFFFFFFFF) + \
        np.ascontiguousarray(frames, dtype="<i2").tobytes()


def unpack_audio(b: bytes) -> tuple[int, int, np.ndarray] | None:
    if len(b) < AUDIO_HDR.size or (len(b) - AUDIO_HDR.size) % 8:
        return None
    m, seq, first = AUDIO_HDR.unpack_from(b)
    if m != AUDIO_MAGIC:
        return None
    return seq, first, np.frombuffer(b, dtype="<i2", offset=AUDIO_HDR.size).reshape(-1, 4)


class AudioRing:
    """Reassembles audio packets into a ring buffer indexed by the ESP32 sample counter.

    Lost packets leave zeros and are counted; a window that overlaps a gap is
    reported as incomplete so the caller can skip it.
    """

    def __init__(self, seconds: float = 2.0, fs: int = 48000):
        self.mutex = threading.Lock()
        self.n = int(seconds * fs)
        self.buf = np.zeros((self.n, 4), np.float32)
        self.valid = np.zeros(self.n, bool)
        self.head = -1          # highest sample index received + 1
        self.lost_packets = 0
        self.last_seq = None

    def push(self, seq: int, first: int, frames: np.ndarray) -> None:
        with self.mutex:
            self._push(seq, first, frames)

    def _push(self, seq: int, first: int, frames: np.ndarray) -> None:
        if self.last_seq is not None and ((seq - self.last_seq) & 0xFFFF) > 1:
            self.lost_packets += ((seq - self.last_seq) & 0xFFFF) - 1
        self.last_seq = seq
        if self.head >= 0 and first + len(frames) > self.head + self.n:
            self.valid[:] = False  # jumped further than the ring: start over
        idx = (first + np.arange(len(frames))) % self.n
        if self.head >= 0 and first > self.head:
            gap = (self.head + np.arange(first - self.head)) % self.n
            self.valid[gap] = False
        self.buf[idx] = frames.astype(np.float32) / 32768.0
        self.valid[idx] = True
        self.head = max(self.head, first + len(frames))

    def window(self, start: int, length: int) -> np.ndarray | None:
        """(4, length) float32 for sample indices [start, start+length), or None if incomplete."""
        with self.mutex:
            if start < self.head - self.n or start + length > self.head:
                return None
            idx = (start + np.arange(length)) % self.n
            if not self.valid[idx].all():
                return None
            return self.buf[idx].T.copy()
