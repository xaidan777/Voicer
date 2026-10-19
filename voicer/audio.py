"""Чтение/запись аудио через PyAV (ffmpeg внутри пакета) и энергетический анализ."""
from __future__ import annotations

from pathlib import Path

import av
import numpy as np

HOP = 0.01  # шаг анализа энергии, сек

FORMATS = {
    "mp3": {"codec": "libmp3lame", "ext": "mp3", "bit_rate": 192_000},
    "ogg": {"codec": "libopus", "ext": "ogg", "bit_rate": 128_000, "rate": 48_000},
    "opus": {"codec": "libopus", "ext": "opus", "bit_rate": 128_000, "rate": 48_000},
    "m4a": {"codec": "aac", "ext": "m4a", "bit_rate": 192_000},
    "flac": {"codec": "flac", "ext": "flac"},
    "wav": {"codec": "pcm_s16le", "ext": "wav"},
}


def _channels(stream) -> int:
    cc = stream.codec_context
    try:
        return int(cc.layout.nb_channels)
    except AttributeError:
        return int(getattr(cc, "channels", 1) or 1)


def probe(path: str | Path) -> dict:
    with av.open(str(path)) as c:
        st = c.streams.audio[0]
        dur = float(c.duration / 1e6) if c.duration else None
        return {"sr": st.codec_context.sample_rate, "channels": _channels(st), "duration": dur}


def load(path: str | Path, rate: int | None = None) -> tuple[np.ndarray, int]:
    """Возвращает (data[ch, n] float32, sample_rate). Больше 2 каналов — сводится в стерео.

    rate — пересэмплировать в эту частоту (по умолчанию — частота файла).
    """
    with av.open(str(path)) as c:
        st = c.streams.audio[0]
        sr = rate or st.codec_context.sample_rate
        layout = "mono" if _channels(st) == 1 else "stereo"
        rs = av.AudioResampler(format="fltp", layout=layout, rate=sr)
        chunks = []
        for frame in c.decode(st):
            for f in rs.resample(frame):
                chunks.append(f.to_ndarray())
        for f in rs.resample(None):
            chunks.append(f.to_ndarray())
    nch = 1 if layout == "mono" else 2
    data = np.concatenate(chunks, axis=1) if chunks else np.zeros((nch, 0), np.float32)
    return np.ascontiguousarray(data, dtype=np.float32), sr


def mono(data: np.ndarray) -> np.ndarray:
    return data.mean(axis=0) if data.shape[0] > 1 else data[0]


def frame_db(m: np.ndarray, sr: int, hop: float = HOP) -> np.ndarray:
    """RMS-уровень в дБ по кадрам длиной hop."""
    n = max(1, int(sr * hop))
    k = len(m) // n
    if k == 0:
        return np.full(1, -120.0)
    x = m[: k * n].astype(np.float64).reshape(k, n)
    rms = np.sqrt((x * x).mean(axis=1))
    return 20.0 * np.log10(rms + 1e-9)


def levels(db: np.ndarray) -> dict:
    """Уровень шума, речи и порог тишины."""
    floor = max(float(np.percentile(db, 10)), -80.0)
    speech = float(np.percentile(db, 95))
    thr = floor + max(6.0, 0.3 * (speech - floor))
    return {"floor": round(floor, 1), "speech": round(speech, 1), "threshold": round(thr, 1)}


def save(data: np.ndarray, sr: int, path: str | Path, fmt: str = "mp3") -> Path:
    spec = FORMATS[fmt]
    path = Path(path)
    rate = spec.get("rate", sr)
    layout = "mono" if data.shape[0] == 1 else "stereo"
    data = np.clip(data, -1.0, 1.0).astype(np.float32)
    with av.open(str(path), "w") as out:
        st = out.add_stream(spec["codec"], rate=rate)
        st.layout = layout
        if "bit_rate" in spec:
            st.bit_rate = spec["bit_rate"]
        step = 8192
        for i in range(0, data.shape[1], step):
            frame = av.AudioFrame.from_ndarray(np.ascontiguousarray(data[:, i:i + step]),
                                               format="fltp", layout=layout)
            frame.sample_rate = sr
            frame.pts = None
            for pkt in st.encode(frame):
                out.mux(pkt)
        for pkt in st.encode(None):
            out.mux(pkt)
    return path


def peak_buckets(n: int, sr: int, per_sec: int = 100) -> int:
    """Сколько точек огибающей: 100 на секунду (для зума на таймлайне), но не больше 400 тыс."""
    return int(max(2000, min(n * per_sec // max(sr, 1), 400_000)))


def peaks(m: np.ndarray, buckets: int = 3000) -> list[float]:
    """Огибающая для отрисовки волны."""
    if len(m) == 0:
        return []
    buckets = min(buckets, len(m))
    edges = np.linspace(0, len(m), buckets + 1).astype(int)
    a = np.abs(m)
    out = np.maximum.reduceat(a, edges[:-1])
    top = float(out.max()) or 1.0
    return [round(float(v) / top, 3) for v in out]
