"""План монтажа (какие куски исходника идут в результат) и сборка аудио/текста/субтитров."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Callable

import numpy as np

from . import audio as A
from . import dsp
from .analyze import widen_empty
from .project import Project
from .textproc import SENTENCE_END

MIN_SIL_INSERT = 0.03


def _limit(p: Project, tok: dict) -> float:
    st = p.settings
    return st["max_pause_sentence"] if SENTENCE_END.search(p.text_of(tok["i"])) else st["max_pause"]


def effective_tokens(p: Project) -> tuple[list[dict], dict[int, dict]]:
    """Копии токенов со временем, где границы вырезок заменены заданными вручную (edit.bounds)."""
    bounds = {int(k): v for k, v in p.edit.get("bounds", {}).items()}
    out = []
    for t in p.tokens:
        if "s" in t:
            b = bounds.get(t["i"], {})
            out.append(dict(t, s=b.get("s", t["s"]), e=b.get("e", t["e"])))
    widen_empty(out)  # проекты, где Whisper отдал слову нулевую длину, а анализатор её ещё не чинил
    return out, bounds


def plan(p: Project) -> dict:
    """segments: [[src_s, src_e] | [None, длительность_тишины]], map: {i: [new_s, new_e]}."""
    st = p.settings
    dur = p.duration
    cuts = p.cut_map()
    toks, bounds = effective_tokens(p)
    eff = {t["i"]: t for t in toks}
    # Вырезка с границами, заданными вручную, режет и соседнее слово, если заходит на него.
    manual = [(t["s"], t["e"]) for t in toks if t["i"] in cuts and t["i"] in bounds]
    kept = []
    for t in toks:
        if t["i"] in cuts:
            continue
        for cs, ce in manual:
            if cs <= t["s"] < ce:
                t["s"] = ce
            if cs < t["e"] <= ce:
                t["e"] = cs
        if t["e"] - t["s"] > 0.005:
            kept.append(t)
    if not kept:
        return {"segments": [], "map": {}, "duration": 0.0}
    pauses = {int(k): v for k, v in p.edit["pauses"].items()}
    pieces: list[list] = []

    def add(s: float, e: float) -> None:
        if e - s > 1e-4:
            pieces.append([s, e])

    def add_sil(n: float) -> None:
        if n >= MIN_SIL_INSERT:
            pieces.append([None, n])

    # Начало: до `lead` тишины перед первым словом (без вырезанного).
    first, last = kept[0], kept[-1]
    left_edge = max((t["e"] for t in toks if t["i"] < first["i"]), default=0.0)
    start = max(left_edge, first["s"] - st["lead"])
    if left_edge > 0:
        add_sil(st["lead"] - (first["s"] - start))
    add(start, first["s"])

    for a, b in zip(kept, kept[1:]):
        add(a["s"], a["e"])
        between = [eff[j] for j in range(a["i"] + 1, b["i"]) if j in eff]
        limit = _limit(p, a)
        override = pauses.get(a["i"])
        if not between:
            g = max(0.0, b["s"] - a["e"])
            want = override if override is not None else min(g, limit)
            if want >= g:
                add(a["e"], a["e"] + g / 2)
                add_sil(want - g)
                add(b["s"] - g / 2, b["s"])
            else:
                add(a["e"], a["e"] + want / 2)
                add(b["s"] - want / 2, b["s"])
        else:
            g1 = max(0.0, between[0]["s"] - a["e"])
            g2 = max(0.0, b["s"] - between[-1]["e"])
            natural = g1 if SENTENCE_END.search(p.text_of(a["i"])) else min(g1, g2)
            want = override if override is not None else min(max(natural, st["join_pause"]), limit)
            left_set = "s" in bounds.get(between[0]["i"], {})
            right_set = "e" in bounds.get(between[-1]["i"], {})
            if left_set or right_set:
                # край вырезки задан вручную — звук до него остаётся целиком
                h1 = g1 if left_set else None
                h2 = g2 if right_set else None
                if h1 is None:
                    h1 = max(0.0, min(want - h2, g1))
                if h2 is None:
                    h2 = max(0.0, min(want - h1, g2))
                add(a["e"], a["e"] + h1)
                add(b["s"] - h2, b["s"])
                continue
            h1 = min(want / 2, g1)
            h2 = min(want - h1, g2)
            h1 = min(want - h2, g1)
            add(a["e"], a["e"] + h1)
            add_sil(want - h1 - h2)
            add(b["s"] - h2, b["s"])
    add(last["s"], last["e"])

    right_edge = min((t["s"] for t in toks if t["i"] > last["i"]), default=dur)
    end = min(right_edge, last["e"] + st["tail"], dur or last["e"] + st["tail"])
    add(last["e"], end)

    pieces = _subtract(pieces, p.edit["ranges"])
    pieces = _merge(pieces)
    return {"segments": pieces, "map": _token_map(pieces, kept),
            "duration": round(sum((e - s) if s is not None else e for s, e in pieces), 3)}


def _subtract(pieces: list[list], ranges: list) -> list[list]:
    if not ranges:
        return pieces
    out = []
    for s, e in pieces:
        if s is None:
            out.append([s, e])
            continue
        cur = [[s, e]]
        for rs, re_, _ in ranges:
            nxt = []
            for a, b in cur:
                if re_ <= a or rs >= b:
                    nxt.append([a, b])
                    continue
                if rs > a:
                    nxt.append([a, rs])
                if re_ < b:
                    nxt.append([re_, b])
            cur = nxt
        out.extend(x for x in cur if x[1] - x[0] > 1e-4)
    return out


def _merge(pieces: list[list]) -> list[list]:
    out: list[list] = []
    for s, e in pieces:
        if out and s is not None and out[-1][0] is not None and abs(out[-1][1] - s) < 1e-4:
            out[-1][1] = e
        elif out and s is None and out[-1][0] is None:
            out[-1][1] += e
        else:
            out.append([s, e])
    return [[round(s, 4) if s is not None else None, round(e, 4)] for s, e in out]


def _token_map(pieces: list[list], kept: list[dict]) -> dict[int, list[float]]:
    spans, pos = [], 0.0
    for s, e in pieces:
        if s is None:
            pos += e
        else:
            spans.append((s, e, pos))
            pos += e - s
    res, k = {}, 0

    def locate(t: float, k: int) -> tuple[float | None, int]:
        while k < len(spans) and spans[k][1] < t - 1e-4:
            k += 1
        if k < len(spans) and spans[k][0] - 1e-4 <= t:
            return spans[k][2] + (t - spans[k][0]), k
        return None, k

    for t in kept:
        ns, k = locate(t["s"], k)
        ne, _ = locate(t["e"], k)
        if ns is not None:
            res[t["i"]] = [round(ns, 3), round(ne if ne is not None else ns, 3)]
    return res


# --- сборка аудио -------------------------------------------------------------

def denoised_source(p: Project, data: np.ndarray, sr: int, strength: float,
                    progress: Callable[[float, str], None] | None = None) -> np.ndarray:
    """Исходник после шумоподавления; результат кэшируется в cache/denoise-<сила>.flac."""
    from . import enhance

    cache = p.dir / "cache"
    f = cache / f"denoise-{int(round(strength))}.flac"
    if f.exists():
        y, ysr = A.load(f)
        if ysr == sr and y.shape == data.shape:
            return y
    if not enhance.available():
        raise ValueError("Шумоподавление не установлено: uv pip install deepfilternet torch torchaudio")
    say = progress or (lambda frac, msg: None)
    y = enhance.denoise(data, sr, strength, progress=lambda frac: say(frac, "шумоподавление"))
    cache.mkdir(exist_ok=True)
    for old in cache.glob("denoise-*.flac"):
        old.unlink()
    A.save(y, sr, f, "flac")
    return y


def render(p: Project, fmt: str | None = None, out: str | Path | None = None,
           progress: Callable[[float, str], None] | None = None) -> dict:
    if p.kind != "audio":
        raise ValueError("Рендер аудио только для аудиопроектов; для текста: voicer export --txt")
    st = p.settings
    fmt = fmt or st.get("format", "mp3")
    if fmt not in A.FORMATS:
        raise ValueError(f"Формат: {', '.join(A.FORMATS)}")
    data, sr = A.load(p.source_path)
    # Без «Очистки голоса» — только монтаж: ни фильтров, ни громкости, ни лимитера.
    clean = bool(st.get("voice_clean"))
    if clean and st.get("denoise"):
        data = denoised_source(p, data, sr, float(st["denoise"]), progress)
    pl = plan(p)
    y = assemble(data, sr, pl["segments"], st["crossfade_ms"], (p.data.get("levels") or {}).get("room"))
    if clean:
        y = dsp.studio_chain(y, sr, st)
        y = dsp.loudness(y, sr, st.get("lufs"), st.get("limit"))
    if out is None:
        (p.dir / "out").mkdir(exist_ok=True)
        out = p.dir / "out" / f"{p.id}.{A.FORMATS[fmt]['ext']}"
    A.save(y, sr, out, fmt)
    save_out_peaks(p, Path(out).name, A.mono(y), sr)
    return {"path": str(out), "duration": round(y.shape[1] / sr, 3), "original": p.duration,
            "segments": len(pl["segments"])}


def save_out_peaks(p: Project, name: str, m: np.ndarray, sr: int) -> dict:
    """Огибающая результата для второй волны в интерфейсе."""
    import json

    data = {"name": name, "duration": round(len(m) / sr, 3), "peaks": A.peaks(m, A.peak_buckets(len(m), sr))}
    (p.dir / "peaks_out.json").write_text(json.dumps(data), encoding="utf-8")
    return data


def record_render(p: Project, r: dict, by: str) -> None:
    """Запоминает последний рендер (интерфейс по rev видит, что он устарел) и сохраняет проект."""
    name = Path(r["path"]).name
    p.data["last_render"] = {"name": name, "rev": int(p.data.get("rev", 0)) + 1, "duration": r["duration"]}
    m, s = divmod(r["duration"], 60)
    p.log(by, f"рендер {name}, {int(m):02d}:{s:04.1f}")
    p.save()


def assemble(data: np.ndarray, sr: int, segments: list[list], crossfade_ms: float,
             room: list | None = None) -> np.ndarray:
    ch, N = data.shape
    half = max(0, int(sr * crossfade_ms / 1000) // 2)
    lens = [int(round(((e - s) if s is not None else e) * sr)) for s, e in segments]
    total = sum(lens)
    y = np.zeros((ch, total + 2 * half + 1), np.float32)

    tone = np.zeros((ch, 1), np.float32)
    if room:
        rs, re_ = int(room[0] * sr), int(room[1] * sr)
        if re_ > rs:
            tone = data[:, rs:re_]

    def src(a: int, b: int) -> np.ndarray:
        seg = np.zeros((ch, b - a), np.float32)
        lo, hi = max(a, 0), min(b, N)
        if hi > lo:
            seg[:, lo - a:hi - a] = data[:, lo:hi]
        return seg

    pos = 0
    last = len(segments) - 1
    for k, ((s, _), n) in enumerate(zip(segments, lens)):
        left = half if k > 0 else 0
        right = half if k < last else 0
        if s is not None:
            s0 = int(round(s * sr))
            chunk = src(s0 - left, s0 + n + right)
        else:
            reps = (n + left + right) // tone.shape[1] + 1
            chunk = np.tile(tone, reps)[:, :n + left + right].copy()
        m = chunk.shape[1]
        fi, fo = min(2 * left, m // 2), min(2 * right, m // 2)
        if fi:
            chunk[:, :fi] *= np.sin(np.linspace(0, np.pi / 2, fi, dtype=np.float32))
        if fo:
            chunk[:, m - fo:] *= np.cos(np.linspace(0, np.pi / 2, fo, dtype=np.float32))
        at = pos - left
        y[:, at:at + m] += chunk
        pos += n
    y = y[:, :total]
    fin, fout = int(0.005 * sr), int(0.01 * sr)
    if total > 2 * (fin + fout):
        y[:, :fin] *= np.linspace(0, 1, fin, dtype=np.float32)
        y[:, -fout:] *= np.linspace(1, 0, fout, dtype=np.float32)
    return y


# --- текст и субтитры ----------------------------------------------------------

def edited_words(p: Project, original: bool = False) -> list[dict]:
    cuts = {} if original else p.cut_map()
    return [t for t in p.tokens if t.get("kind", "word") == "word" and t["i"] not in cuts]


_TRAIL = re.compile(r"[.,!?;:…]+$")


def _strength(punct: str) -> int:
    return 2 if any(c in punct for c in ".!?…") else (1 if punct else 0)


def display_texts(p: Project, original: bool = False) -> dict[int, str]:
    """Тексты оставшихся слов с поправленной пунктуацией и заглавными на стыках вырезок.

    «было, как бы, много» → «было много»; «работать, вот.» → «работать.»; «Ну, мы» → «Мы».
    """
    cuts = {} if original else p.cut_map()
    out: dict[int, str] = {}
    prev: int | None = None
    pending: str | None = None  # самая «сильная» пунктуация из вырезанного куска
    for t in p.tokens:
        if t.get("kind", "word") != "word":
            continue
        txt = p.text_of(t["i"])
        if t["i"] in cuts:
            m = _TRAIL.search(txt)
            tp = m.group(0) if m else ""
            if pending is None or _strength(tp) > _strength(pending):
                pending = tp
            continue
        if pending is not None:
            if prev is not None:
                m = _TRAIL.search(out[prev])
                prev_tp = m.group(0) if m else ""
                if _strength(pending) == 2 and _strength(prev_tp) < 2:
                    out[prev] = _TRAIL.sub("", out[prev]) + pending
                elif pending and _strength(pending) == 1 and prev_tp == ",":
                    out[prev] = out[prev][:-1]
            if prev is None or SENTENCE_END.search(out[prev]):
                txt = txt[:1].upper() + txt[1:]
        out[t["i"]] = txt
        prev, pending = t["i"], None
    if pending and prev is not None and _strength(pending) == 2:
        out[prev] = _TRAIL.sub("", out[prev]) + pending
    return out


def export_text(p: Project, original: bool = False) -> str:
    words = edited_words(p, original)
    texts = display_texts(p, original)
    out, line = [], []
    for k, t in enumerate(words):
        line.append(texts[t["i"]])
        nxt = words[k + 1] if k + 1 < len(words) else None
        gap = (nxt["s"] - t["e"]) if nxt is not None and "s" in t else 0
        if SENTENCE_END.search(line[-1]) and (gap >= 1.5 or nxt is None):
            out.append(" ".join(line))
            line = []
    if line:
        out.append(" ".join(line))
    return "\n".join(out) + "\n"


def _ts(t: float, sep: str = ",") -> str:
    ms = int(round(t * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}{sep}{ms:03d}"


def export_subs(p: Project, original: bool = False, vtt: bool = False,
                max_chars: int = 42, max_dur: float = 5.0) -> str:
    if p.kind != "audio":
        raise ValueError("Субтитры только для аудиопроектов")
    words = edited_words(p, original)
    texts = display_texts(p, original)
    tmap = None if original else plan(p)["map"]
    timed = []
    for t in words:
        if tmap is None:
            timed.append((t["s"], t["e"], texts[t["i"]]))
        elif t["i"] in tmap:
            timed.append((*tmap[t["i"]], texts[t["i"]]))
    cues, cur = [], []
    for k, (s, e, w) in enumerate(timed):
        cur.append((s, e, w))
        text = " ".join(x[2] for x in cur)
        nxt = timed[k + 1] if k + 1 < len(timed) else None
        if (nxt is None or SENTENCE_END.search(w) or len(text) + len(nxt[2]) + 1 > max_chars
                or nxt[1] - cur[0][0] > max_dur or nxt[0] - e > 1.0):
            cues.append((cur[0][0], cur[-1][1], text))
            cur = []
    sep = "." if vtt else ","
    lines = ["WEBVTT", ""] if vtt else []
    for n, (s, e, text) in enumerate(cues, 1):
        if not vtt:
            lines.append(str(n))
        lines += [f"{_ts(s, sep)} --> {_ts(max(e, s + 0.3), sep)}", text, ""]
    return "\n".join(lines)
