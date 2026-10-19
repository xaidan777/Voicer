"""Уточнение границ слов по энергии сигнала и поиск нераспознанных звуков (заминки, вдохи).

Whisper часто «размазывает» паузу по соседним словам: «результат» 8.1–9.1, «и» 9.1–10.3, хотя между ними
почти секунда тишины. Поэтому паузы ищутся по реальной энергии сигнала, в том числе внутри слов.
"""
from __future__ import annotations

import bisect

import numpy as np

from .audio import HOP, frame_db, levels

MIN_SIL = 0.08     # тишина короче этого не считается паузой (смычки согласных)
LONG_SIL = 0.15    # тишина длиннее этого — наверняка пауза, даже если Whisper отдал её слову
MIN_SOUND = 0.10   # озвученный фрагмент между словами длиннее этого — «звук»
BLIP = 0.04        # всплески шума короче этого и тише порог+6 дБ внутри тишины — не речь
EDGE = 0.04        # минимум, который остаётся от слова при сдвиге границ
EDGE_EXT = 0.15    # насколько можно расширить первое/последнее слово по энергии
HES_MIN = 0.25     # тянущийся ровный звук внутри слова длиннее этого — заминка «э-э», «а-а»
MIN_WORD = 0.03    # слово короче — Whisper не дал ему времени («друзья» 2.17–2.17)
WIDEN = 0.3        # насколько такое слово растягивается в промежутки к соседям


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    m = np.concatenate([[False], mask, [False]]).astype(np.int8)
    d = np.diff(m)
    return list(zip(np.where(d == 1)[0].tolist(), np.where(d == -1)[0].tolist()))


def silence_mask(db: np.ndarray, thr: float) -> np.ndarray:
    sil = db < thr
    blip = int(round(BLIP / HOP))
    for a, b in _runs(~sil):
        if b - a < blip and a > 0 and b < len(sil) and db[a:b].max() < thr + 6:
            sil[a:b] = True
    return sil


def _flux(m: np.ndarray, sr: int, a: float, b: float) -> np.ndarray:
    """Скорость изменения тембра по кадрам HOP на участке [a, b]: у «э-э-э», «а-а-а» она низкая."""
    from scipy.ndimage import uniform_filter1d
    from scipy.signal import resample_poly, stft

    lo, hi = max(0, int((a - 0.05) * sr)), min(len(m), int((b + 0.05) * sr))
    x = resample_poly(m[lo:hi].astype(np.float32), 16000, sr) if sr != 16000 else m[lo:hi]
    hop, n = 160, 480
    if len(x) < n * 2:
        return np.full(int(round((b - a) / HOP)), 9.0)
    f, _, z = stft(x, fs=16000, nperseg=n, noverlap=n - hop, boundary=None, padded=False)
    s = np.log(np.abs(z[(f > 80) & (f < 4000)]) ** 2 + 1e-10)
    s = uniform_filter1d(s, 9, axis=0)  # огибающая спектра без гармоник
    s -= s.mean(axis=0)
    k = 4
    fl = np.full(s.shape[1], 9.0)
    fl[k:] = np.sqrt(((s[:, k:] - s[:, :-k]) ** 2).mean(axis=0))
    # кадр j ≈ время lo/sr + j*HOP + 0.015; приводим к сетке [a, b)
    start = int(round(((a - lo / sr) - 0.015) / HOP))
    out = np.full(int(round((b - a) / HOP)), 9.0)
    src = fl[max(0, start):max(0, start) + len(out)]
    out[:len(src)] = src
    return out


def _voicing(m: np.ndarray, sr: int, a: float, b: float) -> float:
    """Доля «голоса» на участке: медиана нормированной автокорреляции в диапазоне 70–400 Гц."""
    from scipy.signal import butter, sosfiltfilt

    lo, hi = max(0, int(a * sr)), min(len(m), int(b * sr))
    if hi - lo < int(0.05 * sr):
        return 0.0
    x = sosfiltfilt(butter(4, [100, 3000], "bandpass", fs=sr, output="sos"), m[lo:hi])
    w, vals = int(0.04 * sr), []
    for i in range(0, len(x) - w, w // 2):
        seg = x[i:i + w] - x[i:i + w].mean()
        e = float((seg * seg).sum())
        if e > 1e-12:
            ac = np.correlate(seg, seg, "full")[w - 1:] / e
            vals.append(float(ac[int(sr / 400):int(sr / 70)].max()))
    return float(np.median(vals)) if vals else 0.0


def is_hesitation(s0: float, e0: float, m: np.ndarray, sr: int, db: np.ndarray, thr: float,
                  steady_thr: float) -> bool:
    """Отдельная заминка «э-э-э», «а-а-а»: ровный звонкий звук, отделённый провалом с обеих сторон."""
    a, b = int(round(s0 / HOP)), int(round(e0 / HOP))
    if (b - a) * HOP < HES_MIN:
        return False
    voiced = db[a:b] > thr + 6
    fl = np.convolve(_flux(m, sr, s0, e0), np.ones(5) / 5, mode="same")[:b - a]
    if np.mean((fl < steady_thr) & voiced[:len(fl)]) < 0.55:
        return False
    level = float(np.percentile(db[a:b], 75))
    before = db[max(0, a - 12):a + 1]
    after = db[max(0, b - 1):b + 12]
    if not len(before) or not len(after) or before.min() > level - 9 or after.min() > level - 9:
        return False
    return _voicing(m, sr, s0, e0) >= 0.5


def _steady_threshold(m: np.ndarray, sr: int, db: np.ndarray, thr: float) -> float:
    """«Ровно» = изменение тембра ниже типичного для этой записи (по первым 2 минутам речи)."""
    ref = _flux(m, sr, 0.0, min(len(m) / sr, 120.0))
    n = min(len(ref), len(db))
    vm = db[:n] > thr + 6
    return float(np.percentile(ref[:n][vm], 40)) if vm.sum() >= 50 else 0.0


def _steady_islands(m: np.ndarray, sr: int, db: np.ndarray, thr: float,
                    steady_thr: float) -> list[tuple[float, float]]:
    """Отдельные ровные звонкие звуки (кандидаты в «э-э-э», «а-а-а»): провал громкости до и после."""
    voiced_thr = thr + 6
    total = len(db) * HOP
    fl = np.concatenate([_flux(m, sr, a, min(a + 60.0, total)) for a in np.arange(0.0, total, 60.0)])[:len(db)]
    fl = np.convolve(fl, np.ones(5) / 5, mode="same")
    voiced = db[:len(fl)] > voiced_thr
    steady = (fl < steady_thr) & voiced
    for g0, g1 in _runs(~steady):
        if g1 - g0 <= 3 and g0 > 0 and g1 < len(steady):
            steady[g0:g1] = True
    out = []
    for r0, r1 in _runs(steady):
        if (r1 - r0) * HOP < 0.15:
            continue
        level = float(db[r0:r1].mean())
        ok = lambda i: 0 <= i < len(fl) and fl[i] < steady_thr * 1.5 and db[i] >= level - 8 and voiced[i]
        lo, hi = r0, r1
        while r0 - lo < 25 and ok(lo - 1):
            lo -= 1
        while hi - r1 < 25 and ok(hi):
            hi += 1
        if (hi - lo) * HOP < HES_MIN:
            continue
        before, after = db[max(0, lo - 12):lo + 1], db[max(0, hi - 1):hi + 12]
        if before.min() > level - 9 or after.min() > level - 9:
            continue  # слитно со словом: растянутое окончание «мыыы», не трогаем
        if _voicing(m, sr, lo * HOP, hi * HOP) >= 0.5:
            out.append((round(lo * HOP, 3), round(hi * HOP, 3)))
    return out


def stt_fillers(toks: list[dict], m: np.ndarray, sr: int, heard) -> list[dict]:
    """Заминки, которых нет в тексте: каждый отдельный ровный звук проверяется распознавателем.

    heard(s, e) -> список слов, которые STT слышит на этом участке. Если среди них есть слово из
    текста рядом — это слово (протяжное «Ээээто», «нууу»), оно остаётся. Если нет — STT слова не
    видит, значит это «э-э-э» / «а-а-а»: участок становится заминкой, слово сдвигается к своему звуку.
    """
    from difflib import SequenceMatcher

    from .textproc import is_strict_filler, norm

    db = frame_db(m, sr)
    thr = levels(db)["threshold"]
    steady_thr = _steady_threshold(m, sr, db, thr)
    if not steady_thr:
        return toks
    words = [t for t in toks if t.get("kind", "word") == "word"]
    sounds = [t for t in toks if t.get("kind") == "sound"]
    added = []
    for s0, e0 in _steady_islands(m, sr, db, thr, steady_thr):
        if any(min(e0, t["e"]) - max(s0, t["s"]) > 0.5 * (e0 - s0) for t in sounds):
            continue  # уже найден как звук между словами
        if any(is_strict_filler(t["t"]) and min(e0, t["e"]) - max(s0, t["s"]) > 0.3 * (e0 - s0) for t in words):
            continue  # Whisper сам записал это как «э-э-э» — это уже паразит-слово
        near = [norm(t["t"]) for t in words if t["e"] > s0 - 0.7 and t["s"] < e0 + 0.7]
        got = [norm(w) for w in heard(s0, e0)]
        got = [w for w in got if w and not is_strict_filler(w)]
        if any(SequenceMatcher(None, g, n).ratio() >= 0.6 or (len(g) >= 2 and (n.startswith(g) or g.startswith(n)))
               for g in got for n in near if n):
            continue
        for t in words:
            if t["e"] <= s0 or t["s"] >= e0:
                continue
            a, b = int(round(t["s"] / HOP)), int(round(t["e"] / HOP))
            i0, i1 = int(round(s0 / HOP)), int(round(e0 / HOP))
            left = (db[a:max(a, i0)] > thr).sum()
            right = (db[min(b, i1):b] > thr).sum()
            if right >= left:
                t["s"], t["e"] = e0, max(t["e"], round(e0 + 0.05, 3))
            else:
                t["s"], t["e"] = min(t["s"], round(s0 - 0.05, 3)), s0
        added.append({"t": "~", "s": s0, "e": e0, "kind": "sound", "hes": True,
                      "db": round(float(db[int(s0 / HOP):int(e0 / HOP)].mean()), 1)})
    out = sorted(toks + added, key=lambda x: x["s"])
    for i, t in enumerate(out):
        t["i"] = i
    return out


def widen_empty(toks: list[dict]) -> None:
    """Слово без длительности растягивается на промежутки к соседям (не дальше WIDEN): иначе монтаж
    примет его место за паузу и сожмёт вместе со словом. Токены — по порядку времени."""
    for k, t in enumerate(toks):
        if "s" not in t or t.get("kind", "word") != "word" or t["e"] - t["s"] >= MIN_WORD:
            continue
        prev = next((x["e"] for x in reversed(toks[:k]) if "s" in x), 0.0)
        nxt = next((x["s"] for x in toks[k + 1:] if "s" in x), t["e"] + WIDEN)
        t["s"] = round(max(min(prev, t["s"]), t["s"] - WIDEN), 3)
        t["e"] = round(min(max(nxt, t["e"]), t["e"] + WIDEN), 3)


def refine(tokens: list[dict], m: np.ndarray, sr: int) -> tuple[list[dict], dict]:
    """Возвращает (токены по времени с добавленными звуками, уровни сигнала)."""
    toks = sorted((dict(t, kind=t.get("kind", "word")) for t in tokens if t.get("kind", "word") == "word"),
                  key=lambda x: x["s"])
    for t in toks:
        t["e"] = max(t["e"], t["s"])
    widen_empty(toks)
    db = frame_db(m, sr)
    lv = levels(db)
    sil = silence_mask(db, lv["threshold"])
    min_fr = int(round(MIN_SIL / HOP))
    long_fr = int(round(LONG_SIL / HOP))
    runs = [r for r in _runs(sil) if r[1] - r[0] >= min_fr]
    short_runs = [r for r in _runs(sil) if r[1] - r[0] >= 3]  # провалы от 30 мс
    starts = [r[0] for r in runs]

    def fr(t: float) -> int:
        return int(round(t / HOP))

    def runs_in(lo: int, hi: int) -> list[tuple[int, int]]:
        out = []
        i = max(bisect.bisect_left(starts, lo) - 1, 0)
        while i < len(runs) and runs[i][0] < hi:
            a, b = max(runs[i][0], lo), min(runs[i][1], hi)
            if b - a >= min_fr:
                out.append((a, b))
            i += 1
        return out

    # Длинная тишина у края слова — не часть слова: «Готовое» 0.77–1.38 при речи с 0.96.
    for t in toks:
        s, e = fr(t["s"]), fr(t["e"])
        rr = [r for r in runs_in(s - 2, e + 2) if r[1] - r[0] >= long_fr]
        if not rr or (rr[0][0] <= s and rr[0][1] >= e):
            continue  # тишины нет или слово целиком в тишине (Whisper промахнулся) — не трогаем
        if rr[0][0] <= s and rr[0][1] < e - EDGE / HOP:
            t["s"] = round(rr[0][1] * HOP, 3)
        if rr[-1][1] >= e and rr[-1][0] > fr(t["s"]) + EDGE / HOP:
            t["e"] = round(rr[-1][0] * HOP, 3)

    sounds = []
    for a, b in zip(toks, toks[1:]):
        if a["e"] > b["s"]:
            a["e"] = b["s"] = round((a["e"] + b["s"]) / 2, 3)
        # 1) тишина около границы, которую дал Whisper
        lo = fr(max(a["e"] - 0.05, (a["s"] + a["e"]) / 2))
        hi = fr(min(b["s"] + 0.05, (b["s"] + b["e"]) / 2))
        rr = runs_in(lo, hi) if hi > lo else []
        b_loose = any(r[1] - r[0] >= long_fr for r in runs_in(fr(b["s"] + EDGE), fr(b["e"] - EDGE)))
        if b_loose:
            # 2) Whisper отдал паузу следующему слову («и» 9.1–10.3 при паузе 9.3–10.2):
            #    граница — длинная тишина, ближайшая к его границе
            span = runs_in(fr(a["s"] + EDGE), fr(b["e"] - EDGE))
            long = [r for r in span if r[1] - r[0] >= long_fr]
            c = fr((a["e"] + b["s"]) / 2)
            best = min(long, key=lambda r: 0 if r[0] <= c <= r[1] else min(abs(r[0] - c), abs(r[1] - c)))
            rr = sorted([r for r in rr if not (r[0] < best[1] and best[0] < r[1])] + [best])
        elif not rr:
            # 3) следующее слово размечено плотно — его начало верное; граница по короткому провалу
            #    рядом с ним («ну|очень»: провал 7.02–7.12 перед «очень» с 7.10)
            dips = [r for r in short_runs if r[1] > fr(b["s"] - 0.12) and r[0] < fr(b["s"] + 0.06)]
            if dips:
                d = min(dips, key=lambda r: abs((r[0] + r[1]) / 2 - fr(b["s"])))
                if d[0] * HOP > a["s"] + 0.02 and d[1] * HOP < b["e"] - 0.02:
                    a["e"], b["s"] = round(d[0] * HOP, 3), round(d[1] * HOP, 3)
            continue
        if not rr:
            continue
        a["e"] = round(max(rr[0][0] * HOP, a["s"] + 0.02), 3)
        b["s"] = round(max(min(rr[-1][1] * HOP, b["e"] - 0.02), a["e"]), 3)
        # Озвученное между первой и последней тишиной — то, что Whisper не записал.
        for (_, e1), (s2, _) in zip(rr, rr[1:]):
            if s2 - e1 >= MIN_SOUND / HOP:
                sounds.append({"t": "~", "s": round(e1 * HOP, 3), "e": round(s2 * HOP, 3),
                               "kind": "sound", "db": round(float(db[e1:s2].mean()), 1)})

    if toks:
        first, last = toks[0], toks[-1]
        rr = [r for r in runs_in(max(fr(first["s"] - 1.0), 0), fr(first["e"] - EDGE))
              if r[0] <= fr(first["s"]) + 5]
        if rr and rr[-1][1] * HOP >= first["s"] - EDGE_EXT:
            first["s"] = round(min(rr[-1][1] * HOP, first["e"] - 0.02), 3)
        rr = [r for r in runs_in(fr(last["s"] + EDGE), fr(last["e"] + 1.0)) if r[1] >= fr(last["e"]) - 5]
        if rr and rr[0][0] * HOP <= last["e"] + EDGE_EXT:
            last["e"] = round(max(rr[0][0] * HOP, last["s"] + 0.02), 3)

    steady_thr = _steady_threshold(m, sr, db, lv["threshold"])
    if steady_thr:
        for snd in sounds:
            if is_hesitation(snd["s"], snd["e"], m, sr, db, lv["threshold"], steady_thr):
                snd["hes"] = True

    # Самый тихий участок — «тон комнаты» для вставки тишины.
    w = max(1, int(0.4 / HOP))
    if len(db) > w:
        k = int(np.argmin(np.convolve(db, np.ones(w) / w, mode="valid")))
        lv["room"] = [round(k * HOP, 3), round((k + w) * HOP, 3)]

    out = sorted(toks + sounds, key=lambda x: x["s"])
    for i, t in enumerate(out):
        t["i"] = i
    return out, lv
