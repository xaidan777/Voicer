"""Пак: несколько записей → одна дорожка; поиск дублей с приоритетом позднего.

Дубли ищутся не по транскрипту: увидев повтор, Whisper часто не пишет его, а растягивает на него соседнее
слово («прочей» на 4 секунды, в которых целиком второй дубль фразы). Поэтому речь режется на фразы по
паузам, каждая фраза распознаётся отдельно, и если фраза начинает заново то, что говорилось чуть раньше,
всё от начала раннего дубля до начала позднего вырезается.
"""
from __future__ import annotations

import re
import tempfile
from difflib import SequenceMatcher
from pathlib import Path
from typing import Callable

import numpy as np

from . import audio as A
from . import config
from .analyze import _runs, silence_mask
from .ops import _ids_label
from .textproc import SENTENCE_END, is_strict_filler, norm

GAP = 0.5            # тишина между файлами пака, с
FADE = 0.005         # микрофейд на краях файлов, чтобы стыки не щёлкали
PHRASE_GAP = 0.3     # пауза, которая делит речь на фразы, с
MIN_PHRASE = 0.12    # участок речи короче — не фраза (щелчок, вдох)
WINDOW = 30.0        # поздний дубль начинается не дальше этого от начала раннего, с
SHORT_WINDOW = 6.0   # то же для фальстарта из 1–2 слов
PRE = 0.15           # сколько оставить перед поздним дублем, с
HEAD = 12            # сколько первых слов дублей сравнивать
HALLUCINATIONS = ("субтитр", "продолжение следует", "спасибо за просмотр", "dimatorzok")

Progress = Callable[[float, str], None] | None


def natural_key(name: str) -> list:
    """Порядок файлов пака: «audio_2» раньше «audio_10»."""
    return [int(x) if x.isdigit() else x.lower() for x in re.split(r"(\d+)", name)]


def concat(files: list[Path], out: Path, progress: Progress = None) -> list[dict]:
    """Склеивает файлы (в заданном порядке) в один FLAC с GAP тишины между ними. Возвращает [{name, s, e}]."""
    info = [A.probe(f) for f in files]
    sr = max(x["sr"] for x in info)
    nch = 2 if any(x["channels"] > 1 for x in info) else 1
    gap = np.zeros((nch, int(GAP * sr)), np.float32)
    parts, chunks, t = [], [], 0.0
    for k, f in enumerate(files):
        if progress:
            progress(k / len(files), f"склейка: {f.name}")
        data, _ = A.load(f, rate=sr)
        if data.shape[0] != nch:
            data = np.repeat(data, nch, axis=0)
        n = min(int(FADE * sr), data.shape[1] // 2)
        if n:
            ramp = np.linspace(0, 1, n, dtype=np.float32)
            data[:, :n] *= ramp
            data[:, -n:] *= ramp[::-1]
        if chunks:
            chunks.append(gap)
            t += GAP
        dur = data.shape[1] / sr
        parts.append({"name": f.name, "s": round(t, 3), "e": round(t + dur, 3)})
        chunks.append(data)
        t += dur
    A.save(np.concatenate(chunks, axis=1), sr, out, "flac")
    return parts


# --- поиск дублей ------------------------------------------------------------

# Латиница для сравнения пишется кириллицей: Whisper пишет то «Asset», то «Ассет».
_LAT = str.maketrans({"a": "а", "b": "б", "c": "к", "d": "д", "e": "е", "f": "ф", "g": "г", "h": "х", "i": "и",
                      "j": "дж", "k": "к", "l": "л", "m": "м", "n": "н", "o": "о", "p": "п", "q": "к", "r": "р",
                      "s": "с", "t": "т", "u": "у", "v": "в", "w": "в", "x": "кс", "y": "и", "z": "з"})


def _key(word: str) -> str:
    return re.sub(r"[-–—]", "", norm(word)).translate(_LAT)


def _words(raw: list[str]) -> list[str]:
    """Слова фразы для сравнения: «3D-генератору» → «3д», «генератору»; «GameReady» → «гаме», «реади»."""
    out = []
    for w in raw:
        if is_strict_filler(w):
            continue
        w = re.sub(r"([a-zа-яё])([A-ZА-ЯЁ])", r"\1 \2", w)
        out += [n.translate(_LAT) for n in (norm(x) for x in re.split(r"[-–—/\s]", w)) if n]
    return out


def _same(a: str, b: str) -> bool:
    if a == b:
        return True
    if min(len(a), len(b)) < 4:
        return False
    return SequenceMatcher(None, a, b).ratio() >= 0.75  # «представить» ≈ «предоставить»


def _lcs(a: list[str], b: list[str]) -> int:
    dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            dp[i + 1][j + 1] = dp[i][j] + 1 if _same(x, y) else max(dp[i][j + 1], dp[i + 1][j])
    return dp[-1][-1]


def is_retake(early: list[str], late: list[str]) -> bool:
    """Начинает ли речь `late` заново то, что сказано в `early` (нормализованные слова)."""
    if not early or not late or not _same(early[0], late[0]):
        return False
    if len(early) < 3:  # фальстарт «Это не…» — целиком совпадает с началом позднего
        return len(late) > len(early) and all(_same(x, y) for x, y in zip(early, late))
    a = early[:HEAD]
    m = _lcs(a, late[:len(a) + 4])
    # совпали два первых слова — достаточно половины: дубль часто перефразирован
    need = 0.45 if len(late) > 1 and _same(a[1], late[1]) else 0.65
    return m >= 3 and m >= need * len(a)


def _phrases(m: np.ndarray, sr: int) -> list[tuple[float, float]]:
    db = A.frame_db(m, sr)
    sil = silence_mask(db, A.levels(db)["threshold"])
    out: list[list[float]] = []
    for a, b in _runs(~sil):
        s, e = a * A.HOP, b * A.HOP
        if out and s - out[-1][1] < PHRASE_GAP:
            out[-1][1] = e
        else:
            out.append([s, e])
    return [(s, e) for s, e in out if e - s >= MIN_PHRASE]


def find_retakes(path: Path, tokens: list[dict], language: str | None = None, model: str | None = None,
                 progress: Progress = None) -> dict:
    """{phrases: [{s, e, raw, w}], spans: [(начало раннего дубля, начало позднего, i, j)]}."""
    from .transcribe import heard_words

    wav, sr = A.load(path)
    spans = sorted((t["s"], t["e"]) for t in tokens if "s" in t)

    def covered(s: float, e: float) -> bool:
        # где распознаватель и анализатор ничего не нашли (шум, тишина), Whisper только галлюцинирует
        ov = sum(min(e, te) - max(s, ts) for ts, te in spans if te > s and ts < e)
        return ov >= min(0.1, (e - s) / 2)

    heard = heard_words(path, language, model)
    ph = _phrases(A.mono(wav), sr)
    phrases = []
    for k, (s, e) in enumerate(ph):
        if progress:
            progress(k / max(len(ph), 1), "поиск дублей")
        raw = heard(s, e) if covered(s, e) else []
        if any(h in " ".join(raw).lower() for h in HALLUCINATIONS):
            raw = []
        phrases.append({"s": round(s, 2), "e": round(e, 2), "raw": raw, "w": _words(raw)})

    found = []  # {s, e, i, j, early}: вырезать [s, e), поздний дубль — фраза j
    for i, p in enumerate(phrases):
        if not p["w"]:
            continue
        j = i + 1
        while j < len(phrases) and phrases[j]["s"] - p["s"] <= WINDOW:
            j += 1
        for j in range(j - 1, i, -1):  # самый поздний дубль
            q = phrases[j]
            if not q["w"]:
                continue
            early = [w for x in phrases[i:j] for w in x["w"]]
            if len(early) < 3 and q["s"] - p["s"] > SHORT_WINDOW:
                continue
            late = []
            for x in phrases[j:]:
                late += x["w"]
                if len(late) >= HEAD + 4:
                    break
            if is_retake(early, late):
                found.append({"s": p["s"], "e": q["s"], "i": i, "j": j,
                              "early": " ".join(w for x in phrases[i:j] for w in x["raw"])})
                break
    found += _restarts_midphrase(phrases, tokens)

    # Поздний дубль, который сам ничем не переписан, — окончательный: через него не режем
    # («…много советов» | «Процесс их генерации…» | «…множество советов» — второе остаётся).
    final = {f["j"] for f in found} - {f["i"] for f in found if f["i"] is not None}
    for f in found:
        f["e"] = min([f["e"]] + [phrases[k]["s"] for k in final if f["s"] < phrases[k]["s"] < f["e"]])
    merged: list[dict] = []
    for f in sorted(found, key=lambda x: x["s"]):
        if merged and f["s"] <= merged[-1]["e"]:
            if f["e"] > merged[-1]["e"]:
                merged[-1].update(e=f["e"], j=f["j"])
        else:
            merged.append(dict(f))
    return {"phrases": phrases, "spans": merged}


def _restarts_midphrase(phrases: list[dict], tokens: list[dict]) -> list[dict]:
    """Повтор с середины фразы без паузы перед ранним дублем: «…в одну геометрию и создают разного ро- |
    и создают разного рода дефекты». Конец фразы совпадает с началом следующей."""
    words = sorted((t for t in tokens if "s" in t and t.get("kind", "word") == "word"), key=lambda t: t["s"])
    out = []
    for j in range(1, len(phrases)):
        p, q = phrases[j - 1], phrases[j]
        if not p["w"] or not q["w"] or q["s"] - p["e"] > 3.0:
            continue
        for n in range(min(8, len(p["w"]) - 1), 2, -1):
            tail = p["w"][-n:]
            if not (_same(tail[0], q["w"][0]) and _lcs(tail, q["w"][:n + 2]) >= n - 1):
                continue
            # где в транскрипте начинается ранний дубль: слова фразы p с конца
            inp = [t for t in words if p["s"] - 0.1 <= (t["s"] + t["e"]) / 2 <= p["e"] + 0.1]
            for k in range(len(inp) - 2, -1, -1):
                if _same(_key(inp[k]["t"]), tail[0]) and _same(_key(inp[k + 1]["t"]), tail[1]):
                    out.append({"s": inp[k]["s"], "e": q["s"], "i": None, "j": j,
                                "early": " ".join(t["t"] for t in inp[k:])})
                    break
            break
    return out


def _fix_text(text: str) -> str:
    return re.sub(r"[#;]", "", text).strip()


def retake_ops(p, found: dict) -> tuple[list[str], list[dict]]:
    """Команды правки для найденных дублей и отчёт [{s, e, early, late}]."""
    toks = [t for t in p.tokens if "s" in t]
    cuts = p.cut_map()
    phrases = found["phrases"]
    ops, report = [], []
    for sp in found["spans"]:
        a, b, j = sp["s"], sp["e"], sp["j"]
        inside = [t for t in toks if a - 0.05 <= (t["s"] + t["e"]) / 2 < b and t["e"] <= b + 0.05]
        if not inside:
            continue  # дубль целиком спрятан в одно слово: вырезать его, не порвав слово, нельзя
        ids = {t["i"] for t in inside}
        first, last = inside[0], inside[-1]
        ops.append(f"cut {_ids_label(sorted(ids))} retake")
        # слово, на которое Whisper растянул начало раннего дубля, — подрезать
        before = [t for t in toks if t["i"] not in ids and (t["s"] + t["e"]) / 2 < a and t["e"] > a + 0.05]
        if before and before[-1]["i"] not in cuts and before[-1]["e"] <= last["e"]:
            ops.append(f"edge {first['i']} start {max(a - 0.05, before[-1]['s'] + 0.05):.2f}")
        # слово, в котором спрятано начало позднего дубля: вырезка заходит в него до начала дубля
        after = [t for t in toks if t["i"] not in ids and t["s"] < b - PRE and t["e"] > b + 0.05]
        if after:
            ops.append(f"edge {last['i']} end {max(b - PRE, last['e']):.2f}")
        # «заминки» в начале позднего дубля — его слова, которые Whisper не записал
        keep = [t["i"] for t in toks if t.get("kind") == "sound" and t["i"] in cuts and t["i"] not in ids
                and b - 0.2 <= t["s"] < b + 1.0]
        if keep:
            ops.append(f"keep {_ids_label(keep)}")
        # текст: слова позднего дубля, которые Whisper спрятал в растянутое слово
        words = [t for t in toks if t["i"] not in ids and t.get("kind", "word") == "word" and t["e"] > b + 0.05]
        if words:
            cand = words[0]
            raw = [w for x in phrases[j:] if x["s"] < cand["e"] for w in x["raw"]]
            k = next((n for n, w in enumerate(raw[:16]) if _same(_key(w), _key(cand["t"]))), 0)
            if k > 0:
                text = _fix_text(" ".join(raw[:k] + [cand["t"]]))
                prev = [t for t in toks if t.get("kind", "word") == "word" and t["e"] <= a + 0.05
                        and t["i"] not in cuts and t["i"] not in ids]
                if prev and not SENTENCE_END.search(prev[-1]["t"]):  # дубль начался посреди предложения
                    text = text[:1].lower() + text[1:]
                ops.append(f"fix {cand['i']} {text}")
        report.append({"s": a, "e": b, "early": sp["early"], "late": " ".join(phrases[j]["raw"])})
    return ops, report


def apply_retakes(p, by: str = "auto", progress: Progress = None, dry: bool = False) -> list[dict]:
    """Находит дубли в проекте и вырезает ранние (dry — только отчёт)."""
    from .ops import apply

    if p.kind != "audio":
        raise ValueError("Дубли ищутся только в аудиопроектах")
    found = find_retakes(p.source_path, p.tokens, p.data.get("language"), p.data.get("model"), progress)
    ops, report = retake_ops(p, found)
    if not dry:
        if ops:
            apply(p, ops, by=by)
        else:
            p.log(by, "дубли не найдены")
            p.save()
    return report


# --- создание проекта --------------------------------------------------------

def create_pack(files: list[str | Path], name: str | None = None, language: str | None = None,
                model: str | None = None, by: str = "cli", auto: bool = True, retakes: bool = True,
                progress: Progress = None, on_created: Callable[[str], None] | None = None):
    """Склеивает записи в одну дорожку (по порядку имён), распознаёт и вычёркивает ранние дубли."""
    from .project import create_audio

    files = sorted((Path(f) for f in files), key=lambda f: natural_key(f.name))
    if not files:
        raise ValueError("Пак пуст")
    say = progress or (lambda frac, msg: None)
    config.WORKSPACE.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=config.WORKSPACE, prefix="_pack-", ignore_cleanup_errors=True) as tmp:
        src = Path(tmp) / "pack.flac"
        parts = concat(files, src, lambda f, m: say(0.05 * f, m))
        stem = files[0].stem + (f" +{len(files) - 1}" if len(files) > 1 else "")
        p = create_audio(src, name=name or stem, language=language, model=model, by=by, auto=auto,
                         progress=lambda f, m: say(0.05 + 0.75 * f, m), on_created=on_created)
    p.data["parts"] = parts
    p.data["original_name"] = files[0].name if len(files) == 1 else \
        f"{files[0].name} … {files[-1].name} ({len(files)} файлов)"
    p.log(by, f"пак: склеено файлов: {len(files)}")
    p.save()
    if retakes:
        apply_retakes(p, progress=lambda f, m: say(0.8 + 0.2 * f, m))
    return p
