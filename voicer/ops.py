"""Мини-язык правок. Один и тот же для CLI, веб-интерфейса и MCP.

    cut 12-15,20 [причина]   вырезать токены
    keep 12-15 | keep all    вернуть токены (auto их больше не тронет)
    pause 42 0.3             пауза после токена 42 = 0.3 с (auto — снять)
    edge 12 start 6.85       начало вырезки токена 12 вручную (end — конец; edge 12 auto — как было)
    cuttime 1:02.5-1:04 [причина]   вырезать интервал времени
    uncuttime 1:02-1:05      снять вырезанные интервалы, пересекающие этот
    fix 17 текст             исправить распознанное слово (для текста/субтитров)
    set max_pause 0.4        настройка (см. voicer settings); set max_pause default — по умолчанию
    auto [fillers] [soft] [softall] [repeats] [sounds] [all]   автоправила (по умолч. fillers soft repeats)
    studio | studio off      «студийный звук»: шумоподавление, гул, де-эссер, левелер, компрессор
    reset                    убрать все правки
"""
from __future__ import annotations

import re
from collections import Counter

from . import config
from .audio import FORMATS
from .project import Project, empty_edit
from .textproc import find_repeats, is_strict_filler, soft_fillers

# fillers — «э-э», «мм» и заминки; soft — «ну», «вот», «короче»… стоящие отдельно;
# softall — те же слова везде, даже внутри фразы; repeats — повторы; sounds — вдохи и прочие звуки.
AUTO_DEFAULT = ("fillers", "soft", "repeats")
AUTO_ALL = ("fillers", "soft", "softall", "repeats", "sounds")
MAX_SOUND = 2.0


def parse_ids(spec: str, n: int) -> list[int]:
    out: list[int] = []
    for part in re.split(r"[,\s]+", spec.strip()):
        if not part:
            continue
        m = re.fullmatch(r"(\d+)(?:(?:-|\.\.)(\d+))?", part)
        if not m:
            raise ValueError(f"не понимаю номера «{part}» (пример: 12 или 12-15 или 3,7,9-11)")
        a = int(m.group(1))
        b = int(m.group(2)) if m.group(2) else a
        if a > b:
            a, b = b, a
        if b >= n:
            raise ValueError(f"номер {b} вне диапазона 0..{n - 1}")
        out.extend(range(a, b + 1))
    return out


def parse_time(s: str) -> float:
    s = s.strip()
    parts = s.split(":")
    try:
        v = 0.0
        for p in parts:
            v = v * 60 + float(p)
        return v
    except ValueError:
        raise ValueError(f"не понимаю время «{s}» (пример: 75.5 или 1:15.5)") from None


def parse_span(s: str) -> tuple[float, float]:
    m = re.fullmatch(r"\s*([\d:.]+)\s*-\s*([\d:.]+)\s*", s)
    if not m:
        raise ValueError(f"не понимаю интервал «{s}» (пример: 1:02.5-1:04)")
    a, b = parse_time(m.group(1)), parse_time(m.group(2))
    return (a, b) if a <= b else (b, a)


def _parse_value(key: str, raw: str):
    if key not in config.DEFAULT_SETTINGS:
        raise ValueError(f"нет настройки «{key}». Есть: {', '.join(config.DEFAULT_SETTINGS)}")
    if key == "format":
        if raw not in FORMATS:
            raise ValueError(f"формат: {', '.join(FORMATS)}")
        return raw
    off = raw.lower() in ("off", "none", "нет", "выкл")
    if key == "voice_clean":
        return raw.lower() in ("on", "1", "true", "да", "вкл")
    if key in ("lufs", "limit") and off:
        return None
    if key in config.STUDIO_KEYS and off:
        return 0
    if key == "dehum" and raw.lower() in ("auto", "авто"):
        return "auto"
    try:
        return float(raw)
    except ValueError:
        raise ValueError(f"{key}: нужно число, получено «{raw}»") from None


def _ids_label(ids: list[int]) -> str:
    if not ids:
        return "—"
    ids = sorted(set(ids))
    parts, start, prev = [], ids[0], ids[0]
    for x in ids[1:] + [None]:
        if x is not None and x == prev + 1:
            prev = x
            continue
        parts.append(str(start) if start == prev else f"{start}-{prev}")
        if x is not None:
            start = prev = x
    return ",".join(parts)


_BOUNDARY = re.compile(r"[,.!?;:…—]+[\"'»”)]*$")
ISOLATION_GAP = 0.25


def isolated_soft(p: Project, strict: set[int]) -> list[int]:
    """«Ну», «вот», «короче», «как бы»… стоящие особняком — между запятыми/паузами/краями фразы.

    «Это, ну, очень важная тема» → «ну» паразит; «Ну и что?» и «это значит, что» — нет.
    """
    toks = p.tokens
    words = [t for t in toks if t.get("kind", "word") == "word" and t["i"] not in strict]
    texts = [p.text_of(t["i"]) for t in words]
    soft = sorted(soft_fillers(texts, p.data.get("language")))

    def noise_between(i0: int, i1: int) -> bool:
        """Между токенами i0 и i1 (не включая) есть заминка, паразит или звук."""
        return any(toks[j].get("kind") == "sound" or j in strict for j in range(i0 + 1, i1))

    def gap(t0: dict, t1: dict) -> float:
        return (t1["s"] - t0["e"]) if "s" in t0 and "s" in t1 else 0.0

    out: list[int] = []
    k = 0
    while k < len(soft):
        a = b = soft[k]
        while k + 1 < len(soft) and soft[k + 1] == b + 1 and not _BOUNDARY.search(texts[b]):
            k += 1
            b = soft[k]
        k += 1
        prev = words[a - 1] if a > 0 else None
        nxt = words[b + 1] if b + 1 < len(words) else None
        before = (prev is None or bool(_BOUNDARY.search(texts[a - 1])) or gap(prev, words[a]) >= ISOLATION_GAP
                  or noise_between(prev["i"], words[a]["i"]))
        after = (nxt is None or bool(_BOUNDARY.search(texts[b])) or gap(words[b], nxt) >= ISOLATION_GAP
                 or noise_between(words[b]["i"], nxt["i"]))
        if before and after:
            out.extend(words[j]["i"] for j in range(a, b + 1))
    return out


def run_auto(p: Project, opts: tuple[str, ...] = AUTO_DEFAULT) -> Counter:
    cuts, keeps = p.edit["cuts"], set(p.edit["keeps"])
    added: Counter = Counter()
    lang = p.data.get("language")

    def mark(i: int, why: str) -> None:
        if i in keeps or str(i) in cuts:
            return
        cuts[str(i)] = why
        added[why] += 1

    words = [t for t in p.tokens if t.get("kind", "word") == "word"]
    strict = {t["i"] for t in words if is_strict_filler(p.text_of(t["i"]), lang)}
    if "fillers" in opts:
        for i in sorted(strict):
            mark(i, "filler")
        # Заминки «э-э», «а-а», которые Whisper не записал, а анализатор нашёл по звуку.
        for t in p.tokens:
            if t.get("hes"):
                mark(t["i"], "filler")
    if "soft" in opts:
        for i in isolated_soft(p, strict):
            mark(i, "soft")
    if "softall" in opts:
        seq = [t for t in words if t["i"] not in strict]
        for k in soft_fillers([p.text_of(t["i"]) for t in seq], lang):
            mark(seq[k]["i"], "soft")
    if "repeats" in opts:
        seq = [t for t in words if t["i"] not in strict]
        for a, b, why in find_repeats([p.text_of(t["i"]) for t in seq]):
            for k in range(a, b):
                mark(seq[k]["i"], why)
    if "sounds" in opts:
        for t in p.tokens:
            if t.get("kind") == "sound" and t["e"] - t["s"] <= MAX_SOUND:
                mark(t["i"], "sound")
    return added


def _execute(p: Project, line: str) -> str:
    parts = line.split(None, 2)
    cmd = parts[0].lower()
    args = parts[1:]
    n = len(p.tokens)
    ed = p.edit

    if cmd in ("cut", "del", "delete"):
        if not args:
            raise ValueError("cut <номера> [причина]")
        ids = parse_ids(args[0], n)
        why = args[1].strip() if len(args) > 1 else "manual"
        for i in ids:
            ed["cuts"][str(i)] = why
            if i in ed["keeps"]:
                ed["keeps"].remove(i)
        return f"вырезано {_ids_label(ids)} ({why})"

    if cmd in ("keep", "restore", "uncut"):
        if not args:
            raise ValueError("keep <номера> | keep all")
        if args[0] == "all":
            cnt = len(ed["cuts"])
            ed["cuts"].clear()
            return f"возвращено всё ({cnt})"
        ids = parse_ids(" ".join(args), n)
        for i in ids:
            ed["cuts"].pop(str(i), None)
            if i not in ed["keeps"]:
                ed["keeps"].append(i)
        return f"возвращено {_ids_label(ids)}"

    if cmd == "pause":
        if len(args) < 2:
            raise ValueError("pause <номер> <сек|auto>")
        ids = parse_ids(args[0], n)
        val = args[1].strip()
        for i in ids:
            if val in ("auto", "-", "reset"):
                ed["pauses"].pop(str(i), None)
            else:
                ed["pauses"][str(i)] = max(0.0, parse_time(val))
        return f"пауза после {_ids_label(ids)}: {val}"

    if cmd == "edge":
        w = line.split()
        if len(w) < 3:
            raise ValueError("edge <номер> start|end <сек|auto> | edge <номер> auto")
        i = parse_ids(w[1], n)[0]
        t = p.tokens[i]
        if "s" not in t:
            raise ValueError("у текстового проекта нет времени")
        if w[2] == "auto":
            ed["bounds"].pop(str(i), None)
            return f"границы {i}: автоматически"
        side = {"start": "s", "end": "e", "начало": "s", "конец": "e"}.get(w[2])
        if side is None or len(w) < 4:
            raise ValueError("edge <номер> start|end <сек|auto>")
        b = ed["bounds"].setdefault(str(i), {})
        if w[3] == "auto":
            b.pop(side, None)
        else:
            v = min(max(0.0, parse_time(w[3])), p.duration or 1e9)
            s0, e0 = b.get("s", t["s"]), b.get("e", t["e"])
            if (side == "s" and v > e0 - 0.02) or (side == "e" and v < s0 + 0.02):
                raise ValueError("начало должно быть раньше конца")
            b[side] = round(v, 3)
        if not b:
            ed["bounds"].pop(str(i), None)
        return f"{i}: {'начало' if side == 's' else 'конец'} вырезки → {w[3]}"

    if cmd == "cuttime":
        if not args:
            raise ValueError("cuttime <от>-<до> [причина]")
        a, b = parse_span(args[0])
        why = args[1].strip() if len(args) > 1 else "manual"
        ed["ranges"].append([round(a, 3), round(b, 3), why])
        ed["ranges"].sort()
        return f"вырезан интервал {a:.2f}-{b:.2f} ({why})"

    if cmd == "uncuttime":
        a, b = parse_span(args[0] if args else "")
        before = len(ed["ranges"])
        ed["ranges"] = [r for r in ed["ranges"] if r[1] <= a or r[0] >= b]
        return f"снято интервалов: {before - len(ed['ranges'])}"

    if cmd == "fix":
        if len(args) < 2:
            raise ValueError("fix <номер> <текст>")
        i = parse_ids(args[0], n)[0]
        ed["fixes"][str(i)] = args[1].strip()
        return f"{i}: «{p.tokens[i]['t']}» → «{args[1].strip()}»"

    if cmd == "set":
        if len(args) < 2:
            raise ValueError("set <настройка> <значение>")
        key = args[0]
        if args[1].strip().lower() in ("default", "по-умолчанию", "умолч"):
            if key not in config.DEFAULT_SETTINGS:
                raise ValueError(f"нет настройки «{key}»")
            p.settings.reset(key)
            return f"{key} = {p.settings[key]} (по умолчанию)"
        p.settings[key] = _parse_value(key, args[1].strip())
        return f"{key} = {p.settings[key]}"

    if cmd == "studio":
        if args and args[0].lower() in ("off", "выкл"):
            p.settings["voice_clean"] = False
            return "очистка голоса выключена"
        p.settings["voice_clean"] = True
        p.settings.update(config.STUDIO_PRESET)
        return "студийный пресет: " + " ".join(f"{k}={v}" for k, v in config.STUDIO_PRESET.items())

    if cmd == "auto":
        words =" ".join(args).lower().split()
        opts = AUTO_ALL if "all" in words else tuple(w for w in words if w in AUTO_ALL) or AUTO_DEFAULT
        bad = [w for w in words if w not in AUTO_ALL and w != "all"]
        if bad:
            raise ValueError(f"auto: неизвестно {bad}; есть {', '.join(AUTO_ALL)}, all")
        added = run_auto(p, opts)
        total = sum(added.values())
        detail = ", ".join(f"{k}: {v}" for k, v in added.items()) or "ничего нового"
        return f"auto ({' '.join(opts)}): вырезано {total} — {detail}"

    if cmd == "reset":
        fixes = ed["fixes"]
        p.data["edit"] = empty_edit()
        p.data["edit"]["fixes"] = fixes
        return "все правки сброшены"

    raise ValueError(f"неизвестная команда «{cmd}». Есть: cut keep pause edge cuttime uncuttime fix set auto studio reset")


def split_commands(text: str) -> list[str]:
    out = []
    for raw in text.replace(";", "\n").splitlines():
        line = raw.split("#", 1)[0].strip()
        if line:
            out.append(line)
    return out


def apply(p: Project, commands: list[str], by: str = "cli") -> list[str]:
    """Применяет команды атомарно: при ошибке ничего не сохраняется."""
    fresh = p.reload()
    p.data, p._settings = fresh.data, fresh.settings
    lines = [c for cmd in commands for c in split_commands(cmd)]
    if not lines:
        return []
    backup = p.state()
    report = []
    try:
        for line in lines:
            report.append(_execute(p, line))
    except ValueError as e:
        p.restore(backup)
        raise ValueError(f"{line}: {e}") from None
    p.snapshot(backup)
    p.log(by, "; ".join(report))
    p.save()
    return report
