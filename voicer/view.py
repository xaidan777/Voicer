"""Компактное текстовое представление проекта — то, что читает LLM."""
from __future__ import annotations

from collections import defaultdict

from .ops import _ids_label
from .project import Project
from .render import plan
from .textproc import SENTENCE_END

LOW_PROB = 0.5


def fmt_t(t: float) -> str:
    m, s = divmod(max(t, 0.0), 60)
    return f"{int(m):02d}:{s:04.1f}"


def header(p: Project) -> list[str]:
    toks = p.tokens
    nsound = sum(1 for t in toks if t.get("kind") == "sound")
    cuts = p.cut_map()
    st = p.settings
    lines = []
    if p.kind == "audio":
        pl = plan(p)
        lines.append(f"# {p.id} «{p.data.get('name')}» — аудио {fmt_t(p.duration)} → {fmt_t(pl['duration'])} "
                     f"после правок | язык {p.data.get('language')} | {len(toks) - nsound} слов, "
                     f"{nsound} звуков | вырезано токенов: {len(cuts)}")
        lines.append("# настройки: " + " ".join(f"{k}={v}" for k, v in st.items()))
        lines.append("# N:слово — токен №N | ~~N:слово~~ — вырезан | N:~0.4s — нераспознанный звук "
                     "(вдох, шум) | N:~заминка0.4s — тянущийся «э-э»/«а-а», который Whisper спрятал в слово | "
                     "‖1.2‖ — пауза после токена, сек | ‖1.2→0.3‖ — пауза задана | "
                     "слово? — низкая уверенность распознавания")
    else:
        lines.append(f"# {p.id} «{p.data.get('name')}» — текст | {len(toks)} слов | вырезано: {len(cuts)}")
        lines.append("# N:слово — токен №N | ~~N:слово~~ — вырезан")
    parts = p.data.get("parts") or []
    if len(parts) > 1:
        lines.append(f"# пак из {len(parts)} файлов, начала файлов: "
                     + ", ".join(fmt_t(x["s"]) for x in parts[1:]))
    if p.edit["ranges"]:
        lines.append("# вырезанные интервалы: " + "; ".join(
            f"{fmt_t(s)}-{fmt_t(e)} ({w})" for s, e, w in p.edit["ranges"]))
    if p.data.get("notes"):
        lines.append("# ЗАМЕТКИ ПОЛЬЗОВАТЕЛЯ: " + p.data["notes"].replace("\n", " / "))
    if p.data.get("script"):
        lines.append(f"# есть эталонный текст (сценарий), {len(p.data['script'])} символов: voicer show --script")
    return lines


def token_str(p: Project, t: dict, cuts: dict) -> str:
    i = t["i"]
    if t.get("kind") == "sound":
        body = f"{i}:~{'заминка' if t.get('hes') else ''}{t['e'] - t['s']:.1f}s"
    else:
        body = f"{i}:{p.text_of(i)}"
        if t.get("p") is not None and t["p"] < LOW_PROB:
            body += "?"
    return f"~~{body}~~" if i in cuts else body


def show(p: Project, start: int | None = None, end: int | None = None,
         tfrom: float | None = None, tto: float | None = None,
         pause_min: float = 0.3, with_header: bool = True, max_line: int = 28) -> str:
    toks = p.tokens
    cuts = p.cut_map()
    pauses = {int(k): v for k, v in p.edit["pauses"].items()}
    sel = toks[start if start is not None else 0:(end + 1) if end is not None else None]
    if tfrom is not None or tto is not None:
        lo, hi = tfrom or 0.0, tto if tto is not None else float("inf")
        sel = [t for t in sel if "s" in t and t["e"] >= lo and t["s"] <= hi]
    out = header(p) if with_header else []
    line: list[str] = []

    def flush() -> None:
        if line:
            out.append(" ".join(line))
            line.clear()

    for t in sel:
        if not line and "s" in t:
            line.append(f"[{fmt_t(t['s'])}]")
        line.append(token_str(p, t, cuts))
        nxt = toks[t["i"] + 1] if t["i"] + 1 < len(toks) else None
        gap = (nxt["s"] - t["e"]) if nxt is not None and "s" in t else 0.0
        if t["i"] in pauses:
            line.append(f"‖{gap:.1f}→{pauses[t['i']]:.2f}‖")
        elif gap >= pause_min:
            line.append(f"‖{gap:.1f}‖")
        words_in_line = len(line)
        if (t.get("kind") != "sound" and SENTENCE_END.search(p.text_of(t["i"]))) or gap >= 1.0 \
                or words_in_line >= max_line:
            flush()
    flush()

    if with_header and cuts:
        groups = defaultdict(list)
        for i, why in cuts.items():
            groups[why].append(i)
        out.append("# вырезано по причинам: " + "; ".join(f"{w}: {_ids_label(ids)}" for w, ids in groups.items()))
    return "\n".join(out)


def stats(p: Project) -> str:
    lines = header(p)
    if p.kind == "audio":
        toks = [t for t in p.tokens if "s" in t]
        gaps = [b["s"] - a["e"] for a, b in zip(toks, toks[1:])]
        buckets = [(0.3, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, 1e9)]
        lines.append("# паузы: " + ", ".join(
            f"{a}-{b if b < 1e9 else '∞'}с: {sum(1 for g in gaps if a <= g < b)}" for a, b in buckets))
        lines.append(f"# суммарно пауз ≥0.3с: {sum(g for g in gaps if g >= 0.3):.1f} с")
        low = [t["i"] for t in toks if t.get("p") is not None and t["p"] < LOW_PROB]
        if low:
            lines.append(f"# низкая уверенность: {_ids_label(low)}")
    log = p.data.get("log", [])[-8:]
    if log:
        lines.append("# последние изменения:")
        lines += [f"#   {x['ts']} [{x['by']}] {x['msg']}" for x in log]
    return "\n".join(lines)
