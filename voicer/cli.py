"""Командная строка: voicer <команда> ... (проект по умолчанию — последний изменённый)."""
from __future__ import annotations

import argparse
import json
import locale
import sys
from pathlib import Path

from . import config


def _out(s: str) -> None:
    sys.stdout.write(s if s.endswith("\n") else s + "\n")


def _decode(raw: bytes) -> str:
    """Текст из stdin или файла: UTF-8 (с BOM и без), UTF-16 (Out-File в PowerShell 5) или ANSI Windows."""
    if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
        return raw.decode("utf-16")
    try:
        return raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        return raw.decode(locale.getpreferredencoding(False), errors="replace")


def _read_input(src: str) -> str:
    """Содержимое файла или stdin (`-`). Кодировку stdin Windows берёт из локали (cp1251), а пишут в него UTF-8."""
    return _decode(sys.stdin.buffer.read() if src == "-" else Path(src).read_bytes())


def _progress(frac: float, msg: str) -> None:
    sys.stderr.write(f"\r[{frac * 100:5.1f}%] {msg:<50}")
    sys.stderr.flush()
    if frac >= 0.97:
        sys.stderr.write("\n")


def _summary(p) -> str:
    from .render import plan
    from .view import fmt_t

    if p.kind == "audio":
        return f"→ {p.id}: {fmt_t(p.duration)} → {fmt_t(plan(p)['duration'])}, вырезано токенов: {len(p.cut_map())}"
    return f"→ {p.id}: вырезано токенов: {len(p.cut_map())} из {len(p.tokens)}"


def cmd_import(a) -> None:
    from .pack import create_pack
    from .project import create_audio, create_text

    srcs = [Path(f) for f in a.file]
    for s in srcs:
        if not s.exists():
            raise SystemExit(f"Нет файла: {s}")
    script = _read_input(a.script) if a.script else None
    if len(srcs) > 1 or srcs[0].is_dir():  # пак: склеить в одну дорожку и вычеркнуть ранние дубли
        files = [f for s in srcs for f in (sorted(s.iterdir()) if s.is_dir() else [s])
                 if f.suffix.lower() in config.AUDIO_EXTS]
        if not files:
            raise SystemExit("Нет аудиофайлов")
        name = a.name or (srcs[0].name if srcs[0].is_dir() else None)
        p = create_pack(files, name=name, language=a.lang, model=a.model, auto=not a.no_auto,
                        retakes=not a.no_retakes, progress=_progress)
        if script:
            p.data["script"] = script
            p.save()
        _out(f"Создан проект {p.id} из {len(files)} файлов")
        for x in p.data.get("log", []):
            _out(f"  [{x['by']}] {x['msg']}")
        _out(_summary(p))
        return
    src = srcs[0]
    if src.suffix.lower() in config.TEXT_EXTS:
        p = create_text(_read_input(str(src)), a.name or src.stem, auto=not a.no_auto)
    else:
        p = create_audio(src, name=a.name, language=a.lang, model=a.model, script=script,
                         auto=not a.no_auto, filler_prompt=not a.no_filler_prompt, progress=_progress)
    _out(f"Создан проект {p.id}")
    for x in p.data.get("log", []):
        _out(f"  [{x['by']}] {x['msg']}")
    _out(_summary(p))


def cmd_text(a) -> None:
    from .project import create_text

    text = _read_input("-") if a.text == "-" else a.text
    p = create_text(text, a.name or "text", auto=not a.no_auto)
    _out(f"Создан проект {p.id}")
    _out(_summary(p))


def cmd_list(a) -> None:
    from .project import list_projects
    from .view import fmt_t

    ps = list_projects()
    if not ps:
        _out("Проектов нет.")
    for p in ps:
        dur = fmt_t(p.duration) if p.kind == "audio" else f"{len(p.tokens)} слов"
        _out(f"{p.id:<32} {p.kind:<5} {dur:>10}  {p.data.get('status', ''):<8} {p.data.get('updated', '')}  "
             f"{p.data.get('name', '')}")


def _span(spec: str | None):
    if not spec:
        return None, None
    a, _, b = spec.partition("-")
    return (int(a) if a else None), (int(b) if b else None)


def cmd_show(a) -> None:
    from .ops import parse_span
    from .project import find
    from .view import show

    p = find(a.project)
    if a.script:
        _out(p.data.get("script") or "(сценария нет)")
        return
    if a.text:
        from .render import export_text

        _out(export_text(p, original=a.orig))
        return
    s, e = _span(a.range)
    tf = tt = None
    if a.time:
        tf, tt = parse_span(a.time)
    _out(show(p, s, e, tf, tt, pause_min=a.pause, with_header=not a.no_header))


def _gather_commands(a) -> list[str]:
    cmds: list[str] = []
    if a.file:
        cmds.append(_read_input(a.file))
    if a.commands:
        if any(" " in c for c in a.commands):
            cmds.extend(a.commands)
        else:
            cmds.append(" ".join(a.commands))
    return cmds


def cmd_edit(a) -> None:
    from .ops import apply
    from .project import find

    p = find(a.project)
    cmds = _gather_commands(a)
    if not cmds:
        raise SystemExit("Нет команд. Пример: voicer edit \"cut 12-15 repeat\" \"pause 42 0.3\"")
    for line in apply(p, cmds, by=a.by):
        _out("  " + line)
    _out(_summary(p))


def cmd_auto(a) -> None:
    from .ops import apply
    from .project import find

    p = find(a.project)
    for line in apply(p, ["auto " + " ".join(a.opts)], by=a.by):
        _out("  " + line)
    _out(_summary(p))


def cmd_undo(a) -> None:
    from .project import find

    p = find(a.project)
    if not p.undo():
        raise SystemExit("Отменять нечего.")
    p.log(a.by, "отмена последнего изменения")
    p.save()
    _out("Отменено.")
    _out(_summary(p))


def cmd_retakes(a) -> None:
    from .pack import apply_retakes
    from .project import find
    from .view import fmt_t

    p = find(a.project)
    report = apply_retakes(p, by=a.by, progress=_progress, dry=a.dry)
    sys.stderr.write("\n")
    for r in report:
        _out(f"{fmt_t(r['s'])}–{fmt_t(r['e'])}  ранний: {r['early']}")
        _out(f"{'':16} поздний: {r['late']}")
    _out(f"дублей: {len(report)}" + (" (ничего не вырезано: --dry)" if a.dry else ""))
    _out(_summary(p))


def cmd_reanalyze(a) -> None:
    from .project import find, reanalyze

    p = find(a.project)
    _out(reanalyze(p, by=a.by))
    _out(_summary(p))


def cmd_render(a) -> None:
    from .project import find
    from .render import record_render, render
    from .view import fmt_t

    p = find(a.project)
    r = render(p, a.format, a.output, progress=_progress)
    record_render(p, r, a.by)
    _out(f"Готово: {r['path']}")
    _out(f"Длительность: {fmt_t(r['original'])} → {fmt_t(r['duration'])}")


def cmd_export(a) -> None:
    from .project import find
    from .render import export_subs, export_text

    p = find(a.project)
    if a.kind == "txt":
        data, ext = export_text(p, original=a.orig), "txt"
    elif a.kind in ("srt", "vtt"):
        data, ext = export_subs(p, original=a.orig, vtt=a.kind == "vtt"), a.kind
    else:
        data, ext = json.dumps(p.data, ensure_ascii=False, indent=1), "json"
    if a.output == "-":
        _out(data)
        return
    out = Path(a.output) if a.output else p.dir / "out" / f"{p.id}{'-orig' if a.orig else ''}.{ext}"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(data, encoding="utf-8")
    _out(f"Готово: {out}")


def cmd_stats(a) -> None:
    from .project import find
    from .view import stats

    _out(stats(find(a.project)))


def cmd_settings(a) -> None:
    from .project import find

    p = find(a.project)
    for k, v in p.settings.items():
        mark = "*" if k in p.settings.overrides else " "
        _out(f"{mark}{k:<20} {str(v):<8} {config.SETTING_HELP.get(k, '')}")
    _out("* — задано в проекте; остальное — по умолчанию (set <ключ> default — вернуть)")


def cmd_notes(a) -> None:
    from .project import find

    p = find(a.project)
    if a.text is None:
        _out(p.data.get("notes") or "(заметок нет)")
        return
    p.data["notes"] = a.text
    p.log(a.by, "заметки обновлены")
    p.save()
    _out("Сохранено.")


def cmd_delete(a) -> None:
    from .project import delete, find

    p = find(a.project)
    if not a.yes:
        raise SystemExit(f"Удалить {p.id} вместе с файлами? Повторите с --yes")
    delete(p)
    _out(f"Удалён {p.id}")


def cmd_serve(a) -> None:
    from .web.server import run

    run(host=a.host, port=a.port, open_browser=not a.no_browser)


def cmd_mcp(a) -> None:
    from .mcp_server import main as mcp_main

    mcp_main()


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="voicer", description="Автомонтаж речи: транскрибация, паузы, "
                                 "слова-паразиты, повторы. Проект по умолчанию — последний изменённый.")
    ap.add_argument("--home", help=f"папка проектов (по умолчанию {config.WORKSPACE})")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def proj(sp):
        sp.add_argument("-p", "--project", help="id проекта (по умолчанию последний)")
        sp.add_argument("--by", default="cli", help=argparse.SUPPRESS)
        return sp

    sp = sub.add_parser("import", help="создать проект из аудио (mp3/ogg/wav/...) или .txt; несколько "
                        "файлов или папка — пак: одна дорожка, ранние дубли вычеркнуты")
    sp.add_argument("file", nargs="+")
    sp.add_argument("--no-retakes", action="store_true", help="пак: не искать дубли")
    sp.add_argument("--name")
    sp.add_argument("--lang", help="ru, en, ... (по умолчанию — автоопределение)")
    sp.add_argument("--model", help=f"модель Whisper (GPU: {config.GPU_MODEL}, CPU: {config.CPU_MODEL})")
    sp.add_argument("--script", help="эталонный текст (сценарий) для сверки")
    sp.add_argument("--no-auto", action="store_true", help="не применять автоправила")
    sp.add_argument("--no-filler-prompt", action="store_true", help="не подсказывать Whisper писать «э-э»")
    sp.set_defaults(fn=cmd_import)

    sp = sub.add_parser("text", help="создать текстовый проект из строки (или - для stdin)")
    sp.add_argument("text")
    sp.add_argument("--name")
    sp.add_argument("--no-auto", action="store_true")
    sp.set_defaults(fn=cmd_text)

    sub.add_parser("list", help="список проектов").set_defaults(fn=cmd_list)

    sp = proj(sub.add_parser("show", help="транскрипт с номерами токенов, паузами и правками"))
    sp.add_argument("--range", help="токены, напр. 100-250")
    sp.add_argument("--time", help="время, напр. 1:00-2:30")
    sp.add_argument("--pause", type=float, default=0.3, help="показывать паузы от N сек")
    sp.add_argument("--text", action="store_true", help="только итоговый текст")
    sp.add_argument("--orig", action="store_true", help="с --text: исходный текст")
    sp.add_argument("--script", action="store_true", help="показать эталонный текст")
    sp.add_argument("--no-header", action="store_true")
    sp.set_defaults(fn=cmd_show)

    from . import ops

    sp = proj(sub.add_parser("edit", help="применить команды правок (см. voicer edit --help)",
                             description=ops.__doc__,
                             formatter_class=argparse.RawDescriptionHelpFormatter))
    sp.add_argument("commands", nargs="*")
    sp.add_argument("-f", "--file", help="файл с командами (по строке) или - для stdin")
    sp.set_defaults(fn=cmd_edit)

    sp = proj(sub.add_parser("auto", help="автоправила: fillers soft repeats sounds | all"))
    sp.add_argument("opts", nargs="*")
    sp.set_defaults(fn=cmd_auto)

    sp = proj(sub.add_parser("retakes", help="найти дубли (каждая фраза распознаётся отдельно) и вырезать "
                             "ранние, оставив поздний"))
    sp.add_argument("--dry", action="store_true", help="только показать, ничего не вырезать")
    sp.set_defaults(fn=cmd_retakes)

    proj(sub.add_parser("undo", help="отменить последнее изменение")).set_defaults(fn=cmd_undo)
    proj(sub.add_parser("reanalyze", help="заново найти паузы и звуки по сигналу (без распознавания)")
         ).set_defaults(fn=cmd_reanalyze)

    sp = proj(sub.add_parser("render", help="собрать итоговое аудио"))
    sp.add_argument("--format", choices=["mp3", "ogg", "opus", "wav", "flac", "m4a"])
    sp.add_argument("-o", "--output")
    sp.set_defaults(fn=cmd_render)

    sp = proj(sub.add_parser("export", help="txt / srt / vtt / json"))
    sp.add_argument("kind", choices=["txt", "srt", "vtt", "json"])
    sp.add_argument("--orig", action="store_true", help="по исходнику, без правок")
    sp.add_argument("-o", "--output", help="путь или - для вывода в консоль")
    sp.set_defaults(fn=cmd_export)

    proj(sub.add_parser("stats", help="сводка: паузы, правки, история")).set_defaults(fn=cmd_stats)
    proj(sub.add_parser("settings", help="настройки монтажа проекта")).set_defaults(fn=cmd_settings)

    sp = proj(sub.add_parser("notes", help="заметки пользователя для LLM (прочитать/записать)"))
    sp.add_argument("text", nargs="?")
    sp.set_defaults(fn=cmd_notes)

    sp = proj(sub.add_parser("delete", help="удалить проект"))
    sp.add_argument("--yes", action="store_true")
    sp.set_defaults(fn=cmd_delete)

    sp = sub.add_parser("serve", help="веб-интерфейс")
    sp.add_argument("--host", default="127.0.0.1")
    sp.add_argument("--port", type=int, default=8765)
    sp.add_argument("--no-browser", action="store_true")
    sp.set_defaults(fn=cmd_serve)

    sub.add_parser("mcp", help="MCP-сервер (stdio) для Claude Desktop и др.").set_defaults(fn=cmd_mcp)
    return ap


def main(argv: list[str] | None = None) -> None:
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    a = build_parser().parse_args(argv)
    if a.home:
        config.WORKSPACE = Path(a.home).resolve()
    try:
        a.fn(a)
    except (LookupError, ValueError) as e:
        raise SystemExit(f"Ошибка: {e}") from None
