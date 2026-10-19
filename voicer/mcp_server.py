"""MCP-сервер (stdio): те же операции, что и CLI, — для Claude Desktop и других MCP-клиентов."""
from __future__ import annotations

import functools

try:  # mcp 2.x
    from mcp.server.mcpserver import MCPServer as FastMCP
except ImportError:  # mcp 1.x
    from mcp.server.fastmcp import FastMCP

from . import ops as O
from .project import create_audio, create_text, find, list_projects as _list
from .render import export_subs, export_text, record_render, render
from .view import fmt_t, show as _show, stats as _stats

mcp = FastMCP("voicer", instructions=(
    "Voicer — автомонтаж речи. Порядок работы: import_file → show (транскрипт с номерами токенов, "
    "паузами ‖сек‖ и вырезанным ~~N:слово~~) → edit (команды правок) → render_audio. "
    "Команды edit: " + (O.__doc__ or "")))


def _safe(fn):
    """Ошибки ввода возвращаются LLM текстом, а не безликим «Error executing tool»."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except (ValueError, LookupError, FileNotFoundError) as e:
            return f"Ошибка: {e}"
    return wrapper


def _summary(p) -> str:
    from .render import plan

    if p.kind == "audio":
        return f"{p.id}: {fmt_t(p.duration)} → {fmt_t(plan(p)['duration'])}, вырезано токенов: {len(p.cut_map())}"
    return f"{p.id}: вырезано {len(p.cut_map())} из {len(p.tokens)}"


@mcp.tool()
@_safe
def list_projects() -> str:
    """Список проектов (последний изменённый — первый)."""
    rows = [f"{p.id} | {p.kind} | {fmt_t(p.duration) if p.kind == 'audio' else str(len(p.tokens)) + ' слов'} | "
            f"{p.data.get('status')} | {p.data.get('name')}" for p in _list()]
    return "\n".join(rows) or "Проектов нет."


@mcp.tool()
@_safe
def import_file(path: str, language: str = "", name: str = "", script_path: str = "") -> str:
    """Создать проект из аудио (mp3/ogg/wav/flac/m4a) или .txt по локальному пути. Распознавание с
    пословными таймкодами, затем автоправила (паразиты, повторы). language: ru/en/... или пусто — авто.
    Папка — пак: её аудиофайлы склеиваются по порядку имён в одну дорожку, ранние дубли вычёркиваются."""
    from pathlib import Path

    from .config import AUDIO_EXTS
    from .pack import create_pack

    src = Path(path)
    if src.is_dir():
        files = [f for f in sorted(src.iterdir()) if f.suffix.lower() in AUDIO_EXTS]
        if not files:
            return "В папке нет аудиофайлов."
        p = create_pack(files, name=name or src.name, language=language or None, by="mcp")
        return f"Создан пак из {len(files)} файлов: " + _summary(p)
    if src.suffix.lower() in (".txt", ".md"):
        p = create_text(src.read_text(encoding="utf-8"), name or src.stem, by="mcp")
    else:
        script = Path(script_path).read_text(encoding="utf-8") if script_path else None
        p = create_audio(src, name=name or None, language=language or None, script=script, by="mcp")
    return "Создан " + _summary(p)


@mcp.tool()
@_safe
def find_retakes(project: str = "last", dry: bool = False) -> str:
    """Найти дубли (фальстарты и перезаписанные фразы) и вырезать ранние, оставив поздний. Каждая фраза
    распознаётся отдельно, поэтому находятся и повторы, которые Whisper спрятал в растянутое слово.
    Около минуты на 10 минут записи (GPU). dry — только показать."""
    from .pack import apply_retakes

    p = find(project)
    report = apply_retakes(p, by="mcp", dry=dry)
    lines = [f"{fmt_t(r['s'])}–{fmt_t(r['e'])} ранний: {r['early']} | поздний: {r['late']}" for r in report]
    return "\n".join(lines + [f"дублей: {len(report)}", _summary(p)])


@mcp.tool()
@_safe
def import_text(text: str, name: str = "text") -> str:
    """Создать текстовый проект (чистка паразитов и повторов в тексте)."""
    return "Создан " + _summary(create_text(text, name, by="mcp"))


@mcp.tool()
@_safe
def show(project: str = "last", token_range: str = "", time_range: str = "", text_only: bool = False) -> str:
    """Транскрипт для правки: N:слово, ~~вырезано~~, ‖пауза‖, N:~0.4s — нераспознанный звук.
    token_range «100-250», time_range «1:00-2:30» — для длинных записей. text_only — итоговый текст."""
    p = find(project)
    if text_only:
        return export_text(p)
    s = e = None
    if token_range:
        a, _, b = token_range.partition("-")
        s, e = (int(a) if a else None), (int(b) if b else None)
    tf = tt = None
    if time_range:
        tf, tt = O.parse_span(time_range)
    return _show(p, s, e, tf, tt)


@mcp.tool()
@_safe
def edit(project: str, commands: list[str]) -> str:
    """Применить команды правок атомарно (см. описание сервера): cut, keep, pause, cuttime, uncuttime,
    fix, set, auto, reset. Пример: ["cut 12-15 repeat", "pause 42 0.3", "set max_pause 0.4"]."""
    p = find(project)
    report = O.apply(p, commands, by="mcp")
    return "\n".join(report + [_summary(p)])


@mcp.tool()
@_safe
def undo(project: str = "last") -> str:
    """Отменить последнее изменение правок."""
    p = find(project)
    if not p.undo():
        return "Отменять нечего."
    p.log("mcp", "отмена последнего изменения")
    p.save()
    return "Отменено. " + _summary(p)


@mcp.tool()
@_safe
def render_audio(project: str = "last", format: str = "") -> str:
    """Собрать итоговое аудио (mp3/ogg/wav/flac/m4a). Возвращает путь к файлу."""
    p = find(project)
    r = render(p, format or None)
    record_render(p, r, "mcp")
    return f"Готово: {r['path']} ({fmt_t(r['original'])} → {fmt_t(r['duration'])})"


@mcp.tool()
@_safe
def export(project: str = "last", kind: str = "txt", original: bool = False) -> str:
    """Вернуть текст (txt) или субтитры (srt/vtt) — по результату или по исходнику (original=True)."""
    p = find(project)
    if kind == "txt":
        return export_text(p, original=original)
    return export_subs(p, original=original, vtt=kind == "vtt")


@mcp.tool()
@_safe
def stats(project: str = "last") -> str:
    """Сводка: длительности, паузы, низкая уверенность, заметки пользователя, история правок."""
    return _stats(find(project))


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
