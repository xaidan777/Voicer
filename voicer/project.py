"""Проект: исходник + транскрипт + правки. Общее состояние для пользователя и LLM."""
from __future__ import annotations

import copy
import json
import os
import shutil
import time
from pathlib import Path
from typing import Callable

from . import config
from .textproc import slugify, tokenize_text

SCHEMA = 1
UNDO_LIMIT = 100


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _write_json(path: Path, data) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    # Windows не даёт подменить файл, пока его читает другой запрос (опрос интерфейса) или антивирус.
    for attempt in range(40):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if attempt == 39:
                raise
            time.sleep(0.05)


def empty_edit() -> dict:
    return {"cuts": {}, "keeps": [], "ranges": [], "pauses": {}, "fixes": {}, "bounds": {}}


# Значения, которые были по умолчанию раньше: в старых проектах они не «выбор пользователя».
LEGACY_DEFAULTS = {"max_pause": (0.5,), "max_pause_sentence": (0.9,)}


class Settings(dict):
    """Настройки проекта = значения по умолчанию + то, что задано в этом проекте явно.

    На диск пишутся только явно заданные значения (overrides), поэтому новые значения
    по умолчанию сразу действуют во всех проектах, где их не меняли.
    """

    def __init__(self, overrides: dict):
        super().__init__({**config.DEFAULT_SETTINGS, **overrides})
        self.overrides = overrides

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        self.overrides[key] = value

    def update(self, *args, **kw):
        for k, v in dict(*args, **kw).items():
            self[k] = v

    def reset(self, key: str) -> None:
        """Вернуть значение по умолчанию."""
        self.overrides.pop(key, None)
        super().__setitem__(key, config.DEFAULT_SETTINGS[key])


def clean_overrides(saved: dict) -> dict:
    """Убирает из настроек проекта то, что совпадает со значениями по умолчанию (старыми или новыми)."""
    out = {}
    for k, v in saved.items():
        if k not in config.DEFAULT_SETTINGS:
            continue
        if v == config.DEFAULT_SETTINGS[k] or v in LEGACY_DEFAULTS.get(k, ()):
            continue
        out[k] = v
    # Студийный звук был выключен — снимаем галочку «Очистка голоса».
    studio = [k for k in config.STUDIO_KEYS if k in saved]
    if studio and not any(saved[k] for k in studio):
        out["voice_clean"] = False
    return out


class Project:
    def __init__(self, directory: Path):
        self.dir = Path(directory)
        self.data = json.loads((self.dir / "project.json").read_text(encoding="utf-8"))
        for k, v in empty_edit().items():
            self.data.setdefault("edit", {}).setdefault(k, v)
        saved = self.data.get("settings", {})
        # Раньше в проект копировались все значения (и старый сервер может записать их снова).
        if self.data.get("settings_v", 1) < 2 or len(saved) >= len(config.DEFAULT_SETTINGS):
            self.data["settings"] = clean_overrides(self.data.get("settings", {}))
            self.data["settings_v"] = 2
            _write_json(self.dir / "project.json", self.data)
        self._settings = Settings(self.data.setdefault("settings", {}))

    # --- свойства -------------------------------------------------------
    @property
    def id(self) -> str:
        return self.data["id"]

    @property
    def kind(self) -> str:
        return self.data["kind"]

    @property
    def tokens(self) -> list[dict]:
        return self.data["tokens"]

    @property
    def edit(self) -> dict:
        return self.data["edit"]

    @property
    def settings(self) -> Settings:
        return self._settings

    @property
    def source_path(self) -> Path | None:
        s = self.data.get("source")
        return self.dir / s if s else None

    @property
    def duration(self) -> float:
        return float((self.data.get("audio") or {}).get("duration") or 0.0)

    def text_of(self, i: int) -> str:
        return self.edit["fixes"].get(str(i), self.tokens[i]["t"])

    def cut_map(self) -> dict[int, str]:
        """Все вырезанные токены (явно и попавшие в вырезанные интервалы) → причина."""
        res = {int(k): v for k, v in self.edit["cuts"].items()}
        for s, e, why in self.edit["ranges"]:
            for t in self.tokens:
                if "s" in t and t["e"] > t["s"]:
                    ov = min(t["e"], e) - max(t["s"], s)
                    if ov > 0.5 * (t["e"] - t["s"]):
                        res.setdefault(t["i"], why or "range")
        return res

    # --- сохранение и история -------------------------------------------
    def state(self) -> dict:
        return {"edit": copy.deepcopy(self.edit), "settings": dict(self.settings.overrides),
                "settings_v": 2}

    def restore(self, state: dict) -> None:
        self.data["edit"] = state["edit"]
        st = state["settings"] if state.get("settings_v") == 2 else clean_overrides(state["settings"])
        self.data["settings"] = dict(st)
        self._settings = Settings(self.data["settings"])

    def snapshot(self, state: dict | None = None) -> None:
        """Кладёт состояние правок (по умолчанию текущее) в стек отмены."""
        f = self.dir / "undo.json"
        stack = json.loads(f.read_text(encoding="utf-8")) if f.exists() else []
        stack.append(state or self.state())
        _write_json(f, stack[-UNDO_LIMIT:])

    def undo(self) -> bool:
        f = self.dir / "undo.json"
        stack = json.loads(f.read_text(encoding="utf-8")) if f.exists() else []
        if not stack:
            return False
        self.restore(stack.pop())
        _write_json(f, stack)
        return True

    def log(self, by: str, msg: str) -> None:
        self.data.setdefault("log", []).append({"ts": _now(), "by": by, "msg": msg})
        self.data["log"] = self.data["log"][-300:]

    def save(self) -> None:
        self.data["rev"] = int(self.data.get("rev", 0)) + 1
        self.data["updated"] = _now()
        _write_json(self.dir / "project.json", self.data)

    def reload(self) -> "Project":
        return Project(self.dir)


# --- поиск проектов ---------------------------------------------------------

def list_projects() -> list[Project]:
    if not config.WORKSPACE.exists():
        return []
    out = []
    for d in config.WORKSPACE.iterdir():
        if (d / "project.json").exists():
            try:
                out.append(Project(d))
            except (json.JSONDecodeError, KeyError, OSError):
                continue
    out.sort(key=lambda p: p.data.get("updated", ""), reverse=True)
    return out


def find(ref: str | None) -> Project:
    projects = list_projects()
    if not projects:
        raise LookupError("Проектов нет. Создайте: voicer import <файл>")
    if ref in (None, "", "last", "."):
        return projects[0]
    for p in projects:
        if p.id == ref:
            return p
    hits = [p for p in projects if p.id.startswith(ref) or p.data.get("name") == ref]
    if len(hits) == 1:
        return hits[0]
    if not hits:
        raise LookupError(f"Проект «{ref}» не найден. Есть: {', '.join(p.id for p in projects)}")
    raise LookupError(f"Неоднозначно «{ref}»: {', '.join(p.id for p in hits)}")


def _new_dir(name: str) -> tuple[str, Path]:
    config.WORKSPACE.mkdir(parents=True, exist_ok=True)
    base = slugify(name)
    pid, n = base, 2
    while (config.WORKSPACE / pid).exists():
        pid, n = f"{base}-{n}", n + 1
    d = config.WORKSPACE / pid
    d.mkdir(parents=True)
    return pid, d


def _base(pid: str, name: str, kind: str) -> dict:
    return {"schema": SCHEMA, "id": pid, "name": name, "kind": kind, "created": _now(),
            "updated": _now(), "rev": 0, "status": "ready", "tokens": [], "edit": empty_edit(),
            "settings": {}, "settings_v": 2, "notes": "", "script": None, "log": []}


def _guess_lang(text: str) -> str:
    cyr = sum(1 for ch in text if "а" <= ch.lower() <= "я" or ch in "ёЁ")
    lat = sum(1 for ch in text if "a" <= ch.lower() <= "z")
    return "ru" if cyr >= lat else "en"


# --- создание ---------------------------------------------------------------

def create_text(text: str, name: str = "text", by: str = "cli", auto: bool = True) -> Project:
    pid, d = _new_dir(name)
    data = _base(pid, name, "text")
    data["language"] = _guess_lang(text)
    data["tokens"] = [{"i": i, "t": w, "kind": "word"} for i, w in enumerate(tokenize_text(text))]
    _write_json(d / "project.json", data)
    p = Project(d)
    p.log(by, f"создан из текста, {len(p.tokens)} слов")
    p.save()
    if auto:
        from .ops import apply

        apply(p, ["auto"], by="auto")
    return p


def create_audio(src: str | Path, name: str | None = None, language: str | None = None,
                 model: str | None = None, script: str | None = None, by: str = "cli",
                 auto: bool = True, filler_prompt: bool = True,
                 progress: Callable[[float, str], None] | None = None,
                 on_created: Callable[[str], None] | None = None) -> Project:
    from . import audio as A
    from .analyze import refine, stt_fillers
    from .transcribe import heard_words, transcribe

    src = Path(src)
    name = name or src.stem
    pid, d = _new_dir(name)
    source = "source" + src.suffix.lower()
    shutil.copyfile(src, d / source)
    data = _base(pid, name, "audio")
    data.update(source=source, original_name=src.name, status="processing", script=script)
    _write_json(d / "project.json", data)
    if on_created:
        on_created(pid)
    try:
        res = transcribe(d / source, language=language, model=model,
                         filler_prompt=filler_prompt, progress=progress)
        wav, sr = A.load(d / source)
        m = A.mono(wav)
        tokens, lv = refine(res["tokens"], m, sr)
        if progress:
            progress(0.98, "проверка заминок")
        tokens = stt_fillers(tokens, m, sr, heard_words(d / source, res["language"], res["model"]))
        _write_json(d / "peaks.json", A.peaks(m, A.peak_buckets(len(m), sr)))
        data.update(tokens=tokens, raw_tokens=res["tokens"], levels=lv, language=res["language"],
                    model=res["model"],
                    device=res["device"], status="ready",
                    audio={"sr": sr, "channels": int(wav.shape[0]),
                           "duration": round(wav.shape[1] / sr, 3)})
        _write_json(d / "project.json", data)
    except Exception as e:
        data["status"] = f"error: {e}"
        _write_json(d / "project.json", data)
        raise
    p = Project(d)
    nsound = sum(1 for t in tokens if t["kind"] == "sound")
    p.log(by, f"распознано: {len(tokens) - nsound} слов, {nsound} звуков, язык {res['language']}, "
              f"модель {res['model']} ({res['device']})")
    p.save()
    if auto:
        from .ops import apply

        apply(p, ["auto"], by="auto")
    return p


def reanalyze(p: Project, by: str = "cli") -> str:
    """Заново ищет паузы и звуки по сигналу (без повторного распознавания) и переносит правки."""
    from . import audio as A
    from .analyze import refine, stt_fillers
    from .transcribe import heard_words

    if p.kind != "audio":
        raise ValueError("Пересчёт пауз только для аудиопроектов")
    old = p.tokens
    raw = p.data.get("raw_tokens") or [{k: t[k] for k in ("t", "s", "e", "p") if k in t}
                                       for t in old if t.get("kind", "word") == "word"]
    wav, sr = A.load(p.source_path)
    m = A.mono(wav)
    new, lv = refine(raw, m, sr)
    new = stt_fillers(new, m, sr, heard_words(p.source_path, p.data.get("language"), p.data.get("model")))

    old_words = [t["i"] for t in old if t.get("kind", "word") == "word"]
    new_words = [t["i"] for t in new if t["kind"] == "word"]
    remap = dict(zip(old_words, new_words)) if len(old_words) == len(new_words) else {}
    new_sounds = [t for t in new if t["kind"] == "sound"]
    for t in old:
        if t.get("kind") == "sound":
            hits = [(min(t["e"], n["e"]) - max(t["s"], n["s"]), n["i"]) for n in new_sounds]
            hits = [h for h in hits if h[0] > 0]
            if hits:
                remap[t["i"]] = max(hits)[1]

    ed = p.edit
    lost = sum(1 for k in list(ed["cuts"]) + list(ed["pauses"]) + list(ed["fixes"]) + list(ed["bounds"])
               if int(k) not in remap)
    lost += sum(1 for k in ed["keeps"] if k not in remap)
    for key in ("cuts", "pauses", "fixes", "bounds"):
        ed[key] = {str(remap[int(k)]): v for k, v in ed[key].items() if int(k) in remap}
    ed["keeps"] = [remap[k] for k in ed["keeps"] if k in remap]
    p.data.update(tokens=new, levels=lv)
    (p.dir / "undo.json").unlink(missing_ok=True)  # индексы токенов изменились — старая история отмены неприменима
    gaps = sum(1 for a, b in zip(new, new[1:]) if b["s"] - a["e"] >= 0.3)
    nsound = len(new_sounds)
    msg = f"паузы пересчитаны: пауз ≥0.3с {gaps}, звуков {nsound}" + (f", потеряно правок: {lost}" if lost else "")
    p.log(by, msg)
    p.save()
    return msg


def delete(p: Project) -> None:
    shutil.rmtree(p.dir)
