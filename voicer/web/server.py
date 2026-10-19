"""Веб-интерфейс: загрузка, просмотр транскрипта, ручные правки, рендер. Изменения от LLM видны сразу."""
from __future__ import annotations

import json
import shutil
import threading
import uuid
import webbrowser
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fastapi import Body, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from .. import config
from ..audio import FORMATS
from ..ops import apply
from ..pack import create_pack
from ..project import Project, create_audio, create_text, delete, find, list_projects, reanalyze
from ..render import plan, record_render, render
from ..textproc import is_strict_filler, soft_fillers

STATIC = Path(__file__).parent / "static"


class NoCacheStatic(StaticFiles):
    """Браузер обязан сверяться с сервером, иначе после обновления остаётся старый app.js."""

    def file_response(self, *args, **kwargs):
        resp = super().file_response(*args, **kwargs)
        resp.headers["Cache-Control"] = "no-cache"
        return resp


app = FastAPI(title="Voicer")
app.mount("/static", NoCacheStatic(directory=STATIC), name="static")

_pool = ThreadPoolExecutor(max_workers=1)
jobs: dict[str, dict] = {}


def _get(pid: str) -> Project:
    try:
        return find(pid)
    except LookupError as e:
        raise HTTPException(404, str(e)) from None


def _mtime(p: Project) -> int:
    return (p.dir / "project.json").stat().st_mtime_ns


def payload(p: Project) -> dict:
    words = [t for t in p.tokens if t.get("kind", "word") == "word"]
    lang = p.data.get("language")
    soft = soft_fillers([p.text_of(t["i"]) for t in words], lang)
    data = {k: p.data.get(k) for k in ("id", "name", "kind", "status", "language", "model", "device",
                                         "notes", "script", "rev", "updated", "original_name",
                                         "last_render")}
    data.update(
        tokens=p.tokens, edit=p.edit, settings=dict(p.settings), cuts=p.cut_map(),
        settings_custom=sorted(p.settings.overrides), settings_defaults=config.DEFAULT_SETTINGS,
        duration=p.duration, log=p.data.get("log", [])[-40:], mtime=_mtime(p),
        flags={"soft": [words[k]["i"] for k in soft],
               "strict": [t["i"] for t in words if is_strict_filler(p.text_of(t["i"]), lang)]},
        renders=sorted(f.name for f in (p.dir / "out").glob("*")) if (p.dir / "out").exists() else [],
    )
    if p.kind == "audio" and p.data.get("status") == "ready":
        pl = plan(p)
        data.update(segments=pl["segments"], map=pl["map"], edited_duration=pl["duration"])
    return data


@app.get("/")
def index():
    # Версия = время изменения файлов интерфейса: новый код — новые адреса скриптов.
    v = str(max(int(f.stat().st_mtime) for f in STATIC.iterdir()))
    html = (STATIC / "index.html").read_text(encoding="utf-8").replace("__V__", v)
    return HTMLResponse(html, headers={"Cache-Control": "no-cache"})


@app.get("/api/projects")
def projects():
    out = []
    for p in list_projects():
        out.append({"id": p.id, "name": p.data.get("name"), "kind": p.kind, "status": p.data.get("status"),
                    "duration": p.duration, "words": len(p.tokens), "updated": p.data.get("updated")})
    return out


@app.get("/api/projects/{pid}")
def project(pid: str):
    return payload(_get(pid))


@app.get("/api/projects/{pid}/rev")
def rev(pid: str):
    p = _get(pid)
    last = (p.data.get("log") or [{}])[-1]
    return {"mtime": _mtime(p), "status": p.data.get("status"), "by": last.get("by"), "msg": last.get("msg")}


@app.post("/api/projects/{pid}/ops")
def ops(pid: str, body: dict = Body(...)):
    p = _get(pid)
    cmds = body.get("commands") or []
    if isinstance(cmds, str):
        cmds = [cmds]
    try:
        report = apply(p, cmds, by=body.get("by", "ui"))
    except ValueError as e:
        raise HTTPException(400, str(e)) from None
    return {"report": report, "project": payload(p)}


@app.post("/api/projects/{pid}/undo")
def undo(pid: str):
    p = _get(pid)
    if not p.undo():
        raise HTTPException(400, "Отменять нечего")
    p.log("ui", "отмена последнего изменения")
    p.save()
    return payload(p)


@app.post("/api/projects/{pid}/reanalyze")
def do_reanalyze(pid: str):
    p = _get(pid)
    try:
        msg = reanalyze(p, by="ui")
    except ValueError as e:
        raise HTTPException(400, str(e)) from None
    return {"msg": msg, "project": payload(p)}


@app.post("/api/projects/{pid}/notes")
def notes(pid: str, body: dict = Body(...)):
    p = _get(pid)
    p.data["notes"] = body.get("notes", "")
    p.log("ui", "заметки обновлены")
    p.save()
    return {"ok": True}


@app.post("/api/projects/{pid}/render")
def do_render(pid: str, body: dict = Body(default={})):
    p = _get(pid)
    fmt = body.get("format") or p.settings.get("format", "mp3")
    if fmt not in FORMATS:
        raise HTTPException(400, f"формат: {', '.join(FORMATS)}")
    try:
        r = render(p, fmt)
    except ValueError as e:
        raise HTTPException(400, str(e)) from None
    record_render(p, r, "ui")
    name = Path(r["path"]).name
    return {**r, "url": f"/api/projects/{p.id}/out/{name}?v={_mtime(p)}", "name": name}


@app.get("/api/projects/{pid}/source")
def source(pid: str):
    p = _get(pid)
    if not p.source_path or not p.source_path.exists():
        raise HTTPException(404, "нет аудио")
    return FileResponse(p.source_path)


@app.get("/api/projects/{pid}/out/{name}")
def out_file(pid: str, name: str, download: int = 0):
    p = _get(pid)
    f = (p.dir / "out" / name).resolve()
    if f.parent != (p.dir / "out").resolve() or not f.exists():
        raise HTTPException(404, "нет файла")
    return FileResponse(f, filename=name if download else None)


@app.get("/api/projects/{pid}/peaks")
def peaks(pid: str):
    from .. import audio as A

    p = _get(pid)
    f = p.dir / "peaks.json"
    data = json.loads(f.read_text(encoding="utf-8")) if f.exists() else []
    if p.kind == "audio" and p.source_path and p.source_path.exists() and len(data) < p.duration * 50:
        y, sr = A.load(p.source_path)  # старые проекты: огибающая была грубой — пересчитываем
        m = A.mono(y)
        data = A.peaks(m, A.peak_buckets(len(m), sr))
        f.write_text(json.dumps(data), encoding="utf-8")
    return JSONResponse(data)


@app.get("/api/projects/{pid}/peaks_out")
def peaks_out(pid: str):
    """Огибающая последнего результата (для старых рендеров считается по файлу и кэшируется)."""
    from .. import audio as A
    from ..render import save_out_peaks

    p = _get(pid)
    last = p.data.get("last_render") or {}
    f = p.dir / "peaks_out.json"
    if f.exists():
        data = json.loads(f.read_text(encoding="utf-8"))
        if data.get("name") == last.get("name") and len(data.get("peaks", [])) >= data.get("duration", 0) * 50:
            return data
    out = p.dir / "out" / str(last.get("name", ""))
    if not last or not out.is_file():
        return {}
    y, sr = A.load(out)
    return save_out_peaks(p, out.name, A.mono(y), sr)


@app.get("/api/projects/{pid}/export/{kind}")
def export(pid: str, kind: str, orig: int = 0):
    from ..render import export_subs, export_text

    p = _get(pid)
    try:
        if kind == "txt":
            data = export_text(p, original=bool(orig))
        elif kind in ("srt", "vtt"):
            data = export_subs(p, original=bool(orig), vtt=kind == "vtt")
        else:
            raise HTTPException(400, "txt | srt | vtt")
    except ValueError as e:
        raise HTTPException(400, str(e)) from None
    name = f"{p.id}{'-orig' if orig else ''}.{kind}"
    return Response(data, media_type="text/plain; charset=utf-8",
                    headers={"Content-Disposition": f'attachment; filename="{name}"'})


@app.delete("/api/projects/{pid}")
def remove(pid: str):
    delete(_get(pid))
    return {"ok": True}


@app.post("/api/text")
def new_text(body: dict = Body(...)):
    text = (body.get("text") or "").strip()
    if not text:
        raise HTTPException(400, "пустой текст")
    p = create_text(text, body.get("name") or "text", by="ui", auto=body.get("auto", True))
    return {"id": p.id}


@app.post("/api/upload")
async def upload(file: UploadFile = File(...), lang: str = Form(""), model: str = Form(""),
                 auto: bool = Form(True), name: str = Form("")):
    fname = Path(file.filename or "audio").name
    ext = Path(fname).suffix.lower()
    if ext not in config.AUDIO_EXTS | config.TEXT_EXTS:
        raise HTTPException(400, f"Неподдерживаемый формат {ext}")
    if ext in config.TEXT_EXTS:
        text = (await file.read()).decode("utf-8", errors="replace")
        p = create_text(text, name or Path(fname).stem, by="ui", auto=auto)
        return {"id": p.id}
    tmp = config.WORKSPACE / "_uploads" / uuid.uuid4().hex
    tmp.mkdir(parents=True, exist_ok=True)
    path = tmp / fname
    with path.open("wb") as f:
        shutil.copyfileobj(file.file, f)
    jid = uuid.uuid4().hex[:8]
    jobs[jid] = {"id": jid, "file": fname, "progress": 0.0, "msg": "в очереди", "project": None,
                 "error": None, "done": False}
    _pool.submit(_import_job, jid, path, name or Path(fname).stem, lang or None, model or None, auto)
    return {"job": jid}


@app.post("/api/upload_pack")
async def upload_pack(files: list[UploadFile] = File(...), lang: str = Form(""), model: str = Form(""),
                      auto: bool = Form(True), name: str = Form("")):
    """Пак: записи склеиваются по порядку имён в одну дорожку, ранние дубли вычёркиваются."""
    names = [Path(f.filename or "audio").name for f in files]
    bad = sorted({Path(n).suffix.lower() for n in names} - config.AUDIO_EXTS)
    if bad:
        raise HTTPException(400, f"В пак — только аудио, а не {', '.join(bad)}")
    tmp = config.WORKSPACE / "_uploads" / uuid.uuid4().hex
    tmp.mkdir(parents=True, exist_ok=True)
    paths = []
    for f, fname in zip(files, names):
        path = tmp / fname
        n = 2
        while path.exists():  # одинаковые имена из разных папок
            path = tmp / f"{Path(fname).stem} ({n}){Path(fname).suffix}"
            n += 1
        with path.open("wb") as out:
            shutil.copyfileobj(f.file, out)
        paths.append(path)
    jid = uuid.uuid4().hex[:8]
    jobs[jid] = {"id": jid, "file": f"пак, файлов: {len(paths)}", "progress": 0.0, "msg": "в очереди",
                 "project": None, "error": None, "done": False}
    _pool.submit(_pack_job, jid, paths, name or None, lang or None, model or None, auto)
    return {"job": jid}


def _pack_job(jid: str, paths: list[Path], name, lang, model, auto) -> None:
    job = jobs[jid]

    def progress(frac: float, msg: str) -> None:
        job["progress"], job["msg"] = round(frac, 3), msg

    try:
        create_pack(paths, name=name, language=lang, model=model, by="ui", auto=auto, progress=progress,
                    on_created=lambda pid: job.update(project=pid))
        job.update(progress=1.0, msg="готово")
    except Exception as e:  # noqa: BLE001 — показываем пользователю любую ошибку
        job["error"] = str(e)
    finally:
        job["done"] = True
        shutil.rmtree(paths[0].parent, ignore_errors=True)


def _import_job(jid: str, path: Path, name: str, lang, model, auto) -> None:
    job = jobs[jid]

    def progress(frac: float, msg: str) -> None:
        job["progress"], job["msg"] = round(frac, 3), msg

    try:
        create_audio(path, name=name, language=lang, model=model, by="ui", auto=auto, progress=progress,
                     on_created=lambda pid: job.update(project=pid))
        job.update(progress=1.0, msg="готово")
    except Exception as e:  # noqa: BLE001 — показываем пользователю любую ошибку
        job["error"] = str(e)
    finally:
        job["done"] = True
        shutil.rmtree(path.parent, ignore_errors=True)


@app.get("/api/jobs")
def list_jobs():
    return [j for j in jobs.values() if not j["done"] or j["error"]]


@app.delete("/api/jobs/{jid}")
def drop_job(jid: str):
    jobs.pop(jid, None)
    return {"ok": True}


def run(host: str = "127.0.0.1", port: int = 8765, open_browser: bool = True) -> None:
    import uvicorn

    url = f"http://{host}:{port}/"
    print(f"Voicer: {url}  (проекты: {config.WORKSPACE})")
    if open_browser:
        threading.Timer(1.2, lambda: webbrowser.open(url)).start()
    uvicorn.run(app, host=host, port=port, log_level="warning")
