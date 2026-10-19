"""Распознавание речи (faster-whisper) с пословными таймкодами."""
from __future__ import annotations

import glob
import os
import site
import sys
import threading
from typing import Callable

from . import config
from .textproc import FILLER_PROMPTS

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

_models: dict = {}
_lock = threading.Lock()


def _add_cuda_paths() -> None:
    """Подключает cuBLAS/cuDNN из pip-пакетов nvidia-* (Windows ищет DLL по PATH)."""
    roots = list(site.getsitepackages()) + [site.getusersitepackages(), sys.prefix]
    for root in roots:
        for d in glob.glob(os.path.join(root, "**", "nvidia", "*", "bin"), recursive=True):
            if d not in os.environ.get("PATH", ""):
                os.environ["PATH"] = d + os.pathsep + os.environ.get("PATH", "")
                if hasattr(os, "add_dll_directory"):
                    try:
                        os.add_dll_directory(d)
                    except OSError:
                        pass


def get_model(name: str | None = None, device: str | None = None):
    _add_cuda_paths()
    import ctranslate2
    from faster_whisper import WhisperModel

    if device is None:
        device = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
    if not name:
        name = config.GPU_MODEL if device == "cuda" else config.CPU_MODEL
    key = (name, device)
    with _lock:
        if key not in _models:
            compute = "float16" if device == "cuda" else "int8"
            config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
            _models[key] = WhisperModel(name, device=device, compute_type=compute,
                                        download_root=str(config.MODELS_DIR))
    return _models[key], name, device


def transcribe(path, language: str | None = None, model: str | None = None,
               filler_prompt: bool = True, device: str | None = None,
               progress: Callable[[float, str], None] | None = None) -> dict:
    """Возвращает {language, model, device, duration, tokens:[{t,s,e,p}]}."""
    from faster_whisper.audio import decode_audio

    say = progress or (lambda f, msg: None)
    say(0.0, "декодирование")
    audio = decode_audio(str(path), sampling_rate=16000)
    duration = len(audio) / 16000.0

    say(0.02, "загрузка модели (первый раз — скачивание)")
    m, name, device = get_model(model, device)
    try:
        return _run(m, name, device, audio, duration, language, filler_prompt, say)
    except RuntimeError as e:
        if device == "cuda" and any(k in str(e).lower() for k in ("cublas", "cudnn", "cuda")):
            say(0.02, f"GPU недоступен ({e}); переключаюсь на CPU")
            m, name, device = get_model(model if model else None, "cpu")
            return _run(m, name, device, audio, duration, language, filler_prompt, say)
        raise


def _run(m, name, device, audio, duration, language, filler_prompt, say) -> dict:
    lang = language if language and language != "auto" else None
    if lang is None:
        try:
            lang, _, _ = m.detect_language(audio)
        except Exception:
            lang = None
    prompt = FILLER_PROMPTS.get(lang or "") if filler_prompt else None

    say(0.05, "распознавание")
    segments, info = m.transcribe(
        audio,
        language=lang,
        initial_prompt=prompt,
        word_timestamps=True,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 1000, "speech_pad_ms": 300},
        beam_size=5,
    )
    tokens = []
    for seg in segments:
        for w in seg.words or []:
            t = w.word.strip()
            if not t:
                continue
            # Whisper дробит «э-э-э» и «что-то» на «э», «-э», «-э» — склеиваем обратно в одно слово.
            if tokens and len(t) > 1 and t[0] == "-" and t[1].isalpha() and w.start - tokens[-1]["e"] < 0.5:
                prev = tokens[-1]
                prev.update(t=prev["t"] + t, e=round(float(w.end), 3),
                            p=round(min(prev["p"], float(w.probability)), 3))
                continue
            tokens.append({"t": t, "s": round(float(w.start), 3), "e": round(float(w.end), 3),
                           "p": round(float(w.probability), 3)})
        say(0.05 + 0.9 * min(1.0, seg.end / max(duration, 1e-6)), "распознавание")
    say(0.97, "готово")
    return {"language": info.language, "model": name, "device": device,
            "duration": round(duration, 3), "tokens": tokens}


def heard_words(path, language: str | None = None, model: str | None = None):
    """Функция heard(s, e): какие слова распознаватель слышит на участке [s, e] (без подсказок)."""
    from faster_whisper.audio import decode_audio
    import numpy as np

    audio = decode_audio(str(path), sampling_rate=16000)
    m, _, _ = get_model(model)

    def heard(s: float, e: float) -> list[str]:
        clip = audio[max(0, int((s - 0.05) * 16000)):int((e + 0.05) * 16000)]
        clip = np.pad(clip, (8000, 8000))  # полсекунды тишины по краям
        segs, _ = m.transcribe(clip, language=language, beam_size=5, vad_filter=False,
                               condition_on_previous_text=False, without_timestamps=True)
        return [w for seg in segs if seg.no_speech_prob < 0.6 for w in seg.text.split()]

    return heard
