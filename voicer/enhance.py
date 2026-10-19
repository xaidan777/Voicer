"""Нейросетевое шумоподавление: DeepFilterNet3 (MIT/Apache-2.0), 48 кГц, работает на CPU.

Убирает фоновые шумы (гул техники, шипение, улица, клавиатура), голос не трогает.
Результат для каждой силы подавления кэшируется в папке проекта: обработка идёт один раз.
"""
from __future__ import annotations

import math
import os
import sys
import threading
import types
from typing import Callable

import numpy as np
from scipy.signal import resample_poly

from . import config

MODEL = "DeepFilterNet3"
MODEL_URL = f"https://github.com/Rikorose/DeepFilterNet/raw/main/models/{MODEL}.zip"
CHUNK = 30.0     # сек — обработка кусками, чтобы длинные записи не съедали память
WARMUP = 1.5     # сек — разгон рекуррентной сети в начале куска (выбрасывается)
XFADE = 0.5      # сек — перекрёстное затухание между кусками

_state = None
_lock = threading.Lock()


def _compat() -> None:
    """DeepFilterNet 0.5.6 импортирует torchaudio.backend.common, которого нет в новом torchaudio."""
    try:
        import torchaudio.backend.common  # noqa: F401
    except ImportError:
        common = types.ModuleType("torchaudio.backend.common")
        common.AudioMetaData = type("AudioMetaData", (), {})
        backend = sys.modules.setdefault("torchaudio.backend", types.ModuleType("torchaudio.backend"))
        backend.common = common
        sys.modules["torchaudio.backend.common"] = common


def available() -> bool:
    try:
        _compat()
        import df  # noqa: F401
        return True
    except ImportError:
        return False


def _model():
    global _state
    with _lock:
        if _state is None:
            _compat()
            import df.logger
            import torch
            from df.enhance import init_df
            from df.utils import download_file

            # Логгер DeepFilterNet вызывает git ради номера коммита и сыпет «fatal: not a git repository».
            df.logger.get_commit_hash = lambda: None
            df.logger.get_branch_name = lambda: None

            torch.set_num_threads(max(1, os.cpu_count() or 1))
            base = config.MODELS_DIR / MODEL
            if not (base / "config.ini").exists():
                config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
                download_file(MODEL_URL, str(config.MODELS_DIR), extract=True)
            model, df_state, _ = init_df(str(base), log_level="WARNING", log_file=None)
            _state = (model, df_state)
    return _state


def denoise(x: np.ndarray, sr: int, atten_db: float | None = None,
            progress: Callable[[float], None] | None = None) -> np.ndarray:
    """x: [каналы, отсчёты]. atten_db — сколько дБ шума убрать максимум (None/≥100 — всё)."""
    _compat()
    import torch
    from df.enhance import enhance

    model, st = _model()
    dsr = st.sr()
    g = math.gcd(sr, dsr)
    y = resample_poly(x, dsr // g, sr // g, axis=-1).astype(np.float32) if sr != dsr else x.astype(np.float32)
    lim = None if atten_db is None or atten_db >= 100 else float(atten_db)

    n = y.shape[1]
    out = np.zeros_like(y)
    chunk, warm, xf = int(CHUNK * dsr), int(WARMUP * dsr), int(XFADE * dsr)
    ramp = np.linspace(0, 1, xf, dtype=np.float32)
    pos = 0
    while pos < n:
        a = max(0, pos - warm - xf)
        b = min(n, pos + chunk)
        st.reset()
        with torch.no_grad():
            e = enhance(model, st, torch.from_numpy(np.ascontiguousarray(y[:, a:b])), atten_lim_db=lim).numpy()
        if pos == 0:
            out[:, :b] = e
        else:
            k = pos - a          # сколько отсчётов перед pos в этом куске
            f0 = pos - xf        # начало перекрёстного затухания
            out[:, f0:pos] = out[:, f0:pos] * (1 - ramp) + e[:, k - xf:k] * ramp
            out[:, pos:b] = e[:, k:]
        pos = b
        if progress:
            progress(pos / n)
    if sr != dsr:
        out = resample_poly(out, sr // g, dsr // g, axis=-1).astype(np.float32)
    m = x.shape[1]
    if out.shape[1] < m:
        out = np.pad(out, ((0, 0), (0, m - out.shape[1])))
    return out[:, :m]
