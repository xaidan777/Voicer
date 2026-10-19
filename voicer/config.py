"""Пути и настройки по умолчанию."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WORKSPACE = Path(os.environ.get("VOICER_HOME", ROOT / "projects"))
MODELS_DIR = Path(os.environ.get("VOICER_MODELS", ROOT / "models"))

# Модель Whisper: на GPU — large-v3-turbo (быстро и точно), на CPU — small.
GPU_MODEL = os.environ.get("VOICER_MODEL", "large-v3-turbo")
CPU_MODEL = os.environ.get("VOICER_CPU_MODEL", "small")

AUDIO_EXTS = {".mp3", ".ogg", ".oga", ".opus", ".wav", ".flac", ".m4a", ".aac",
              ".wma", ".webm", ".mp4", ".mkv", ".mov"}
TEXT_EXTS = {".txt", ".md"}

# Настройки монтажа (секунды). Хранятся в каждом проекте и меняются командой `set`.
DEFAULT_SETTINGS = {
    "max_pause": 0.15,          # пауза внутри фразы длиннее этого — сжимается
    "max_pause_sentence": 0.5,  # пауза после конца предложения (. ! ? …)
    "join_pause": 0.12,         # минимальная пауза на месте вырезанного фрагмента
    "lead": 0.25,               # тишина в начале
    "tail": 0.5,                # тишина в конце
    "crossfade_ms": 15,         # кроссфейд на склейках
    "lufs": -16.0,              # нормализация громкости (None — выключено), только с «Очисткой голоса»
    "format": "mp3",            # формат рендера по умолчанию
    "voice_clean": True,        # «Очистка голоса»: без галочки — только монтаж (ни параметров ниже, ни lufs)
    # параметры очистки голоса (0 — ступень выключена)
    "denoise": 100,             # шумоподавление DeepFilterNet: сколько дБ шума убрать (100 — всё)
    "dereverb": 1.5,            # подавление эха комнаты, 0..2
    "gate": 32,                 # гейт: глушить всё, что тише речи на столько дБ (паузы, хвосты эха)
    "dehum": "auto",            # гул сети: 0 | 50 | 60 | auto
    "highpass": 80,             # срез низов, Гц
    "deess": 6,                 # де-эссер: макс. ослабление «с», «ш», дБ
    "leveler": 0.6,             # выравнивание громкости фраз, 0..1
    "compress": 2.0,            # компрессор, степень сжатия (2–3 — мягко)
    "limit": -1.0,              # лимитер: потолок пиков, дБ (None — выключен)
}

# Пресет «студийный звук»: команда `studio` и кнопка в интерфейсе.
STUDIO_KEYS = ("denoise", "dereverb", "gate", "dehum", "highpass", "deess", "leveler", "compress")
STUDIO_PRESET = {k: DEFAULT_SETTINGS[k] for k in STUDIO_KEYS + ("limit",)}

SETTING_HELP = {
    "max_pause": "макс. пауза внутри фразы, сек",
    "max_pause_sentence": "макс. пауза между предложениями, сек",
    "join_pause": "мин. пауза на месте вырезанного, сек",
    "lead": "тишина в начале, сек",
    "tail": "тишина в конце, сек",
    "crossfade_ms": "кроссфейд на склейках, мс",
    "lufs": "целевая громкость LUFS или off",
    "format": "mp3 | ogg | wav | flac | m4a",
    "voice_clean": "очистка голоса: on | off",
    "denoise": "шумоподавление, дБ (0 — выкл, 100 — всё)",
    "dereverb": "подавление эха комнаты, 0..2 (0 — выкл)",
    "gate": "гейт пауз: на сколько дБ ниже речи глушить (0 — выкл)",
    "dehum": "гул сети: 0 | 50 | 60 | auto",
    "highpass": "срез низов, Гц (0 — выкл)",
    "deess": "де-эссер, дБ (0 — выкл)",
    "leveler": "выравнивание громкости фраз, 0..1",
    "compress": "компрессор, степень (0 — выкл)",
    "limit": "лимитер, потолок дБ или off",
}
