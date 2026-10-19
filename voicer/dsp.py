"""«Студийная» обработка голоса без нейросетей: фильтры, де-эссер, левелер, компрессор, лимитер.

Все функции принимают и возвращают массив [каналы, отсчёты] float32 и работают без фазовых сдвигов
(фильтрация вперёд-назад), поэтому их можно применять к уже смонтированной записи.
"""
from __future__ import annotations

import numpy as np
from scipy import signal
from scipy.ndimage import gaussian_filter1d, minimum_filter1d, uniform_filter1d


def _db(x: np.ndarray) -> np.ndarray:
    return 20 * np.log10(np.maximum(x, 1e-9))


def _mono(x: np.ndarray) -> np.ndarray:
    return x.mean(axis=0) if x.shape[0] > 1 else x[0]


def _rms_db(m: np.ndarray, sr: int, win: float, hop: float) -> tuple[np.ndarray, int]:
    """Скользящий RMS в дБ: (уровни по кадрам, шаг в отсчётах)."""
    h = max(1, int(sr * hop))
    p = uniform_filter1d(m.astype(np.float64) ** 2, size=max(1, int(sr * win)), mode="nearest")[::h]
    return 10 * np.log10(np.maximum(p, 1e-18)), h


def _apply_gain_db(x: np.ndarray, gain_db: np.ndarray, hop: int) -> np.ndarray:
    n = x.shape[1]
    centers = np.arange(len(gain_db)) * hop
    g = np.interp(np.arange(n), centers, gain_db)
    return (x * (10 ** (g / 20))).astype(np.float32)


# --- фильтры ------------------------------------------------------------------

def highpass(x: np.ndarray, sr: int, hz: float) -> np.ndarray:
    """Срез низов: гул, стук по столу, «бубнение» от ветра и дыхания в микрофон."""
    sos = signal.butter(2, hz, "highpass", fs=sr, output="sos")
    return signal.sosfiltfilt(sos, x, axis=-1).astype(np.float32)


def detect_hum(x: np.ndarray, sr: int) -> int:
    """Возвращает 50 или 60, если в записи есть сетевой гул, иначе 0."""
    m = _mono(x)[: sr * 60]
    if len(m) < sr * 2:
        return 0
    f, pxx = signal.welch(m, fs=sr, nperseg=sr * 2)  # разрешение 0.5 Гц
    db = 10 * np.log10(pxx + 1e-20)
    best, best_score = 0, 0.0
    for f0 in (50, 60):
        scores = []
        for k in (1, 2, 3):
            fk = f0 * k
            peak = db[np.abs(f - fk) <= 1.0].max()
            ring = db[(np.abs(f - fk) > 2.0) & (np.abs(f - fk) <= 8.0)]
            scores.append(peak - np.median(ring))
        score = float(np.mean(sorted(scores)[1:]))  # два лучших из трёх
        if score > best_score:
            best, best_score = f0, score
    return best if best_score >= 10.0 else 0


def dehum(x: np.ndarray, sr: int, f0: float, harmonics: int = 8) -> np.ndarray:
    """Узкие вырезающие фильтры на частоте сети и её гармониках."""
    y = x
    for k in range(1, harmonics + 1):
        fk = f0 * k
        if fk >= sr / 2 * 0.9:
            break
        b, a = signal.iirnotch(fk, Q=30 + 5 * k, fs=sr)
        y = signal.filtfilt(b, a, y, axis=-1)
    return y.astype(np.float32)


# --- эхо комнаты и паузы ------------------------------------------------------------

def estimate_t60(x: np.ndarray, sr: int) -> float:
    """Время реверберации комнаты по затуханию звука после концов слов (медиана наклонов)."""
    lv, h = _rms_db(_mono(x), sr, 0.01, 0.005)
    top = np.percentile(lv, 95)
    floor = np.percentile(lv, 10)
    slopes = []
    i, n = 0, len(lv)
    step = h / sr
    while i < n - 2:
        if lv[i] > top - 12 and lv[i + 1] < lv[i]:
            j = i
            while j + 1 < n and lv[j + 1] < lv[j] + 1.0 and lv[j + 1] > floor + 8:
                j += 1
            seg = lv[i:j + 1]
            if len(seg) * step >= 0.06 and seg[0] - seg[-1] >= 12:
                # берём хвост: от -10 дБ ниже начала спада (сама согласная уже закончилась)
                k = np.argmax(seg < seg[0] - 10)
                tail = seg[k:]
                if len(tail) >= 6:
                    slope = -np.polyfit(np.arange(len(tail)) * step, tail, 1)[0]
                    if slope > 0:
                        slopes.append(slope)
            i = j + 1
        else:
            i += 1
    if len(slopes) < 3:
        return 0.5
    return float(np.clip(60.0 / np.median(slopes), 0.2, 1.5))


def dereverb(x: np.ndarray, sr: int, strength: float = 1.0, t60: float | None = None,
             floor_db: float = -18.0) -> np.ndarray:
    """Подавление позднего эха (метод Лебара): эхо = задержанная и затухшая копия сигнала."""
    if strength <= 0:
        return x
    t60 = t60 or estimate_t60(x, sr)
    nper = 1 << int(round(np.log2(sr * 0.032)))
    hop = nper // 4
    td = 0.05  # раньше 50 мс — «ранние отражения», часть голоса; позже — эхо
    nd = max(1, int(round(td * sr / hop)))
    decay = np.exp(-2 * (3 * np.log(10) / t60) * td)
    gmin = 10 ** (floor_db / 20)
    out = np.empty_like(x)
    for c in range(x.shape[0]):
        _, _, z = signal.stft(x[c], fs=sr, nperseg=nper, noverlap=nper - hop)
        p = np.abs(z) ** 2
        ps = signal.lfilter([0.4], [1, -0.6], p, axis=1)  # сглаживание по времени
        late = np.zeros_like(p)
        late[:, nd:] = decay * ps[:, :-nd]
        g = 1 - strength * late / (ps + 1e-12)
        g = np.clip(g, gmin, 1.0)
        g = uniform_filter1d(g, size=3, axis=0)  # по частоте — против «музыкального шума»
        g = signal.lfilter([0.5], [1, -0.5], g, axis=1)
        _, y = signal.istft(z * g, fs=sr, nperseg=nper, noverlap=nper - hop)
        out[c] = y[: x.shape[1]] if len(y) >= x.shape[1] else np.pad(y, (0, x.shape[1] - len(y)))
    return out.astype(np.float32)


def gate(x: np.ndarray, sr: int, depth_db: float = 45.0, below_db: float = 32.0,
         ratio: float = 4.0, hold: float = 0.05, release: float = 0.06,
         lookahead: float = 0.01) -> np.ndarray:
    """Экспандер: всё тише (речь − below_db) прижимается к тишине — паузы и хвосты эха.

    Удержание (hold) и упреждение не дают съесть тихие согласные в начале и конце слов.
    """
    step = 0.0025
    lv, h = _rms_db(_mono(x), sr, 0.005, step)
    speech = np.percentile(lv, 95)
    thr = speech - below_db
    want = -np.minimum(depth_db, np.maximum(thr - lv, 0) * (ratio - 1))
    # удержание и упреждение: открытое состояние распространяется на hold вперёд и 10 мс назад
    opened = lv > thr
    k_hold, k_ahead = int(hold / step), int(lookahead / step)
    span = np.convolve(opened.astype(float), np.ones(k_hold + k_ahead + 1), mode="full")
    opened_ext = span[k_ahead:k_ahead + len(lv)] > 0
    want[opened_ext] = 0.0
    # быстрая атака, плавное отпускание
    a_rel = np.exp(-step / release)
    g = np.empty_like(want)
    cur = 0.0
    for i, w in enumerate(want):
        cur = w if w > cur else a_rel * cur + (1 - a_rel) * w
        g[i] = cur
    return _apply_gain_db(x, g, h)


# --- динамика -------------------------------------------------------------------

def deess(x: np.ndarray, sr: int, max_db: float) -> np.ndarray:
    """Де-эссер: приглушает свистящие «с», «ш», «ч», когда они резко выделяются по яркости."""
    hi = min(10_000, sr / 2 * 0.9)
    if hi <= 5_000:
        return x
    sos = signal.butter(2, [4_500, hi], "bandpass", fs=sr, output="sos")
    band = signal.sosfiltfilt(sos, x, axis=-1)
    hop = 0.002
    lb, h = _rms_db(_mono(band), sr, 0.006, hop)
    lf, _ = _rms_db(_mono(x), sr, 0.006, hop)
    speech = np.percentile(lf, 95)
    # Свист: в полосе почти вся энергия кадра, и кадр не тихий.
    over = (lb - lf + 9.0) * 1.5
    red = np.clip(over, 0, max_db) * (lf > speech - 35)
    red = gaussian_filter1d(red, sigma=0.004 / hop)
    g = 10 ** (-np.interp(np.arange(x.shape[1]), np.arange(len(red)) * h, red) / 20)
    return (x - band * (1 - g)).astype(np.float32)


def leveler(x: np.ndarray, sr: int, strength: float, max_db: float = 12.0) -> np.ndarray:
    """Выравнивает громкость фраз (тихие подтягивает, громкие опускает), паузы не раздувает."""
    lv, h = _rms_db(_mono(x), sr, 0.4, 0.05)
    gate = max(np.percentile(lv, 95) - 25, -65)
    speech = lv > gate
    if speech.sum() < 10:
        return x
    target = np.median(lv[speech])
    want = np.clip((target - lv) * strength, -max_db, max_db)
    idx = np.where(speech)[0]
    gain = np.interp(np.arange(len(lv)), idx, want[idx])
    gain[~speech] = np.minimum(gain[~speech], 0.0)  # шум в паузах не усиливаем
    gain = gaussian_filter1d(gain, sigma=0.25 / 0.05)
    return _apply_gain_db(x, gain, h)


def compress(x: np.ndarray, sr: int, ratio: float) -> np.ndarray:
    """Мягкий компрессор: громкие слоги и выкрики ближе к общему уровню."""
    if ratio <= 1.0:
        return x
    lv, h = _rms_db(_mono(x), sr, 0.02, 0.005)
    speech = lv > max(np.percentile(lv, 95) - 25, -65)
    if speech.sum() < 10:
        return x
    thr = np.percentile(lv[speech], 70)
    gr = np.maximum(lv - thr, 0) * (1 - 1 / ratio)
    # быстрая атака (≈10 мс), плавное отпускание (≈80 мс)
    gr = minimum_filter1d(-gr, size=3)  # удерживаем пик
    gr = gaussian_filter1d(-gr, sigma=0.04 / 0.005)
    return _apply_gain_db(x, -gr, h)


def limiter(x: np.ndarray, sr: int, ceiling_db: float = -1.0, lookahead: float = 0.005) -> np.ndarray:
    """Лимитер с упреждением: пики не выше ceiling_db без клиппинга."""
    ceil = 10 ** (ceiling_db / 20)
    peak = np.abs(x).max(axis=0)
    if peak.max() <= ceil:
        return x
    need = np.minimum(1.0, ceil / np.maximum(peak, 1e-9))
    w = max(1, int(sr * lookahead))
    g = minimum_filter1d(need, size=2 * w + 1)
    g = uniform_filter1d(g, size=w)
    g = minimum_filter1d(g, size=3)
    y = x * g
    return np.clip(y, -ceil, ceil).astype(np.float32)


def loudness(x: np.ndarray, sr: int, target_lufs: float | None, ceiling_db: float | None) -> np.ndarray:
    """Громкость по EBU R128 + лимитер (а не снижение всей громкости ради пиков)."""
    if target_lufs is not None and x.shape[1] >= sr // 2:
        import pyloudnorm

        meter = pyloudnorm.Meter(sr)
        for _ in range(2):  # второй проход компенсирует то, что срезал лимитер
            loud = meter.integrated_loudness(x.T.astype(np.float64))
            if not np.isfinite(loud):
                break
            x = (x * 10 ** ((target_lufs - loud) / 20)).astype(np.float32)
            if ceiling_db is not None:
                x = limiter(x, sr, ceiling_db)
            if abs(target_lufs - loud) < 0.3:
                break
    elif ceiling_db is not None:
        x = limiter(x, sr, ceiling_db)
    return x


def studio_chain(x: np.ndarray, sr: int, st: dict) -> np.ndarray:
    """Цепочка после монтажа: гул → срез низов → эхо → гейт → де-эссер → левелер → компрессор."""
    hum = st.get("dehum") or 0
    if hum == "auto":
        hum = detect_hum(x, sr)
    if hum:
        x = dehum(x, sr, float(hum))
    if st.get("highpass"):
        x = highpass(x, sr, float(st["highpass"]))
    if st.get("dereverb"):
        s = float(st["dereverb"])
        x = dereverb(x, sr, s, floor_db=-6 - 12 * s)
    if st.get("gate"):
        x = gate(x, sr, below_db=float(st["gate"]), hold=0.04, lookahead=0.015)
    if st.get("deess"):
        x = deess(x, sr, float(st["deess"]))
    if st.get("leveler"):
        x = leveler(x, sr, float(st["leveler"]))
    if st.get("compress"):
        x = compress(x, sr, float(st["compress"]))
    return x
