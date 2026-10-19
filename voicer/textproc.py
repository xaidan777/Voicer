"""Работа с текстом: нормализация, слова-паразиты, повторы и фальстарты."""
from __future__ import annotations

import re

_PUNCT = "\"'«»„“”.,!?;:()[]{}…—–"
SENTENCE_END = re.compile(r"[.!?…]+[\"'»”)]*$")

# Однозначный мусор (междометия-заминки): режется автоматически. Проверяется слово без дефисов:
# «э-э-э» → «эээ», «м-м» → «мм», «ну-у-у» → «нууу».
STRICT = re.compile(r"^(э+|э+м+|м+|х+м+|а{2,}|о{2,}|у{2,}|ы{2,}|нуу+|"
                    r"u+h+|u+m+|u+h+m+|e+r+|e+r+m+|a+h+|h+m+|m+h+m+)$")
# Возможный мусор: зависит от контекста, решает LLM или пользователь.
SOFT_WORDS = {
    "ru": {"ну", "вот", "типа", "короче", "значит", "собственно", "слушай", "блин",
           "получается", "скажем", "кстати", "вообще", "вообще-то"},
    "en": {"like", "basically", "actually", "literally", "so", "well", "right", "okay"},
}
SOFT_PHRASES = {
    "ru": [("как", "бы"), ("так", "сказать"), ("в", "общем"), ("в", "общем-то"),
           ("это", "самое"), ("ну", "вот"), ("как", "говорится"), ("в", "принципе")],
    "en": [("you", "know"), ("i", "mean"), ("kind", "of"), ("sort", "of")],
}

# Подсказка для Whisper: заставляет его записывать «э-э», «мм» вместо того, чтобы выкидывать.
FILLER_PROMPTS = {
    "ru": "Э-э, ну, это... Мм, как бы, вот. Значит, э-э, короче, мы мы начали.",
    "en": "Um, uh, so, like... Hmm, you know, I mean, er, we we started.",
}


def norm(word: str) -> str:
    w = word.lower().replace("ё", "е").strip(_PUNCT + " ")
    return w.strip("-")


def is_truncated(word: str) -> bool:
    w = word.strip("\"'«»„“”,;:() ")
    return w.endswith("-") or w.endswith("…") or w.endswith("...")


def lang_key(lang: str | None) -> str:
    return "en" if (lang or "").startswith("en") else "ru"


def is_strict_filler(word: str, lang: str | None = None) -> bool:
    w = norm(word)
    flat = w.replace("-", "")
    if not flat:
        return False
    if STRICT.match(flat):
        return True
    # Растянутая буква через дефис: «а-а-а», «о-о».
    return "-" in w and flat.isalpha() and len(set(flat)) == 1


def soft_fillers(words: list[str], lang: str | None) -> set[int]:
    """Позиции (в переданном списке) возможных слов-паразитов."""
    k = lang_key(lang)
    n = [norm(w) for w in words]
    out = {i for i, w in enumerate(n) if w in SOFT_WORDS[k]}
    for ph in SOFT_PHRASES[k]:
        L = len(ph)
        for i in range(len(n) - L + 1):
            if tuple(n[i:i + L]) == ph:
                out.update(range(i, i + L))
    return out


def find_repeats(words: list[str], max_len: int = 6) -> list[tuple[int, int, str]]:
    """Ищет повторы и фальстарты. Возвращает (начало, конец_не_включ., тип) — что вырезать.

    Оставляется последний вариант: «мы мы пошли» → режем первое «мы»;
    «программа должна, программа должна убирать» → режем первую пару;
    «мы пошли в ма- мы пошли в магазин» → режем первую попытку целиком.
    """
    # Whisper дробит «5.5» на «5» и «.5» — дробная часть не повтор целой.
    n = ["." + norm(w) if re.match(r"^[.,]\d", w.strip()) else norm(w) for w in words]
    trunc = [is_truncated(w) for w in words]
    res: list[tuple[int, int, str]] = []
    i = 0
    N = len(n)
    while i < N:
        found = False
        for L in range(min(max_len, (N - i) // 2), 0, -1):
            a, b = n[i:i + L], n[i + L:i + 2 * L]
            if not all(a):
                continue
            if a == b:
                res.append((i, i + L, "repeat"))
                i += L
                found = True
                break
            # Фальстарт: совпадает всё, кроме последнего слова, которое оборвано.
            if a[:-1] == b[:-1] and b[-1].startswith(a[-1]) and len(a[-1]) < len(b[-1]):
                if trunc[i + L - 1] or (L >= 2 and len(a[-1]) >= 2):
                    res.append((i, i + L, "falsestart"))
                    i += L
                    found = True
                    break
        if not found:
            # Оборванное слово, которое тут же сказано целиком: «ма- магазин».
            if i + 1 < N and trunc[i] and n[i] and n[i + 1].startswith(n[i]) and n[i] != n[i + 1]:
                res.append((i, i + 1, "falsestart"))
            i += 1
    return res


def tokenize_text(text: str) -> list[str]:
    """Разбивка на слова; одиночные знаки препинания приклеиваются к предыдущему слову."""
    out: list[str] = []
    for line in text.splitlines():
        for w in line.split():
            if out and all(ch in _PUNCT + "-" for ch in w):
                out[-1] += ("" if w in ",.!?;:…" else " ") + w
            else:
                out.append(w)
    return out


_TR = dict(zip("абвгдеёжзийклмнопрстуфхцчшщъыьэюя",
               ["a", "b", "v", "g", "d", "e", "e", "zh", "z", "i", "y", "k", "l", "m", "n", "o", "p",
                "r", "s", "t", "u", "f", "h", "ts", "ch", "sh", "sch", "", "y", "", "e", "yu", "ya"]))


def slugify(name: str) -> str:
    s = "".join(_TR.get(ch, ch) for ch in name.lower())
    s = re.sub(r"[^a-z0-9]+", "-", s).strip("-")
    return s[:40] or "project"
