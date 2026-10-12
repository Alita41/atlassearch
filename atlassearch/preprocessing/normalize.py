from __future__ import annotations
import re
import unicodedata

_ARABIC_VARIANTS = str.maketrans({
    "ي": "ی", "ى": "ی", "ك": "ک", "ۀ": "ہ", "ة": "ہ",
})

ROMAN_VARIANTS = {
    "hai": "he", "hain": "hen", "hy": "he",
    "kya": "kia", "kyun": "kiun", "mein": "men", "main": "men",
}

_URL = re.compile(r"https?://\S+|www\.\S+", re.I)
_EMAIL = re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b")
_PHONE = re.compile(r"(?<!\w)(?:\+?\d[\d\s-]{7,}\d)(?!\w)")
_USER = re.compile(r"(?<!\w)@[A-Za-z0-9_]{2,}")
_PUNCT = re.compile(r"[^\w\u0600-\u06FF]+", re.UNICODE)
_REPEAT = re.compile(r"([A-Za-z])\1{2,}")


def normalize_text(text: str, roman_map: dict[str, str] | None = None) -> str:
    text = unicodedata.normalize("NFKC", text or "")
    text = text.translate(_ARABIC_VARIANTS)
    text = _URL.sub(" <URL> ", text)
    text = _EMAIL.sub(" <EMAIL> ", text)
    text = _PHONE.sub(" <PHONE> ", text)
    text = _USER.sub(" <USER> ", text)
    text = text.lower()
    text = _REPEAT.sub(r"\1\1", text)
    text = _PUNCT.sub(" ", text)
    tokens = text.split()
    mapping = ROMAN_VARIANTS if roman_map is None else roman_map
    tokens = [mapping.get(tok, tok) for tok in tokens]
    return " ".join(tokens)


def detect_script(text: str) -> str:
    urdu = sum("\u0600" <= ch <= "\u06ff" for ch in text)
    latin = sum(("a" <= ch.lower() <= "z") for ch in text)
    if urdu and latin:
        return "mixed"
    if urdu:
        return "urdu_script"
    if latin:
        return "latin"
    return "other"
