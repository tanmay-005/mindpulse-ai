"""Language-mix tagging for reviews (step 1.4).

Each word is tagged as Devanagari Hindi, Roman-script Hindi (if it is in
hindi_words.txt) or English; then the whole review gets a label:

    'english' | 'hinglish' | 'hindi_dev' (mostly Devanagari) | 'other' (no words)

Devanagari is NOT converted to Roman script: our encoder (hing-roberta-mixed)
was trained on both scripts, so converting would only lose information.
"""
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

WORDS_PATH = Path(__file__).resolve().parent / "hindi_words.txt"

# a "token" is a run of English letters OR a run of Devanagari letters
TOKEN_RE = re.compile(r"[A-Za-z]+|[\u0900-\u097F]+")
DEVANAGARI_RE = re.compile(r"^[\u0900-\u097F]+$")


@lru_cache(maxsize=1)
def roman_hindi():
    lines = WORDS_PATH.read_text(encoding="utf-8").splitlines()
    return frozenset(w.strip().lower() for w in lines if w.strip() and not w.startswith("#"))


@dataclass(frozen=True)
class LangMix:
    label: str
    code_mix_index: float   # share of Hindi words (Roman or Devanagari), 0 to 1
    n_tokens: int


def tag_token(token):
    """'hi_dev' (Devanagari), 'hi_rom' (Roman-script Hindi) or 'en'."""
    if DEVANAGARI_RE.match(token):
        return "hi_dev"
    if token.lower() in roman_hindi():
        return "hi_rom"
    return "en"


def lang_mix(text):
    tokens = TOKEN_RE.findall(text or "")
    if not tokens:                       # empty or emoji-only
        return LangMix("other", 0.0, 0)

    tags = [tag_token(t) for t in tokens]
    n = len(tags)
    n_dev = tags.count("hi_dev")
    n_rom = tags.count("hi_rom")
    cmi = (n_dev + n_rom) / n

    if n_dev / n >= 0.5:
        label = "hindi_dev"
    elif n_rom >= 2 or (n_rom >= 1 and n <= 4) or cmi >= 0.2:
        label = "hinglish"
    else:
        label = "english"
    return LangMix(label, round(cmi, 3), n)