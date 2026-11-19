"""Punctuation normalization, pause cue scoring, English abbreviation/apostrophe protection, and word counting."""

from __future__ import annotations

import re

# Sentence-ending strong punctuation
STRONG_PUNCT = set("。？！?!；;")
# Secondary clause punctuation
SECONDARY_PUNCT = set("，,、：:")
# Pairing/enclosing/decorative punctuation
DECORATIVE_PUNCT = set("“”\"'‘’（）()《》〈〉【】[]「」『』—~～·")

ALL_PUNCT_CHARS = STRONG_PUNCT | SECONDARY_PUNCT | DECORATIVE_PUNCT | set(".…-—")

_ABBREV_WORDS = {
    "dr", "mr", "mrs", "ms", "prof", "sr", "jr", "st", "vs", "etc",
    "inc", "ltd", "corp", "fig", "no", "vol", "approx",
    "u.s", "u.k", "u.n", "e.g", "i.e", "a.m", "p.m",
}

# Patterns for protected characters that must NOT be removed even when stripping punctuation
_DECIMAL_DOT_RE = re.compile(r"(?<=\d)\.(?=\d)")
_THOUSANDS_COMMA_RE = re.compile(r"(?<=\d),(?=\d{3}(?!\d))")
_ALPHA_HYPHEN_RE = re.compile(r"(?<=[A-Za-z0-9])[-–](?=[A-Za-z0-9])")
_ALPHA_DOT_RE = re.compile(r"(?<=[A-Za-z])\.(?=[A-Za-z])")
_APOSTROPHE_INNER_RE = re.compile(r"(?<=[A-Za-z])['’](?=[A-Za-z])")
_ABBREV_DOT_RE = re.compile(
    r"\b(?:Dr|Mr|Mrs|Ms|Prof|Sr|Jr|St|vs|etc|Inc|Ltd|Corp|Fig|No|Vol|Approx|U\.S|U\.K|U\.N|e\.g|i\.e|A\.M|P\.M)\.(?=\W|$|[A-Za-z])",
    re.IGNORECASE,
)
_WHITESPACE_RE = re.compile(r"\s+")

_PROTECT_DECIMAL = "\uE001"
_PROTECT_COMMA = "\uE002"
_PROTECT_HYPHEN = "\uE003"
_PROTECT_ADOT = "\uE004"
_PROTECT_APOS = "\uE005"
_PROTECT_ABBR_DOT = "\uE006"


def normalize_ellipsis_and_dashes(text: str) -> str:
    """Normalize multi-character punctuation sequences before analysis."""
    text = text.replace("......", "……").replace("...", "……")
    text = text.replace("--", "——")
    return text


def is_abbreviation_dot(text: str, idx: int) -> bool:
    """Return True if period at `idx` terminates an English abbreviation like 'Dr.' or 'U.S.'."""
    if idx < 0 or idx >= len(text) or text[idx] != ".":
        return False
    # Scan backwards to start of token
    j = idx - 1
    while j >= 0 and (text[j].isalpha() or text[j] == "."):
        j -= 1
    tok = text[j + 1 : idx].lower()
    if tok in _ABBREV_WORDS:
        return True
    # Initialism ending with single letter after a dot, e.g. 'U.S.' or 'J.K.'
    if "." in tok and len(tok.split(".")[-1]) == 1 and tok.split(".")[-1].isalpha():
        return True
    return False


def is_protected_punct(text: str, idx: int) -> bool:
    """Return True if the punctuation character at index `idx` carries semantic/numeric/lexical value."""
    ch = text[idx]
    prev_ch = text[idx - 1] if idx > 0 else ""
    next_ch = text[idx + 1] if idx + 1 < len(text) else ""

    if ch == "." and prev_ch.isdigit() and next_ch.isdigit():
        return True
    if ch == "," and prev_ch.isdigit() and next_ch.isdigit():
        after = text[idx + 1 : idx + 4]
        if len(after) == 3 and after.isdigit() and (idx + 4 >= len(text) or not text[idx + 4].isdigit()):
            return True
    if ch in {"-", "–"} and prev_ch.isalnum() and next_ch.isalnum():
        if prev_ch.isascii() and next_ch.isascii():
            return True
    if ch in {"'", "’"} and prev_ch.isascii() and prev_ch.isalpha() and next_ch.isascii() and next_ch.isalpha():
        return True
    if ch == "." and prev_ch.isascii() and prev_ch.isalpha() and next_ch.isascii() and next_ch.isalpha():
        return True
    if is_abbreviation_dot(text, idx):
        return True
    if ch in {"+", "%", "‰", "℃", "°", "$", "¥", "￥", "#", "@", "&", "/"}:
        return True
    return False


def strip_display_punctuation(text: str, keep_inner_ellipsis: bool = False) -> str:
    """Remove standard subtitle punctuation while preserving decimals, hyphens, apostrophes, and abbrevs."""
    lines = text.splitlines()
    cleaned_lines: list[str] = []

    for line in lines:
        s = line.strip()
        if not s:
            continue

        # Protect meaningful dots/commas/hyphens/apostrophes/abbreviations
        s = _DECIMAL_DOT_RE.sub(_PROTECT_DECIMAL, s)
        s = _THOUSANDS_COMMA_RE.sub(_PROTECT_COMMA, s)
        s = _ALPHA_HYPHEN_RE.sub(_PROTECT_HYPHEN, s)
        s = _APOSTROPHE_INNER_RE.sub(_PROTECT_APOS, s)
        s = _ABBREV_DOT_RE.sub(lambda m: m.group(0)[:-1] + _PROTECT_ABBR_DOT, s)
        s = _ALPHA_DOT_RE.sub(_PROTECT_ADOT, s)

        if not keep_inner_ellipsis:
            s = s.replace("……", " ").replace("...", " ").replace("——", " ")

        out_chars: list[str] = []
        for ch in s:
            if ch in STRONG_PUNCT or ch in SECONDARY_PUNCT or ch in DECORATIVE_PUNCT or ch in {".", "…", "-", "—"}:
                out_chars.append(" ")
            else:
                out_chars.append(ch)
        merged = "".join(out_chars)
        merged = _WHITESPACE_RE.sub(" ", merged).strip()

        merged = re.sub(r"(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])", "", merged)
        merged = re.sub(r"(?<=[\u4e00-\u9fff])\s+(?=[A-Za-z0-9])", "", merged)
        merged = re.sub(r"(?<=[A-Za-z0-9])\s+(?=[\u4e00-\u9fff])", "", merged)

        merged = (
            merged.replace(_PROTECT_DECIMAL, ".")
            .replace(_PROTECT_COMMA, ",")
            .replace(_PROTECT_HYPHEN, "-")
            .replace(_PROTECT_ADOT, ".")
            .replace(_PROTECT_APOS, "'")
            .replace(_PROTECT_ABBR_DOT, ".")
        )
        if merged:
            cleaned_lines.append(merged)

    return "\n".join(cleaned_lines)


def extract_canonical_spoken_sequence(text: str) -> str:
    """Extract spoken content sequence (letters/digits/CJK/meaningful symbols) for strict text integrity verification."""
    stripped = strip_display_punctuation(text, keep_inner_ellipsis=False)
    chars = []
    for ch in stripped:
        if ch.isspace():
            continue
        if ch in ALL_PUNCT_CHARS:
            continue
        chars.append(ch)
    return "".join(chars)


def classify_trailing_punctuation_strength(punct_seq: str) -> float:
    """Return a semantic break bonus weight based on punctuation at a candidate split point."""
    if not punct_seq:
        return 0.0
    if any(c in STRONG_PUNCT for c in punct_seq) or "……" in punct_seq or "——" in punct_seq:
        return 1.0
    if "." in punct_seq:
        return 1.0
    if any(c in SECONDARY_PUNCT for c in punct_seq):
        return 0.65
    if any(c in DECORATIVE_PUNCT for c in punct_seq):
        return 0.2
    return 0.0


def count_english_words(text: str) -> int:
    """Count complete English/alphanumeric words in a line or text (preserving contractions like don't / you're)."""
    if not text or not text.strip():
        return 0
    cnt = 0
    for raw_tok in text.strip().split():
        # Strip surrounding punctuation while keeping inner apostrophes/hyphens/dots
        trimmed = raw_tok.strip(".,?!:;\"'“”‘’()[]{}—–-…~")
        if any(c.isalnum() for c in trimmed):
            cnt += 1
    return cnt


def detect_text_language(text: str) -> str:
    """Detect whether text is primarily Chinese ('zh') or English ('en')."""
    if not text:
        return "zh"
    cjk = sum(1 for c in text if "\u4e00" <= c <= "\u9fff")
    latin = sum(1 for c in text if c.isascii() and c.isalpha())
    if cjk == 0 and latin > 0:
        return "en"
    if latin > max(6, cjk * 3):
        return "en"
    return "zh"
