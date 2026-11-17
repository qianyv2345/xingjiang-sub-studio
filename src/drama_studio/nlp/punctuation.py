"""Punctuation normalization, pause cue scoring, and safe punctuation removal."""

from __future__ import annotations

import re

# Sentence-ending strong punctuation
STRONG_PUNCT = set("。？！?!；;")
# Secondary clause punctuation
SECONDARY_PUNCT = set("，,、：:")
# Pairing/enclosing/decorative punctuation
DECORATIVE_PUNCT = set("“”\"'‘’（）()《》〈〉【】[]「」『』—~～·")

ALL_PUNCT_CHARS = STRONG_PUNCT | SECONDARY_PUNCT | DECORATIVE_PUNCT | set(".…-—")

# Patterns for protected characters that must NOT be removed even when stripping punctuation
_DECIMAL_DOT_RE = re.compile(r"(?<=\d)\.(?=\d)")
_THOUSANDS_COMMA_RE = re.compile(r"(?<=\d),(?=\d{3}(?!\d))")
_ALPHA_HYPHEN_RE = re.compile(r"(?<=[A-Za-z0-9])[-–](?=[A-Za-z0-9])")
_ALPHA_DOT_RE = re.compile(r"(?<=[A-Za-z])\.(?=[A-Za-z])")
_WHITESPACE_RE = re.compile(r"\s+")

_PROTECT_DECIMAL = "\uE001"
_PROTECT_COMMA = "\uE002"
_PROTECT_HYPHEN = "\uE003"
_PROTECT_ADOT = "\uE004"


def normalize_ellipsis_and_dashes(text: str) -> str:
    """Normalize multi-character punctuation sequences before analysis."""
    text = text.replace("......", "……").replace("...", "……")
    text = text.replace("--", "——")
    return text


def is_protected_punct(text: str, idx: int) -> bool:
    """Return True if the punctuation character at index `idx` carries semantic/numeric value."""
    ch = text[idx]
    prev_ch = text[idx - 1] if idx > 0 else ""
    next_ch = text[idx + 1] if idx + 1 < len(text) else ""

    if ch == "." and prev_ch.isdigit() and next_ch.isdigit():
        return True
    if ch == "," and prev_ch.isdigit() and next_ch.isdigit():
        # Check thousands separator style e.g. 1,000
        after = text[idx + 1 : idx + 4]
        if len(after) == 3 and after.isdigit() and (idx + 4 >= len(text) or not text[idx + 4].isdigit()):
            return True
    if ch in {"-", "–"} and prev_ch.isalnum() and next_ch.isalnum():
        # Preserve ASCII/alphanumeric compound words like T-shirt, AI-2026, e-mail
        if prev_ch.isascii() and next_ch.isascii():
            return True
    if ch == "." and prev_ch.isascii() and prev_ch.isalpha() and next_ch.isascii() and next_ch.isalpha():
        return True
    if ch in {"+", "%", "‰", "℃", "°", "$", "¥", "￥", "#", "@", "&", "/"}:
        return True
    return False


def strip_display_punctuation(text: str, keep_inner_ellipsis: bool = False) -> str:
    """Remove standard subtitle punctuation while preserving decimals, alphanumeric hyphens, and all spoken words.

    Line breaks (`\\n`) inside `text` are preserved cleanly per line.
    """
    lines = text.splitlines()
    cleaned_lines: list[str] = []

    for line in lines:
        s = line.strip()
        if not s:
            continue

        # Protect meaningful dots/commas/hyphens
        s = _DECIMAL_DOT_RE.sub(_PROTECT_DECIMAL, s)
        s = _THOUSANDS_COMMA_RE.sub(_PROTECT_COMMA, s)
        s = _ALPHA_HYPHEN_RE.sub(_PROTECT_HYPHEN, s)
        s = _ALPHA_DOT_RE.sub(_PROTECT_ADOT, s)

        if not keep_inner_ellipsis:
            s = s.replace("……", " ").replace("...", " ").replace("——", " ")

        # Replace standard punctuation with space between words (keeps English words separated),
        # then collapse excessive spaces around CJK characters.
        out_chars: list[str] = []
        for ch in s:
            if ch in STRONG_PUNCT or ch in SECONDARY_PUNCT or ch in DECORATIVE_PUNCT or ch in {".", "…", "-", "—"}:
                out_chars.append(" ")
            else:
                out_chars.append(ch)
        merged = "".join(out_chars)
        merged = _WHITESPACE_RE.sub(" ", merged).strip()

        # Remove spaces inserted between CJK characters, while preserving space between Latin/ASCII words
        merged = re.sub(r"(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])", "", merged)
        merged = re.sub(r"(?<=[\u4e00-\u9fff])\s+(?=[A-Za-z0-9])", "", merged)
        merged = re.sub(r"(?<=[A-Za-z0-9])\s+(?=[\u4e00-\u9fff])", "", merged)

        # Restore protected semantic punctuation
        merged = (
            merged.replace(_PROTECT_DECIMAL, ".")
            .replace(_PROTECT_COMMA, ",")
            .replace(_PROTECT_HYPHEN, "-")
            .replace(_PROTECT_ADOT, ".")
        )
        if merged:
            cleaned_lines.append(merged)

    return "\n".join(cleaned_lines)


def extract_canonical_spoken_sequence(text: str) -> str:
    """Extract spoken content sequence (letters/digits/CJK/meaningful symbols) for strict text integrity verification."""
    stripped = strip_display_punctuation(text, keep_inner_ellipsis=False)
    # Remove all whitespace and any remaining punctuation marks for pure character integrity comparison
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
    if any(c in SECONDARY_PUNCT for c in punct_seq):
        return 0.65
    if any(c in DECORATIVE_PUNCT for c in punct_seq):
        return 0.2
    return 0.0
