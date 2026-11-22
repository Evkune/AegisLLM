import re
import string
import unicodedata

from aegis.security.canonicalizer import (
    _extract_plane14_tags,
    _normalize_mixed_script_confusables,
)

ZERO_WIDTH_PATTERN = re.compile(r"[\u200B-\u200D\uFEFF\u00AD\u2060\u180E\u202A-\u202E\u2066-\u2069]")
WHITESPACE_COLLAPSE = re.compile(r"\s+")
PUNCT_DIGIT_PATTERN = re.compile(r"[" + re.escape(string.punctuation + string.digits) + r"]")


def normalize_text(text: str) -> str:
    """
    Canonical pre-normalization sharing the production canonicalizer primitives
    (Plane-14 tag stripping, Bidi/zero-width removal, NFKC, mixed-script confusables).
    """
    if not text:
        return ""
    cleaned, _, _ = _extract_plane14_tags(text)
    cleaned = ZERO_WIDTH_PATTERN.sub("", cleaned)
    cleaned = "".join(ch for ch in cleaned if unicodedata.category(ch) != "Cf")
    cleaned = unicodedata.normalize("NFKC", cleaned)
    cleaned, _ = _normalize_mixed_script_confusables(cleaned)
    cleaned = WHITESPACE_COLLAPSE.sub(" ", cleaned).strip()
    return cleaned


def extract_structural_skeleton(text: str) -> str:
    """
    Extracts structural invariant skeleton:
    lowercased, stripped of punctuation, digits, and collapsed whitespace.
    Used for invariant group hashing and MinHash LSH.
    """
    normalized = normalize_text(text).lower()
    skeleton = PUNCT_DIGIT_PATTERN.sub(" ", normalized)
    skeleton = WHITESPACE_COLLAPSE.sub(" ", skeleton).strip()
    return skeleton
