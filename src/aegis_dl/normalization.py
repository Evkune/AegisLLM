import re
import string
import unicodedata

# Zero-width & non-printing unicode characters
ZERO_WIDTH_PATTERN = re.compile(r"[\u200B-\u200D\uFEFF\u00AD\u2060\u180E]")
WHITESPACE_COLLAPSE = re.compile(r"\s+")
PUNCT_DIGIT_PATTERN = re.compile(r"[" + re.escape(string.punctuation + string.digits) + r"]")


def normalize_text(text: str) -> str:
    """Canonical pre-normalization applying NFKC and stripping non-printing codepoints."""
    if not text:
        return ""
    cleaned = ZERO_WIDTH_PATTERN.sub("", text)
    cleaned = unicodedata.normalize("NFKC", cleaned)
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
