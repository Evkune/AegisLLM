from __future__ import annotations

import base64
import binascii
import codecs
import re
import time
import unicodedata
from dataclasses import dataclass, field
from urllib.parse import unquote

# Plane-14 Tag characters (U+E0000..U+E007F) used in Unicode Tag Smuggling
_TAG_START = 0xE0000
_TAG_END = 0xE007F
_BLACK_FLAG = "\U0001F3F4"

# Explicit bidirectional overrides & isolates (CVE-2021-42574 Trojan Source)
_BIDI_DANGEROUS: set[str] = {
    "\u202A",  # LRE
    "\u202B",  # RLE
    "\u202C",  # PDF
    "\u202D",  # LRO
    "\u202E",  # RLO
    "\u2066",  # LRI
    "\u2067",  # RLI
    "\u2068",  # FSI
    "\u2069",  # PDI
}
_BIDI_MARKS: set[str] = {"\u200E", "\u200F", "\u061C"}

_ZERO_WIDTH_RE = re.compile(r"[\u200B-\u200D\u2060\uFEFF\u00AD\u180E\uFFF9-\uFFFB]")
_SPACED_LETTERS_RE = re.compile(r"\b(?:[A-Za-z][ \t]){3,}[A-Za-z]\b")
_INTERLEAVED_WORD_RE = re.compile(r"\b[A-Za-z](?:[._\-|/*~][A-Za-z]){3,}\b")
_HORIZONTAL_WS_RE = re.compile(r"[ \t]+")
_MULTI_NEWLINE_RE = re.compile(r"\n{3,}")

# False-positive shields: UUID, SHA/MD5 hashes, and valid 3-part JWTs
_UUID_RE = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)
_PURE_HEX_HASH_RE = re.compile(r"^[0-9a-fA-F]{32}$|^[0-9a-fA-F]{40}$|^[0-9a-fA-F]{64}$|^[0-9a-fA-F]{128}$")
_JWT_RE = re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b")

_BASE64_TOKEN_RE = re.compile(r"(?<![A-Za-z0-9+/_=-])([A-Za-z0-9+/_-]{16,1024}={0,2})(?![A-Za-z0-9+/_=-])")
_BASE64_FULL_RE = re.compile(r"^[A-Za-z0-9+/_-]{16,1024}={0,2}$")
_HEX_ESCAPE_SEQ_RE = re.compile(r"(?:\\x[0-9a-fA-F]{2}){8,}")
_HEX_RAW_TOKEN_RE = re.compile(r"(?<![0-9a-fA-F])(?:0x)?([0-9a-fA-F]{20,1024})(?![0-9a-fA-F])")
_URL_ENC_BYTE_RE = re.compile(r"%[0-9a-fA-F]{2}")
_ROT13_HINT_RE = re.compile(r"\b(?:rot13|rot-13|caesar)\b", re.IGNORECASE)

# Targeted UTR#39 Latin skeleton confusables (Cyrillic & Greek lookalikes)
_CONFUSABLE_TO_LATIN: dict[str, str] = {
    # Cyrillic lowercase
    "\u0430": "a",  # а
    "\u0435": "e",  # е
    "\u043E": "o",  # о
    "\u0440": "p",  # р
    "\u0441": "c",  # с
    "\u0443": "y",  # у
    "\u0445": "x",  # х
    "\u0456": "i",  # і
    "\u0458": "j",  # ј
    "\u0455": "s",  # ѕ
    "\u043A": "k",  # к
    "\u043C": "m",  # м
    "\u043D": "h",  # н
    "\u0442": "t",  # т
    "\u0432": "b",  # в
    "\u044A": "b",  # ъ
    "\u0433": "r",  # г
    # Cyrillic uppercase
    "\u0410": "A",  # А
    "\u0412": "B",  # В
    "\u0421": "C",  # С
    "\u0415": "E",  # Е
    "\u041D": "H",  # Н
    "\u0406": "I",  # І
    "\u0408": "J",  # Ј
    "\u041A": "K",  # К
    "\u041C": "M",  # М
    "\u041E": "O",  # О
    "\u0420": "P",  # Р
    "\u0405": "S",  # Ѕ
    "\u0422": "T",  # Т
    "\u0425": "X",  # Х
    "\u0423": "Y",  # У
    # Greek lowercase
    "\u03B1": "a",  # α
    "\u03B5": "e",  # ε
    "\u03B9": "i",  # ι
    "\u03BF": "o",  # ο
    "\u03C1": "p",  # ρ
    "\u03C4": "t",  # τ
    "\u03C5": "u",  # υ
    "\u03C7": "x",  # χ
    "\u03BA": "k",  # κ
    "\u03BD": "v",  # ν
    # Greek uppercase
    "\u0391": "A",  # Α
    "\u0392": "B",  # Β
    "\u0395": "E",  # Ε
    "\u0396": "Z",  # Ζ
    "\u0397": "H",  # Η
    "\u0399": "I",  # Ι
    "\u039A": "K",  # Κ
    "\u039C": "M",  # Μ
    "\u039D": "N",  # Ν
    "\u039F": "O",  # Ο
    "\u03A1": "P",  # Ρ
    "\u03A4": "T",  # Τ
    "\u03A7": "X",  # Χ
    "\u03A5": "Y",  # Υ
    # Latin/extended lookalikes
    "\u0131": "i",  # ı (dotless i)
    "\u0237": "j",  # ȷ
    "\u2010": "-",
    "\u2011": "-",
    "\u2012": "-",
    "\u2013": "-",
    "\u2014": "-",
    "\u2212": "-",
}

_SECURITY_SKELETON_WORDS: set[str] = {
    "ignore",
    "disregard",
    "forget",
    "override",
    "previous",
    "prior",
    "instructions",
    "system",
    "prompt",
    "jailbreak",
    "bypass",
    "developer",
    "reveal",
    "output",
    "secret",
    "admin",
    "assistant",
    "rules",
    "policy",
    "safety",
}

_WORD_TOKEN_RE = re.compile(r"\w+")


@dataclass(slots=True)
class CanonicalResult:
    raw_text: str
    canonical_text: str
    views: list[str] = field(default_factory=list)
    anomalies: list[str] = field(default_factory=list)
    hard_block_anomalies: list[str] = field(default_factory=list)
    latency_ms: float = 0.0


def _is_plausible_human_text(decoded: str, require_space: bool = False) -> bool:
    s = decoded.strip()
    if len(s) < 8:
        return False
    printable = sum(1 for ch in s if ch.isprintable() or ch in "\n\r\t")
    if printable / len(s) < 0.88:
        return False
    alpha = sum(1 for ch in s if ("A" <= ch <= "Z") or ("a" <= ch <= "z"))
    if alpha / len(s) < 0.45:
        return False
    if require_space and (" " not in s and "\n" not in s):
        return False
    # Reject high-entropy random tokens lacking vowel/word structure
    words = [w for w in re.split(r"[^A-Za-z]+", s) if len(w) >= 2]
    if not words:
        return False
    has_vowel_word = any(any(v in w.lower() for v in "aeiouy") for w in words)
    return has_vowel_word


def _extract_plane14_tags(text: str) -> tuple[str, list[str], bool]:
    """
    Separates Plane-14 Tag codepoints (U+E0000..U+E007F) from visible text.
    Legitimate emoji subdivision flag sequences (U+1F3F4 + [a-z0-9]{2,6} + U+E007F)
    are tolerated without triggering a tag-smuggling alert.
    """
    out_chars: list[str] = []
    hidden_payloads: list[str] = []
    smuggling_detected = False

    n = len(text)
    i = 0
    while i < n:
        ch = text[i]
        cp = ord(ch)
        if _TAG_START <= cp <= _TAG_END:
            preceded_by_flag = i > 0 and text[i - 1] == _BLACK_FLAG
            run_chars: list[str] = []
            j = i
            while j < n and _TAG_START <= ord(text[j]) <= _TAG_END:
                tag_cp = ord(text[j]) - _TAG_START
                if 0x20 <= tag_cp <= 0x7E:
                    run_chars.append(chr(tag_cp))
                j += 1
            decoded_run = "".join(run_chars)
            is_valid_flag_seq = (
                preceded_by_flag
                and 2 <= len(decoded_run) <= 6
                and decoded_run.isalnum()
                and decoded_run.islower()
                and ord(text[j - 1]) == 0xE007F
            )
            if not is_valid_flag_seq:
                smuggling_detected = True
                if decoded_run.strip():
                    hidden_payloads.append(decoded_run.strip())
            i = j
        else:
            out_chars.append(ch)
            i += 1

    return "".join(out_chars), hidden_payloads, smuggling_detected


def _normalize_mixed_script_confusables(text: str) -> tuple[str, bool]:
    """
    Applies targeted UTR#39 confusable mapping only to words exhibiting mixed Latin +
    Cyrillic/Greek scripts, or single-script Cyrillic/Greek tokens whose skeleton matches
    a security keyword inside a predominantly Latin prompt.
    """
    has_latin_context = any(("A" <= c <= "Z") or ("a" <= c <= "z") for c in text)
    if not has_latin_context:
        return text, False

    replaced = False

    def _replace_word(match: re.Match[str]) -> str:
        nonlocal replaced
        token = match.group(0)
        has_ascii_latin = any(("A" <= c <= "Z") or ("a" <= c <= "z") for c in token)
        has_confusable = any(c in _CONFUSABLE_TO_LATIN for c in token)
        if not has_confusable:
            return token

        mapped = "".join(_CONFUSABLE_TO_LATIN.get(c, c) for c in token)
        if has_ascii_latin:
            replaced = True
            return mapped
        if mapped.lower() in _SECURITY_SKELETON_WORDS:
            replaced = True
            return mapped
        return token

    normalized = _WORD_TOKEN_RE.sub(_replace_word, text)
    return normalized, replaced


def _is_valid_jwt(token: str) -> bool:
    parts = token.split(".")
    if len(parts) != 3:
        return False
    header_b64 = parts[0]
    pad = "=" * ((4 - len(header_b64) % 4) % 4)
    try:
        decoded = base64.urlsafe_b64decode(header_b64 + pad).decode("utf-8")
    except (binascii.Error, UnicodeDecodeError, ValueError):
        return False
    return decoded.startswith("{") and ('"alg"' in decoded or '"typ"' in decoded)


class Canonicalizer:
    """
    Single Source of Truth pre-tokenization canonicalizer shared between
    the production proxy (`aegis`) and training/data pipelines (`aegis_dl`).
    """

    def __init__(
        self,
        max_input_chars: int = 32768,
        max_decode_depth: int = 2,
        max_views: int = 4,
        max_decoded_bytes: int = 2048,
    ) -> None:
        self.max_input_chars = max_input_chars
        self.max_decode_depth = max_decode_depth
        self.max_views = max_views
        self.max_decoded_bytes = max_decoded_bytes

    def canonicalize(self, text: str) -> CanonicalResult:
        t0 = time.perf_counter()
        if not text:
            return CanonicalResult(raw_text="", canonical_text="", views=[""], latency_ms=0.0)

        bounded = text[: self.max_input_chars]
        anomalies: list[str] = []
        hard_block: list[str] = []
        extra_views: list[str] = []

        # 1. Plane-14 Unicode Tag Smuggling
        cleaned, tag_payloads, tag_smuggling = _extract_plane14_tags(bounded)
        if tag_smuggling:
            anomalies.append("anomaly:unicode_tag_smuggling")
            hard_block.append("anomaly:unicode_tag_smuggling")
            for payload in tag_payloads:
                extra_views.append(payload[: self.max_decoded_bytes])

        # 2. Bidi Trojan Source (explicit overrides / isolates)
        if any(ch in _BIDI_DANGEROUS for ch in cleaned):
            anomalies.append("anomaly:bidi_trojan_source")
            hard_block.append("anomaly:bidi_trojan_source")
            cleaned = "".join(ch for ch in cleaned if ch not in _BIDI_DANGEROUS)
        if any(ch in _BIDI_MARKS for ch in cleaned):
            cleaned = "".join(ch for ch in cleaned if ch not in _BIDI_MARKS)

        # 3. Zero-width & invisible Format (Cf) codepoints
        if _ZERO_WIDTH_RE.search(cleaned):
            anomalies.append("anomaly:zero_width_chars")
            cleaned = _ZERO_WIDTH_RE.sub("", cleaned)
        cleaned = "".join(ch for ch in cleaned if unicodedata.category(ch) != "Cf")

        # 4. NFKC canonicalization (fullwidth, mathematical alphanumeric, compatibility forms)
        normalized = unicodedata.normalize("NFKC", cleaned)

        # 5. Targeted UTR#39 homoglyph folding on mixed-script words
        normalized, had_homoglyphs = _normalize_mixed_script_confusables(normalized)
        if had_homoglyphs:
            anomalies.append("anomaly:mixed_script_homoglyphs")

        # 6. Anti-spacing & anti-interleaving collapse
        if _SPACED_LETTERS_RE.search(normalized):
            anomalies.append("anomaly:spaced_characters")
            temp = re.sub(r"[ \t]{2,}", " _W_ ", normalized)
            temp = re.sub(r"(?<=\b[A-Za-z])[ \t](?=[A-Za-z]\b)", "", temp)
            normalized = temp.replace(" _W_ ", " ")

        if _INTERLEAVED_WORD_RE.search(normalized):
            anomalies.append("anomaly:char_interleaving")
            deinterleaved = _INTERLEAVED_WORD_RE.sub(
                lambda m: re.sub(r"[._\-|/*~]", "", m.group(0)),
                normalized,
            )
            if deinterleaved != normalized:
                extra_views.append(deinterleaved)
                normalized = deinterleaved

        # 7. Whitespace hygiene (preserve single newlines for structure)
        normalized = _HORIZONTAL_WS_RE.sub(" ", normalized)
        normalized = _MULTI_NEWLINE_RE.sub("\n\n", normalized).strip()

        # 8. Bounded multi-view de-obfuscation (with JWT / UUID / SHA-256 FP shields)
        decoded_views, decode_anomalies = self._extract_decoded_views(normalized)
        anomalies.extend(decode_anomalies)
        extra_views.extend(decoded_views)

        # Deduplicate views while preserving order and respecting max_views budget
        views: list[str] = [normalized]
        seen: set[str] = {normalized}
        for view in extra_views:
            v_clean = _HORIZONTAL_WS_RE.sub(" ", unicodedata.normalize("NFKC", view)).strip()
            if v_clean and v_clean not in seen and len(views) < self.max_views:
                views.append(v_clean)
                seen.add(v_clean)

        # Preserve unique anomalies in insertion order
        dedup_anomalies = list(dict.fromkeys(anomalies))
        dedup_hard = list(dict.fromkeys(hard_block))

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return CanonicalResult(
            raw_text=bounded,
            canonical_text=normalized,
            views=views,
            anomalies=dedup_anomalies,
            hard_block_anomalies=dedup_hard,
            latency_ms=round(elapsed_ms, 4),
        )

    def _extract_decoded_views(self, text: str) -> tuple[list[str], list[str]]:
        views: list[str] = []
        anomalies: list[str] = []
        queue: list[tuple[str, int]] = [(text, 0)]
        visited: set[str] = {text}

        while queue and len(views) < (self.max_views - 1):
            current, depth = queue.pop(0)
            if depth >= self.max_decode_depth:
                continue

            # Mask valid JWTs so Base64URL scan never raises false positives on bearer tokens
            jwt_spans: list[tuple[int, int]] = [
                m.span() for m in _JWT_RE.finditer(current) if _is_valid_jwt(m.group(0))
            ]

            def _inside_jwt(
                start: int,
                end: int,
                spans: list[tuple[int, int]] = jwt_spans,
            ) -> bool:
                return any(js <= start and end <= je for js, je in spans)

            # A. Base64 / Base64URL candidates
            for match in _BASE64_TOKEN_RE.finditer(current):
                if _inside_jwt(*match.span()):
                    continue
                cand = match.group(1)
                if _UUID_RE.match(cand) or _PURE_HEX_HASH_RE.match(cand):
                    continue
                decoded = self._try_decode_base64(cand)
                if decoded and decoded not in visited:
                    visited.add(decoded)
                    anomalies.append("anomaly:base64_payload")
                    views.append(decoded)
                    queue.append((decoded, depth + 1))
                    if len(views) >= (self.max_views - 1):
                        break

            if len(views) >= (self.max_views - 1):
                break

            # B. Hex-encoded payloads (\xNN sequences or continuous hex decoding to English)
            for match in _HEX_ESCAPE_SEQ_RE.finditer(current):
                hex_digits = match.group(0).replace("\\x", "")
                decoded = self._try_decode_hex(hex_digits, require_space=False)
                if decoded and decoded not in visited:
                    visited.add(decoded)
                    anomalies.append("anomaly:hex_payload")
                    views.append(decoded)
                    queue.append((decoded, depth + 1))
                    if len(views) >= (self.max_views - 1):
                        break

            for match in _HEX_RAW_TOKEN_RE.finditer(current):
                cand = match.group(1)
                if len(cand) % 2 != 0:
                    continue
                # Strict space requirement prevents SHA-256 / MD5 / Git SHAs from false-triggering
                decoded = self._try_decode_hex(cand, require_space=True)
                if decoded and decoded not in visited:
                    visited.add(decoded)
                    anomalies.append("anomaly:hex_payload")
                    views.append(decoded)
                    queue.append((decoded, depth + 1))
                    if len(views) >= (self.max_views - 1):
                        break

            if len(views) >= (self.max_views - 1):
                break

            # C. URL percent-encoding (>= 3 %XX escapes anywhere in payload)
            if len(_URL_ENC_BYTE_RE.findall(current)) >= 3:
                unquoted = unquote(current)
                if unquoted != current and unquoted not in visited and len(unquoted) <= self.max_input_chars:
                    visited.add(unquoted)
                    anomalies.append("anomaly:url_encoded_payload")
                    views.append(unquoted[: self.max_decoded_bytes])
                    queue.append((unquoted[: self.max_decoded_bytes], depth + 1))

            # D. Explicit ROT13 hint
            if depth == 0 and _ROT13_HINT_RE.search(current) and len(views) < (self.max_views - 1):
                rot_decoded = codecs.decode(current, "rot_13")
                if rot_decoded not in visited:
                    visited.add(rot_decoded)
                    anomalies.append("anomaly:rot13_hint")
                    views.append(rot_decoded[: self.max_decoded_bytes])

        return views, anomalies

    def _raw_decode_base64(self, candidate: str) -> str | None:
        raw = candidate.strip()
        if len(raw) < 16:
            return None
        is_urlsafe = ("-" in raw) or ("_" in raw)
        pad_len = (4 - len(raw) % 4) % 4
        if pad_len == 3:
            return None
        padded = raw + ("=" * pad_len)
        try:
            decoded_bytes = (
                base64.urlsafe_b64decode(padded)
                if is_urlsafe
                else base64.b64decode(padded, validate=True)
            )
        except (binascii.Error, ValueError):
            return None
        if len(decoded_bytes) > self.max_decoded_bytes:
            decoded_bytes = decoded_bytes[: self.max_decoded_bytes]
        try:
            return decoded_bytes.decode("utf-8").strip()
        except UnicodeDecodeError:
            return None

    def _try_decode_base64(self, candidate: str) -> str | None:
        raw = candidate.strip()
        decoded_str = self._raw_decode_base64(raw)
        if decoded_str is None:
            return None
        is_urlsafe = ("-" in raw) or ("_" in raw)
        # Support 2-layer nested Base64 (e.g. base64(base64(payload))) without FP on random keys
        if _BASE64_FULL_RE.match(decoded_str) and not _PURE_HEX_HASH_RE.match(decoded_str):
            inner = self._raw_decode_base64(decoded_str)
            if inner is not None and _is_plausible_human_text(inner, require_space=True):
                return inner
        require_space = is_urlsafe or (not raw.endswith("="))
        if not _is_plausible_human_text(decoded_str, require_space=require_space):
            return None
        return decoded_str

    def _try_decode_hex(self, hex_digits: str, require_space: bool) -> str | None:
        try:
            raw_bytes = bytes.fromhex(hex_digits[: self.max_decoded_bytes * 2])
        except ValueError:
            return None
        try:
            decoded_str = raw_bytes.decode("utf-8").strip()
        except UnicodeDecodeError:
            return None
        if not _is_plausible_human_text(decoded_str, require_space=require_space):
            return None
        return decoded_str


_DEFAULT_CANONICALIZER = Canonicalizer()


def canonicalize_text(text: str) -> CanonicalResult:
    return _DEFAULT_CANONICALIZER.canonicalize(text)
