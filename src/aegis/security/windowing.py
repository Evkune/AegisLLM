from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class ExtractedSegment:
    role: str
    text: str
    turn_index: int


@dataclass(slots=True)
class TextWindow:
    text: str
    source_role: str
    window_kind: str  # "single", "head", "tail", "hotspot", "decoded_view"
    char_start: int
    char_end: int
    approx_tokens: int


@dataclass(slots=True)
class WindowPlan:
    windows: list[TextWindow] = field(default_factory=list)
    latency_ms: float = 0.0


def _flatten_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, Mapping):
                if item.get("type") in ("text", "input_text") and isinstance(item.get("text"), str):
                    parts.append(item["text"])
                elif isinstance(item.get("content"), str):
                    parts.append(item["content"])
        return "\n".join(parts)
    return ""


def extract_openai_segments(
    payload: Mapping[str, Any],
    last_k_user_turns: int = 3,
    inspect_system: bool = False,
) -> list[ExtractedSegment]:
    """
    Role-aware extractor for OpenAI `/v1/chat/completions` and `/v1/completions` payloads.
    Extracts untrusted surfaces: all `tool`/`function` outputs (indirect prompt injection)
    and the last `k` `user` turns (multi-turn crescendo & direct injection).
    """
    segments: list[ExtractedSegment] = []

    if "messages" in payload and isinstance(payload["messages"], list):
        messages = payload["messages"]
        user_turns: list[ExtractedSegment] = []
        for idx, msg in enumerate(messages):
            if not isinstance(msg, Mapping):
                continue
            role = str(msg.get("role", "user")).lower()
            text = _flatten_content(msg.get("content")).strip()
            if not text:
                continue
            if role in ("tool", "function"):
                segments.append(ExtractedSegment(role=role, text=text, turn_index=idx))
            elif role == "user":
                user_turns.append(ExtractedSegment(role="user", text=text, turn_index=idx))
            elif inspect_system and role in ("system", "developer"):
                segments.append(ExtractedSegment(role=role, text=text, turn_index=idx))

        if last_k_user_turns > 0:
            segments.extend(user_turns[-last_k_user_turns:])
        segments.sort(key=lambda s: s.turn_index)
        return segments

    if "prompt" in payload:
        prompt_val = payload["prompt"]
        if isinstance(prompt_val, str) and prompt_val.strip():
            return [ExtractedSegment(role="user", text=prompt_val.strip(), turn_index=0)]
        if isinstance(prompt_val, list):
            for idx, item in enumerate(prompt_val):
                if isinstance(item, str) and item.strip():
                    segments.append(ExtractedSegment(role="user", text=item.strip(), turn_index=idx))
            return segments

    return segments


class WindowBuilder:
    """
    Adaptive sequence windowing enforcing `L <= max_window_tokens` (256) without padding
    short prompts up to 256. Short prompts emit 1 single window (preserving `< 15 ms` P99
    on `L <= 64`), while long RAG documents emit `Head + Tail + Top-2 Hot-Spots`.
    """

    def __init__(
        self,
        max_window_tokens: int = 256,
        max_windows_per_segment: int = 4,
        approx_chars_per_token: float = 3.8,
        token_counter: Callable[[str], int] | None = None,
    ) -> None:
        self.max_window_tokens = max(32, int(max_window_tokens))
        self.max_windows_per_segment = max(1, int(max_windows_per_segment))
        self.approx_chars_per_token = float(approx_chars_per_token)
        self._token_counter = token_counter
        self._window_chars = int(self.max_window_tokens * self.approx_chars_per_token)

    def _estimate_tokens(self, text: str) -> int:
        if self._token_counter is not None:
            return int(self._token_counter(text))
        return max(1, int(len(text) / self.approx_chars_per_token) + 2)

    def build(
        self,
        views: Sequence[str],
        hotspot_spans: Sequence[tuple[int, int, float]] = (),
        source_role: str = "user",
    ) -> WindowPlan:
        t0 = time.perf_counter()
        if not views or not views[0]:
            return WindowPlan(windows=[], latency_ms=0.0)

        primary = views[0]
        windows: list[TextWindow] = []
        est_tokens = self._estimate_tokens(primary)

        if est_tokens <= self.max_window_tokens and len(primary) <= self._window_chars:
            windows.append(
                TextWindow(
                    text=primary,
                    source_role=source_role,
                    window_kind="single",
                    char_start=0,
                    char_end=len(primary),
                    approx_tokens=est_tokens,
                )
            )
        else:
            w_len = self._window_chars
            n = len(primary)
            covered_spans: list[tuple[int, int]] = []

            # 1. Head window
            head_end = min(n, w_len)
            head_text = primary[:head_end]
            windows.append(
                TextWindow(
                    text=head_text,
                    source_role=source_role,
                    window_kind="head",
                    char_start=0,
                    char_end=head_end,
                    approx_tokens=min(self.max_window_tokens, self._estimate_tokens(head_text)),
                )
            )
            covered_spans.append((0, head_end))

            # 2. Tail window
            tail_start = max(0, n - w_len)
            if tail_start > 0 and _max_span_iou((tail_start, n), covered_spans) < 0.65:
                tail_text = primary[tail_start:n]
                windows.append(
                    TextWindow(
                        text=tail_text,
                        source_role=source_role,
                        window_kind="tail",
                        char_start=tail_start,
                        char_end=n,
                        approx_tokens=min(self.max_window_tokens, self._estimate_tokens(tail_text)),
                    )
                )
                covered_spans.append((tail_start, n))

            # 3. Up to Top-2 non-overlapping Hot-Spot windows centered on Aho-Corasick hits
            for hs_start, hs_end, _ in hotspot_spans:
                if len(windows) >= self.max_windows_per_segment:
                    break
                center = (hs_start + hs_end) // 2
                w_start = max(0, center - w_len // 2)
                w_end = min(n, w_start + w_len)
                w_start = max(0, w_end - w_len)
                if _max_span_iou((w_start, w_end), covered_spans) >= 0.50:
                    continue
                hs_text = primary[w_start:w_end]
                windows.append(
                    TextWindow(
                        text=hs_text,
                        source_role=source_role,
                        window_kind="hotspot",
                        char_start=w_start,
                        char_end=w_end,
                        approx_tokens=min(self.max_window_tokens, self._estimate_tokens(hs_text)),
                    )
                )
                covered_spans.append((w_start, w_end))

        # 4. Append de-obfuscated / tag-smuggled secondary views for neural inspection
        for extra_view in views[1:]:
            if not extra_view:
                continue
            clipped = extra_view[: self._window_chars]
            windows.append(
                TextWindow(
                    text=clipped,
                    source_role=source_role,
                    window_kind="decoded_view",
                    char_start=0,
                    char_end=len(clipped),
                    approx_tokens=min(self.max_window_tokens, self._estimate_tokens(clipped)),
                )
            )

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return WindowPlan(windows=windows, latency_ms=round(elapsed_ms, 4))


def _max_span_iou(span: tuple[int, int], existing: Sequence[tuple[int, int]]) -> float:
    best = 0.0
    s1, e1 = span
    for s2, e2 in existing:
        inter = max(0, min(e1, e2) - max(s1, s2))
        if inter <= 0:
            continue
        union = max(e1, e2) - min(s1, s2)
        if union > 0:
            best = max(best, inter / float(union))
    return best
