"""Convert :class:`TranscriptionResult` objects into downloadable formats.

The Streamlit UI offers the four formats used by the reference Colab
notebook: plain text, SRT subtitles, WebVTT subtitles, and the Whisper-style
JSON dump. All builders return ``str`` — the caller is responsible for
wrapping them in ``st.download_button`` or writing them to disk.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - import cycle guard
    from .transcriber import TranscriptionResult, TranscriptionSegment


def build_txt(result: "TranscriptionResult") -> str:
    """Return the transcript text, or a readable placeholder if empty."""

    return (result.text or "").strip() + "\n"


def build_json(result: "TranscriptionResult") -> str:
    """Whisper-style JSON dump of the result (pretty-printed, UTF-8 safe)."""

    return json.dumps(result.to_dict(), ensure_ascii=False, indent=2) + "\n"


def build_srt(result: "TranscriptionResult") -> str:
    """Return an SRT subtitle file for *result*.

    If the result has no segments, a single entry covering the whole
    duration (or 0–5s as a fallback) is emitted so the download is still
    valid SRT.
    """

    segments = _segments_or_fallback(result)
    lines: list[str] = []
    for index, segment in enumerate(segments, start=1):
        lines.append(str(index))
        lines.append(
            f"{_format_srt_timestamp(segment.start_seconds)} --> "
            f"{_format_srt_timestamp(segment.end_seconds)}"
        )
        lines.append(segment.text.strip())
        lines.append("")  # blank line between cues
    return "\n".join(lines).rstrip() + "\n"


def build_vtt(result: "TranscriptionResult") -> str:
    """Return a WebVTT subtitle file for *result*."""

    segments = _segments_or_fallback(result)
    lines: list[str] = ["WEBVTT", ""]
    for segment in segments:
        lines.append(
            f"{_format_vtt_timestamp(segment.start_seconds)} --> "
            f"{_format_vtt_timestamp(segment.end_seconds)}"
        )
        lines.append(segment.text.strip())
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


# --- internals ---------------------------------------------------------------


def _segments_or_fallback(
    result: "TranscriptionResult",
) -> "tuple[TranscriptionSegment, ...]":
    from .transcriber import TranscriptionSegment  # local import avoids cycle

    if result.segments:
        return result.segments

    end = max(result.duration_seconds or 0.0, 5.0)
    text = (result.text or "").strip() or "No speech detected."
    fallback = TranscriptionSegment(
        start_seconds=0.0,
        end_seconds=end,
        text=text,
        confidence=result.confidence or 0.0,
    )
    return (fallback,)


def _format_srt_timestamp(seconds: float) -> str:
    """SRT uses ``HH:MM:SS,mmm`` (comma decimal separator)."""

    total_ms = int(round(max(seconds, 0.0) * 1000))
    hours, remainder_ms = divmod(total_ms, 3600 * 1000)
    minutes, remainder_ms = divmod(remainder_ms, 60 * 1000)
    seconds_whole, millis = divmod(remainder_ms, 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds_whole:02d},{millis:03d}"


def _format_vtt_timestamp(seconds: float) -> str:
    """WebVTT uses ``HH:MM:SS.mmm`` (dot decimal separator)."""

    return _format_srt_timestamp(seconds).replace(",", ".")


__all__ = [
    "build_json",
    "build_srt",
    "build_txt",
    "build_vtt",
]
