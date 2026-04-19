"""Fetch an already-transcribed caption track from YouTube.

The captions endpoint is far less aggressively bot-checked than the
video-stream endpoint, so this path works from datacenter IPs (like
Streamlit Community Cloud) where ``yt-dlp`` typically fails. It is also
dramatically faster than downloading audio and running Whisper: we get the
author's captions (or YouTube's auto-generated ones) back as structured
segments in under a second.

When no captions exist for the URL (e.g. many music videos, private
streams, some Shorts), this module raises :class:`CaptionsUnavailableError`
and the caller is expected to tell the user to try a different URL or use
the video-file upload tab.
"""

from __future__ import annotations

import re
from typing import Optional, Sequence

from .transcriber import TranscriptionResult, TranscriptionSegment


class CaptionsError(RuntimeError):
    """Base class for caption-fetching failures."""


class CaptionsUnavailableError(CaptionsError):
    """Raised when the video has no retrievable captions at all."""


class InvalidYouTubeUrlError(CaptionsError):
    """Raised when we cannot extract a video ID from the given URL."""


# Matches the common YouTube URL shapes and captures the 11-char video ID.
_VIDEO_ID_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"youtu\.be/([A-Za-z0-9_-]{11})"),
    re.compile(r"youtube\.com/watch\?[^ ]*?\bv=([A-Za-z0-9_-]{11})"),
    re.compile(r"youtube\.com/shorts/([A-Za-z0-9_-]{11})"),
    re.compile(r"youtube\.com/embed/([A-Za-z0-9_-]{11})"),
    re.compile(r"youtube\.com/live/([A-Za-z0-9_-]{11})"),
    re.compile(r"youtube\.com/v/([A-Za-z0-9_-]{11})"),
)


def extract_video_id(url: str) -> str:
    """Return the YouTube video ID embedded in *url*, or raise."""

    if not url:
        raise InvalidYouTubeUrlError("No URL was provided.")
    trimmed = url.strip()
    for pattern in _VIDEO_ID_PATTERNS:
        match = pattern.search(trimmed)
        if match:
            return match.group(1)
    raise InvalidYouTubeUrlError(
        "Could not find a YouTube video ID in the URL. "
        "Paste a link like https://www.youtube.com/watch?v=… or https://youtu.be/…"
    )


def fetch_youtube_captions(
    url: str,
    language_hint: Optional[str] = None,
) -> TranscriptionResult:
    """Fetch captions for *url* and return them as a :class:`TranscriptionResult`.

    Parameters
    ----------
    url:
        A YouTube URL (watch, shorts, youtu.be, embed, live).
    language_hint:
        Preferred ISO language code (e.g. ``"en"``). When supplied, the
        corresponding track is tried first; we otherwise fall back to
        whatever track YouTube has. ``None`` means "let YouTube pick".
    """

    try:
        from youtube_transcript_api import (  # type: ignore
            YouTubeTranscriptApi,
        )
        from youtube_transcript_api._errors import (  # type: ignore
            NoTranscriptFound,
            TranscriptsDisabled,
            VideoUnavailable,
        )
    except ImportError as exc:  # pragma: no cover - env-dependent
        raise CaptionsError(
            "youtube-transcript-api is not installed. Add it to requirements.txt."
        ) from exc

    video_id = extract_video_id(url)

    # The language priority order we try. We always fall through to the
    # generated/auto tracks at the end so we get *something* when we can.
    preferred_languages: list[str] = []
    if language_hint:
        preferred_languages.append(language_hint)
    for fallback in ("en", "en-US", "en-GB"):
        if fallback not in preferred_languages:
            preferred_languages.append(fallback)

    entries: Sequence[dict]
    chosen_language = "unknown"

    try:
        # list_transcripts gives us the set of available tracks so we can
        # prefer a manually-authored caption over an auto-generated one.
        try:
            transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        except AttributeError:  # pragma: no cover - older/newer API variants
            transcript_list = None

        if transcript_list is not None:
            transcript = None
            try:
                transcript = transcript_list.find_manually_created_transcript(
                    preferred_languages
                )
            except Exception:
                try:
                    transcript = transcript_list.find_generated_transcript(
                        preferred_languages
                    )
                except Exception:
                    # Fall back to whatever is available.
                    try:
                        transcript = next(iter(transcript_list), None)
                    except Exception:
                        transcript = None
            if transcript is None:
                raise NoTranscriptFound(
                    video_id, preferred_languages, transcript_list
                )
            entries = transcript.fetch()
            chosen_language = getattr(transcript, "language_code", "unknown") or "unknown"
        else:
            # Very old / very new API fallback: pull directly.
            entries = YouTubeTranscriptApi.get_transcript(
                video_id, languages=preferred_languages
            )
            chosen_language = language_hint or preferred_languages[0] or "unknown"

    except TranscriptsDisabled as exc:
        raise CaptionsUnavailableError(
            "Captions are disabled on this video."
        ) from exc
    except NoTranscriptFound as exc:
        raise CaptionsUnavailableError(
            "No captions are available for this video."
        ) from exc
    except VideoUnavailable as exc:
        raise CaptionsUnavailableError(
            "YouTube reports this video as unavailable."
        ) from exc
    except Exception as exc:  # pragma: no cover - defensive / network
        message = str(exc).strip() or exc.__class__.__name__
        lowered = message.lower()
        if "could not retrieve" in lowered or "no element found" in lowered:
            raise CaptionsUnavailableError(
                "YouTube returned no captions for this video."
            ) from exc
        raise CaptionsError(
            f"Captions fetch failed. Details: {message}"
        ) from exc

    segments = _entries_to_segments(entries)
    if not segments:
        raise CaptionsUnavailableError(
            "Captions track exists but is empty."
        )

    transcript_text = " ".join(seg.text for seg in segments).strip()
    duration = max((seg.end_seconds for seg in segments), default=0.0)

    return TranscriptionResult(
        text=transcript_text or "No speech detected.",
        # Captions are an official track; confidence doesn't map cleanly
        # onto a probability. Report a neutral 1.0 to distinguish from a
        # Whisper run, and surface the "source" via other UI affordances.
        confidence=1.0,
        language=chosen_language,
        language_confidence=None,
        duration_seconds=duration,
        segments=tuple(segments),
    )


def _entries_to_segments(
    entries: Sequence[object],
) -> list[TranscriptionSegment]:
    segments: list[TranscriptionSegment] = []
    for raw in entries:
        # The library has shipped two shapes: dicts in older versions, and
        # objects (``FetchedTranscriptSnippet``) in newer ones. Handle both.
        if isinstance(raw, dict):
            text = str(raw.get("text", "") or "").strip()
            start = float(raw.get("start", 0.0) or 0.0)
            duration = float(raw.get("duration", 0.0) or 0.0)
        else:
            text = str(getattr(raw, "text", "") or "").strip()
            start = float(getattr(raw, "start", 0.0) or 0.0)
            duration = float(getattr(raw, "duration", 0.0) or 0.0)

        if not text or text == "[Music]":
            continue
        segments.append(
            TranscriptionSegment(
                start_seconds=start,
                end_seconds=start + max(duration, 0.0),
                text=text,
                confidence=1.0,
            )
        )
    return segments


__all__ = [
    "CaptionsError",
    "CaptionsUnavailableError",
    "InvalidYouTubeUrlError",
    "extract_video_id",
    "fetch_youtube_captions",
]
