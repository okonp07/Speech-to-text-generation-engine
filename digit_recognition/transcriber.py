"""Speech-to-text transcription helpers."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import tempfile
from typing import Iterable, Optional, Sequence

import numpy as np

from .audio import AudioProcessor


DEFAULT_TRANSCRIPTION_MODEL = "tiny"
DEFAULT_BEAM_SIZE = 5


def _clamp_probability(value: float) -> float:
    return max(0.0, min(1.0, value))


def _probability_from_logprob(avg_logprob: Optional[float]) -> float:
    if avg_logprob is None:
        return 0.0
    return _clamp_probability(math.exp(avg_logprob))


def _segment_confidence(segment: object) -> float:
    words = getattr(segment, "words", None)
    if words:
        probabilities = [
            _clamp_probability(float(getattr(word, "probability")))
            for word in words
            if getattr(word, "probability", None) is not None
        ]
        if probabilities:
            return float(np.mean(probabilities))
    return _probability_from_logprob(getattr(segment, "avg_logprob", None))


def _weighted_confidence(segments: Iterable[object]) -> float:
    weighted_score = 0.0
    total_weight = 0.0

    for segment in segments:
        confidence = _segment_confidence(segment)
        start = float(getattr(segment, "start", 0.0) or 0.0)
        end = float(getattr(segment, "end", start) or start)
        duration = max(end - start, 0.0)
        text = str(getattr(segment, "text", "") or "").strip()
        weight = duration if duration > 0 else max(float(len(text)), 1.0)
        weighted_score += confidence * weight
        total_weight += weight

    if total_weight == 0:
        return 0.0
    return _clamp_probability(weighted_score / total_weight)


@dataclass(frozen=True)
class TranscriptionSegment:
    start_seconds: float
    end_seconds: float
    text: str
    confidence: float


@dataclass(frozen=True)
class TranscriptionResult:
    text: str
    confidence: float
    language: str
    language_confidence: float | None
    duration_seconds: float
    segments: tuple[TranscriptionSegment, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "text": self.text,
            "confidence": self.confidence,
            "language": self.language,
            "language_confidence": self.language_confidence,
            "duration_seconds": self.duration_seconds,
            "segments": [
                {
                    "start_seconds": segment.start_seconds,
                    "end_seconds": segment.end_seconds,
                    "text": segment.text,
                    "confidence": segment.confidence,
                }
                for segment in self.segments
            ],
        }


class SpeechTranscriber:
    """Transcribe speech from audio files or arrays."""

    def __init__(
        self,
        model_size: str = DEFAULT_TRANSCRIPTION_MODEL,
        device: str = "auto",
        compute_type: str | None = None,
        beam_size: int = DEFAULT_BEAM_SIZE,
    ) -> None:
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type or ("float16" if device == "cuda" else "int8")
        self.beam_size = beam_size
        self.processor = AudioProcessor()
        self._model = None

    def _load_model(self):
        if self._model is not None:
            return self._model
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise ImportError(
                "faster-whisper is required for speech transcription. Install requirements.txt first."
            ) from exc

        resolved_device = self.device
        if resolved_device == "auto":
            resolved_device = "cpu"
            try:
                import torch

                if torch.cuda.is_available():
                    resolved_device = "cuda"
            except ImportError:
                resolved_device = "cpu"

        self._model = WhisperModel(
            self.model_size,
            device=resolved_device,
            compute_type=self.compute_type,
        )
        return self._model

    def _result_from_segments(self, segments: Sequence[object], info: object) -> TranscriptionResult:
        normalized_segments = tuple(
            TranscriptionSegment(
                start_seconds=float(getattr(segment, "start", 0.0) or 0.0),
                end_seconds=float(getattr(segment, "end", 0.0) or 0.0),
                text=str(getattr(segment, "text", "") or "").strip(),
                confidence=_segment_confidence(segment),
            )
            for segment in segments
            if str(getattr(segment, "text", "") or "").strip()
        )

        transcript = " ".join(segment.text for segment in normalized_segments).strip()
        duration = max(
            (segment.end_seconds for segment in normalized_segments),
            default=0.0,
        )
        language = str(getattr(info, "language", "unknown") or "unknown")
        language_probability = getattr(info, "language_probability", None)

        return TranscriptionResult(
            text=transcript or "No speech detected.",
            confidence=_weighted_confidence(segments),
            language=language,
            language_confidence=None if language_probability is None else float(language_probability),
            duration_seconds=duration,
            segments=normalized_segments,
        )

    def transcribe_file(
        self,
        audio_path: str | Path,
        language: str | None = None,
        vad_filter: bool = True,
    ) -> TranscriptionResult:
        model = self._load_model()
        segments, info = model.transcribe(
            str(audio_path),
            beam_size=self.beam_size,
            vad_filter=vad_filter,
            word_timestamps=True,
            language=language,
        )
        materialized_segments = tuple(segments)
        return self._result_from_segments(materialized_segments, info)

    def transcribe_array(
        self,
        audio_array: Sequence[float] | np.ndarray,
        sample_rate: int | None = None,
        language: str | None = None,
        vad_filter: bool = True,
    ) -> TranscriptionResult:
        try:
            import soundfile as sf
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise ImportError(
                "soundfile is required to transcribe NumPy arrays. Install requirements.txt first."
            ) from exc

        audio = np.asarray(audio_array, dtype=np.float32)
        if sample_rate and sample_rate != self.processor.sample_rate:
            import librosa

            audio = librosa.resample(
                audio,
                orig_sr=sample_rate,
                target_sr=self.processor.sample_rate,
            )

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            temp_path = Path(temp_file.name)
        try:
            sf.write(temp_path, audio, self.processor.sample_rate)
            return self.transcribe_file(temp_path, language=language, vad_filter=vad_filter)
        finally:
            temp_path.unlink(missing_ok=True)

    def metadata(self) -> dict[str, object]:
        return {
            "model_size": self.model_size,
            "device": self.device,
            "compute_type": self.compute_type,
            "beam_size": self.beam_size,
        }
