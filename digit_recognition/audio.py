"""Audio preprocessing utilities shared by training, evaluation, and the app."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import tempfile
from typing import Optional, Sequence
import wave

import numpy as np


DEFAULT_SAMPLE_RATE = 22_050
DEFAULT_MAX_DURATION = 1.0
DEFAULT_N_MFCC = 13
DEFAULT_N_FFT = 512
DEFAULT_HOP_LENGTH = 256
EXPECTED_TIME_FRAMES = 87


def _require_librosa():
    cache_dir = Path(tempfile.gettempdir()) / "digit_classifier_numba_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("NUMBA_CACHE_DIR", str(cache_dir))
    try:
        import librosa
    except ImportError as exc:  # pragma: no cover - exercised in real runtime only
        raise ImportError(
            "librosa is required for audio processing. Install project requirements first."
        ) from exc
    return librosa


def _resample_audio(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr or len(audio) == 0:
        return audio.astype(np.float32, copy=False)
    duration = len(audio) / float(orig_sr)
    target_length = max(int(round(duration * target_sr)), 1)
    original_positions = np.linspace(0.0, duration, num=len(audio), endpoint=False)
    target_positions = np.linspace(0.0, duration, num=target_length, endpoint=False)
    return np.interp(target_positions, original_positions, audio).astype(np.float32, copy=False)


def _load_wav_via_stdlib(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as wav_file:
        sample_rate = wav_file.getframerate()
        sample_width = wav_file.getsampwidth()
        channels = wav_file.getnchannels()
        frames = wav_file.readframes(wav_file.getnframes())

    dtype_map = {1: np.uint8, 2: np.int16, 4: np.int32}
    dtype = dtype_map.get(sample_width)
    if dtype is None:
        raise ValueError(f"Unsupported WAV sample width: {sample_width}")

    audio = np.frombuffer(frames, dtype=dtype)
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)

    if sample_width == 1:
        audio = (audio.astype(np.float32) - 128.0) / 128.0
    else:
        max_value = float(np.iinfo(dtype).max)
        audio = audio.astype(np.float32) / max_value

    return audio.astype(np.float32, copy=False), sample_rate


@dataclass(frozen=True)
class AudioQualityReport:
    duration_seconds: float
    peak_amplitude: float
    rms_amplitude: float
    issues: tuple[str, ...]


class AudioProcessor:
    """Load audio, normalize duration, and extract MFCC features."""

    def __init__(
        self,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        max_duration: float = DEFAULT_MAX_DURATION,
        n_mels: int = DEFAULT_N_MFCC,
        n_fft: int = DEFAULT_N_FFT,
        hop_length: int = DEFAULT_HOP_LENGTH,
    ) -> None:
        self.sample_rate = sample_rate
        self.max_duration = max_duration
        self.max_length = int(max_duration * sample_rate)
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self._df_model = None

    def _load_df_model(self):
        if self._df_model is not None:
            return self._df_model
        try:
            from df.enhance import init_df
            self._df_model = init_df()
        except ImportError:
            # Fallback if deepfilternet is not available
            return None
        return self._df_model

    def denoise(self, audio: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
        """Apply DeepFilterNet noise reduction."""
        model = self._load_df_model()
        if model is None:
            return audio
            
        try:
            from df.enhance import enhance, load_audio, save_audio
            # DeepFilterNet expects 48kHz internally but can handle 16kHz
            # For simplicity in this Task 2.1, we wrap the array in the required format
            import torch
            
            # Convert to torch tensor
            audio_torch = torch.from_numpy(audio).unsqueeze(0) # [1, samples]
            
            # Enhance
            enhanced = enhance(model, model[0], audio_torch, sample_rate=sample_rate)
            
            # Convert back to numpy
            return enhanced.squeeze(0).numpy()
        except Exception:
            return audio

    def load_audio(self, path: str | Path) -> np.ndarray:
        audio_path = Path(path)
        try:
            librosa = _require_librosa()
        except ImportError:
            try:
                import soundfile as sf

                audio, sample_rate = sf.read(audio_path, dtype="float32", always_2d=False)
                if np.ndim(audio) > 1:
                    audio = np.mean(audio, axis=1)
                return _resample_audio(np.asarray(audio, dtype=np.float32), sample_rate, self.sample_rate)
            except Exception:
                if audio_path.suffix.lower() != ".wav":
                    raise
                audio, sample_rate = _load_wav_via_stdlib(audio_path)
                return _resample_audio(audio, sample_rate, self.sample_rate)

        audio, _ = librosa.load(audio_path, sr=self.sample_rate, mono=True)
        return audio.astype(np.float32, copy=False)

    def to_mono(self, audio: Sequence[float] | np.ndarray) -> np.ndarray:
        array = np.asarray(audio, dtype=np.float32)
        if array.ndim > 1:
            array = np.mean(array, axis=-1)
        if not len(array):
            return np.zeros(self.max_length, dtype=np.float32)
        array = np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)
        return array.astype(np.float32, copy=False)

    def normalize_audio(self, audio: Sequence[float] | np.ndarray) -> np.ndarray:
        array = self.to_mono(audio)
        array = array - float(np.mean(array))
        peak = float(np.max(np.abs(array))) if len(array) else 0.0
        if peak > 0:
            array = 0.95 * (array / peak)
        return array.astype(np.float32, copy=False)

    def trim_silence(self, audio: Sequence[float] | np.ndarray, top_db: int = 30) -> np.ndarray:
        array = self.to_mono(audio)
        try:
            librosa = _require_librosa()
        except ImportError:
            if len(array) == 0:
                return array
            threshold = float(np.max(np.abs(array))) * (10 ** (-top_db / 20))
            active = np.flatnonzero(np.abs(array) > threshold)
            if len(active) == 0:
                return array
            return array[int(active[0]) : int(active[-1]) + 1].astype(np.float32, copy=False)

        trimmed, _ = librosa.effects.trim(
            array,
            top_db=top_db,
            frame_length=self.n_fft,
            hop_length=self.hop_length,
        )
        if len(trimmed) == 0:
            return array
        return trimmed.astype(np.float32, copy=False)

    def select_active_window(self, audio: Sequence[float] | np.ndarray) -> np.ndarray:
        array = self.to_mono(audio)
        if len(array) <= self.max_length:
            return array
        energy = np.square(array)
        window = np.ones(self.max_length, dtype=np.float32)
        scores = np.convolve(energy, window, mode="valid")
        start = int(np.argmax(scores))
        return array[start : start + self.max_length].astype(np.float32, copy=False)

    def prepare_audio(self, audio: Sequence[float] | np.ndarray) -> np.ndarray:
        array = self.normalize_audio(audio)
        trimmed = self.trim_silence(array)
        return self.normalize_audio(trimmed)

    def pad_or_trim(self, audio: Sequence[float] | np.ndarray) -> np.ndarray:
        array = self.to_mono(audio)
        if len(array) > self.max_length:
            array = self.select_active_window(array)
        elif len(array) < self.max_length:
            padding = self.max_length - len(array)
            array = np.pad(array, (0, padding), mode="constant")
        return array.astype(np.float32, copy=False)

    def inference_clips(self, audio: Sequence[float] | np.ndarray) -> list[np.ndarray]:
        prepared = self.prepare_audio(audio)
        if len(prepared) <= self.max_length:
            return [self.pad_or_trim(prepared)]

        energy = np.square(prepared)
        window = np.ones(self.max_length, dtype=np.float32)
        scores = np.convolve(energy, window, mode="valid")
        best_start = int(np.argmax(scores))
        offset = max(self.max_length // 8, 1)
        max_start = len(prepared) - self.max_length
        candidate_starts = [
            max(0, min(best_start - offset, max_start)),
            best_start,
            max(0, min(best_start + offset, max_start)),
        ]

        clips: list[np.ndarray] = []
        seen: set[int] = set()
        for start in candidate_starts:
            if start in seen:
                continue
            seen.add(start)
            clips.append(prepared[start : start + self.max_length].astype(np.float32, copy=False))
        return clips

    def extract_mfcc(self, audio: Sequence[float] | np.ndarray) -> np.ndarray:
        librosa = _require_librosa()
        processed = self.pad_or_trim(self.prepare_audio(audio))
        mfcc = librosa.feature.mfcc(
            y=processed,
            sr=self.sample_rate,
            n_mfcc=self.n_mels,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
        )
        if mfcc.shape[1] != EXPECTED_TIME_FRAMES:
            mfcc = librosa.util.fix_length(mfcc, size=EXPECTED_TIME_FRAMES, axis=1)
        return mfcc.astype(np.float32, copy=False)

    def load_and_preprocess(self, path: str | Path) -> np.ndarray:
        return self.extract_mfcc(self.load_audio(path))

    def resample_audio(
        self,
        audio: Sequence[float] | np.ndarray,
        orig_sample_rate: int,
        target_sample_rate: int | None = None,
    ) -> np.ndarray:
        target_sr = target_sample_rate or self.sample_rate
        return _resample_audio(np.asarray(audio, dtype=np.float32), orig_sample_rate, target_sr)

    def quality_report(
        self,
        audio: Sequence[float] | np.ndarray,
        sample_rate: Optional[int] = None,
    ) -> AudioQualityReport:
        array = np.asarray(audio, dtype=np.float32)
        sr = sample_rate or self.sample_rate
        duration = float(len(array) / sr) if sr else 0.0
        peak = float(np.max(np.abs(array))) if len(array) else 0.0
        rms = float(np.sqrt(np.mean(np.square(array)))) if len(array) else 0.0

        issues: list[str] = []
        if duration < 0.5:
            issues.append("Audio is shorter than 0.5 seconds.")
        if peak < 0.01:
            issues.append("Audio appears very quiet.")
        if peak > 0.99:
            issues.append("Audio may be clipping.")

        return AudioQualityReport(
            duration_seconds=duration,
            peak_amplitude=peak,
            rms_amplitude=rms,
            issues=tuple(issues),
        )
