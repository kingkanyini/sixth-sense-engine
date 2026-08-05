"""
Audio feature extraction module.

Extracts energy, tempo, pitch, and silence information from the audio.
Think of this as the "energy meter" for the video — it tells you where
the speaker gets loud, fast, quiet, or pauses dramatically.

These features help the Emotion agents understand the audio dynamics
without needing to listen to the actual audio.
"""

import numpy as np
import librosa

from sixth_sense.models import AudioFeatures


def extract_audio_features(
    audio_path: str,
    window_seconds: float = 30.0,
) -> list[AudioFeatures]:
    """Extract audio features in time windows across the full audio.

    Args:
        audio_path: Path to the WAV audio file (16kHz mono, from extract_audio)
        window_seconds: Size of each analysis window in seconds.
                       30 seconds gives a good overview. Smaller windows (10-15s)
                       give more granular data but create more entries.

    Returns:
        List of AudioFeatures, one per window
    """
    print(f"  Loading audio for feature extraction...")

    # Load audio (librosa auto-resamples to 22050 by default, which is fine)
    y, sr = librosa.load(audio_path, sr=None)  # sr=None keeps original 16kHz
    duration = len(y) / sr

    print(f"  Audio: {duration:.0f}s at {sr}Hz")
    print(f"  Analyzing in {window_seconds}s windows...")

    window_samples = int(window_seconds * sr)
    features_list = []

    for start_sample in range(0, len(y), window_samples):
        end_sample = min(start_sample + window_samples, len(y))
        window = y[start_sample:end_sample]

        # Skip tiny leftover windows at the end
        if len(window) < sr:  # less than 1 second
            break

        start_time = start_sample / sr
        end_time = end_sample / sr

        features_list.append(_analyze_window(window, sr, start_time, end_time))

    print(f"  Extracted features for {len(features_list)} windows")

    return features_list


def _analyze_window(
    window: np.ndarray,
    sr: int,
    start: float,
    end: float,
) -> AudioFeatures:
    """Analyze a single audio window and extract features.

    Each feature explained:
    - energy: How loud this section is (0=silent, 1=loudest in video)
    - tempo: Speaking pace or musical tempo in BPM
    - pitch_mean: Average pitch in Hz (higher = more excited usually)
    - pitch_variance: How much pitch changes (monotone vs expressive)
    - silence_ratio: What fraction of this window is silence
    """
    # Energy (RMS — root mean square, a measure of loudness)
    rms = librosa.feature.rms(y=window)[0]
    energy = float(np.mean(rms))

    # Normalize energy to 0-1 range (will be globally normalized later)
    # For now, just store the raw RMS value
    energy = min(energy, 1.0)

    # Tempo (beats per minute — useful for detecting pace changes)
    tempo, _ = librosa.beat.beat_track(y=window, sr=sr)
    tempo_value = float(np.atleast_1d(tempo)[0])

    # Pitch (fundamental frequency via piptrack)
    pitches, magnitudes = librosa.piptrack(y=window, sr=sr)
    # Get pitches where magnitude is significant
    pitch_values = pitches[magnitudes > np.median(magnitudes)]
    pitch_values = pitch_values[pitch_values > 0]  # remove zeros

    if len(pitch_values) > 0:
        pitch_mean = float(np.mean(pitch_values))
        pitch_variance = float(np.var(pitch_values))
    else:
        pitch_mean = 0.0
        pitch_variance = 0.0

    # Silence ratio (fraction of frames below a threshold)
    silence_threshold = 0.01  # RMS below this = "silence"
    silent_frames = np.sum(rms < silence_threshold)
    total_frames = len(rms)
    silence_ratio = float(silent_frames / total_frames) if total_frames > 0 else 0.0

    return AudioFeatures(
        start=round(start, 2),
        end=round(end, 2),
        energy=round(energy, 4),
        tempo=round(tempo_value, 1),
        pitch_mean=round(pitch_mean, 1),
        pitch_variance=round(pitch_variance, 1),
        silence_ratio=round(silence_ratio, 3),
    )
