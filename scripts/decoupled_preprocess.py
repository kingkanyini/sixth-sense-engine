"""
Decoupled preprocessor — NSA-fix edition.

Replaces the monolithic pipeline.py with a checkpoint-first design:
1. Re-runs Whisper on existing audio.wav
2. SAVES transcript.json IMMEDIATELY when transcription finishes
3. Generates transcript_timestamped.txt
4. Runs librosa for audio features
5. Generates energy_map.txt

Frames + scenes are NOT in this script — they run separately on a local copy
of the MP4 via FFmpeg + PySceneDetect. See: build_p3_inputs.py.

Why this exists:
- The original pipeline holds the transcript in memory until ALL 6 steps finish.
- cv2 frame sampling has been crashing silently on a Dropbox-synced MP4.
- Two consecutive Whisper runs were lost because no checkpoint.
- This script never loses the transcript.

Run from the sixth-sense project root:
    uv run python scripts/decoupled_preprocess.py --name "my-video"

    # Full override:
    uv run python scripts/decoupled_preprocess.py \\
        --output-dir "output/MyVideo" \\
        --audio-path "output/MyVideo/MyVideo_audio.wav"
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

# Force unbuffered output (PHANTOM's recommendation)
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)


def log(msg: str) -> None:
    """Print with explicit flush + timestamp so the harness sees output."""
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# === Configuration ===
# Derived from this file's location so the script is machine-independent.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_NAME = "my-video"
OUTPUT_DIR = PROJECT_ROOT / "output" / DEFAULT_NAME
AUDIO_PATH = OUTPUT_DIR / f"{DEFAULT_NAME}_audio.wav"

TRANSCRIPT_JSON = OUTPUT_DIR / "transcript.json"
TRANSCRIPT_TXT = OUTPUT_DIR / "transcript_timestamped.txt"
ENERGY_JSON = OUTPUT_DIR / "audio_features.json"
ENERGY_TXT = OUTPUT_DIR / "energy_map.txt"


def fmt_timestamp(seconds: float) -> str:
    """Convert seconds to MM:SS."""
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m:02d}:{s:02d}"


def step_1_transcribe() -> list[dict]:
    """Re-run Whisper on existing WAV, save JSON immediately."""
    log("=" * 60)
    log("STEP 1: Transcribe (will save JSON IMMEDIATELY when done)")
    log("=" * 60)

    if TRANSCRIPT_JSON.exists():
        log(f"  Existing transcript found at {TRANSCRIPT_JSON}")
        log(f"  Loading from disk (skipping Whisper re-run)")
        with open(TRANSCRIPT_JSON, "r", encoding="utf-8") as f:
            return json.load(f)

    if not AUDIO_PATH.exists():
        raise FileNotFoundError(f"Audio file missing: {AUDIO_PATH}")

    log(f"  Audio file: {AUDIO_PATH} ({AUDIO_PATH.stat().st_size / 1e6:.1f} MB)")
    log(f"  Loading faster-whisper...")

    from faster_whisper import WhisperModel

    log(f"  Loading model large-v3 on cuda...")
    model = WhisperModel("large-v3", device="cuda", compute_type="float16")

    log(f"  Starting transcription (this takes ~10 min for 87 min audio)...")
    segments_iter, info = model.transcribe(
        str(AUDIO_PATH),
        word_timestamps=True,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=500),
    )
    log(f"  Detected language: {info.language} ({info.language_probability:.0%})")

    transcript = []
    word_count = 0
    last_log_time = time.time()

    # Iterate through generator — each segment may take ~0.5s.
    # Periodically print heartbeat so the harness sees output.
    for i, segment in enumerate(segments_iter):
        words = []
        if segment.words:
            for w in segment.words:
                words.append({
                    "word": w.word.strip(),
                    "start": round(w.start, 3),
                    "end": round(w.end, 3),
                    "confidence": round(w.probability, 3),
                })
            word_count += len(words)

        transcript.append({
            "start": round(segment.start, 3),
            "end": round(segment.end, 3),
            "text": segment.text.strip(),
            "speaker": None,
            "words": words,
        })

        # Heartbeat every 30s
        if time.time() - last_log_time > 30:
            log(f"  ... transcribed {len(transcript)} segments so far ({fmt_timestamp(segment.end)} of audio)")
            last_log_time = time.time()

    log(f"  Transcribed {len(transcript)} segments, {word_count} words")

    # ============================================================
    # CRITICAL: SAVE TRANSCRIPT TO DISK IMMEDIATELY.
    # No matter what blows up next, we keep this work.
    # ============================================================
    log(f"  >>> CHECKPOINTING transcript.json (the work is now safe)")
    with open(TRANSCRIPT_JSON, "w", encoding="utf-8") as f:
        json.dump(transcript, f, indent=2)
    log(f"  >>> Saved {TRANSCRIPT_JSON} ({TRANSCRIPT_JSON.stat().st_size / 1e6:.2f} MB)")

    # Free the Whisper model + CUDA memory before next step (CIPHER's recommendation)
    log(f"  Releasing Whisper model + CUDA memory...")
    del model
    try:
        import torch
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        log(f"  CUDA memory cleared")
    except Exception as e:
        log(f"  CUDA cleanup skipped: {e}")
    import gc
    gc.collect()
    log(f"  Step 1 complete")

    return transcript


def step_2_write_transcript_txt(transcript: list[dict]) -> None:
    """Write human-readable transcript_timestamped.txt for the agents."""
    log("=" * 60)
    log("STEP 2: Generate transcript_timestamped.txt")
    log("=" * 60)

    lines = []
    for seg in transcript:
        ts = fmt_timestamp(seg["start"])
        lines.append(f"[{ts}] {seg['text']}")

    content = "\n".join(lines)
    TRANSCRIPT_TXT.write_text(content, encoding="utf-8")
    log(f"  Saved {TRANSCRIPT_TXT} ({len(lines)} segments, {len(content):,} chars)")


def step_3_audio_features() -> list[dict]:
    """Run librosa on the WAV to produce audio features per 30s window."""
    log("=" * 60)
    log("STEP 3: Audio features (librosa, 30s windows)")
    log("=" * 60)

    if ENERGY_JSON.exists():
        log(f"  Existing audio_features.json found, loading from disk")
        with open(ENERGY_JSON, "r", encoding="utf-8") as f:
            return json.load(f)

    log(f"  Importing librosa + numpy...")
    import numpy as np
    import librosa

    log(f"  Loading audio at native sample rate...")
    y, sr = librosa.load(str(AUDIO_PATH), sr=None)
    duration = len(y) / sr
    log(f"  Audio: {duration:.0f}s at {sr}Hz, {len(y):,} samples")

    window_seconds = 30.0
    window_samples = int(window_seconds * sr)
    features_list = []

    log(f"  Analyzing in {window_seconds}s windows...")
    last_log = time.time()

    for start_sample in range(0, len(y), window_samples):
        end_sample = min(start_sample + window_samples, len(y))
        window = y[start_sample:end_sample]
        if len(window) < sr:  # less than 1s leftover
            break

        start_time = start_sample / sr
        end_time = end_sample / sr

        # Energy (RMS)
        rms = librosa.feature.rms(y=window)[0]
        energy = float(min(np.mean(rms), 1.0))

        # Tempo
        tempo, _ = librosa.beat.beat_track(y=window, sr=sr)
        tempo_value = float(np.atleast_1d(tempo)[0])

        # Pitch
        pitches, magnitudes = librosa.piptrack(y=window, sr=sr)
        pitch_values = pitches[magnitudes > np.median(magnitudes)]
        pitch_values = pitch_values[pitch_values > 0]
        pitch_mean = float(np.mean(pitch_values)) if len(pitch_values) > 0 else 0.0
        pitch_variance = float(np.var(pitch_values)) if len(pitch_values) > 0 else 0.0

        # Silence ratio
        silence_threshold = 0.01
        silent_frames = np.sum(rms < silence_threshold)
        total_frames = len(rms)
        silence_ratio = float(silent_frames / total_frames) if total_frames > 0 else 0.0

        features_list.append({
            "start": round(start_time, 2),
            "end": round(end_time, 2),
            "energy": round(energy, 4),
            "tempo": round(tempo_value, 1),
            "pitch_mean": round(pitch_mean, 1),
            "pitch_variance": round(pitch_variance, 1),
            "silence_ratio": round(silence_ratio, 3),
        })

        if time.time() - last_log > 30:
            log(f"  ... processed {len(features_list)} windows ({fmt_timestamp(end_time)} of audio)")
            last_log = time.time()

    log(f"  Extracted features for {len(features_list)} windows")

    # Checkpoint
    with open(ENERGY_JSON, "w", encoding="utf-8") as f:
        json.dump(features_list, f, indent=2)
    log(f"  Saved {ENERGY_JSON}")

    return features_list


def step_4_write_energy_map(features: list[dict]) -> None:
    """Write human-readable energy_map.txt for the agents."""
    log("=" * 60)
    log("STEP 4: Generate energy_map.txt")
    log("=" * 60)

    lines = []
    for f in features:
        ts = fmt_timestamp(f["start"])
        # Energy bar: 0..1 -> 0..30 chars
        bar_len = int(f["energy"] * 30)
        bar_len = max(0, min(30, bar_len))
        bar = "█" * bar_len + "░" * (30 - bar_len)
        silence_pct = int(f["silence_ratio"] * 100)
        lines.append(
            f"[{ts}] Energy: {bar} | Tempo: {f['tempo']:.1f} BPM | {silence_pct}% silence"
        )

    content = "\n".join(lines)
    ENERGY_TXT.write_text(content, encoding="utf-8")
    log(f"  Saved {ENERGY_TXT} ({len(lines)} windows, {len(content):,} chars)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Decoupled SixthSense preprocessor (Whisper + librosa, checkpoint-first)"
    )
    parser.add_argument(
        "--name",
        default=DEFAULT_NAME,
        help="Video display name. Drives OUTPUT_DIR and audio filename.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Override OUTPUT_DIR (default: PROJECT_ROOT/output/<name>)",
    )
    parser.add_argument(
        "--audio-path",
        default=None,
        help="Override AUDIO_PATH (default: OUTPUT_DIR/<name>_audio.wav)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    global OUTPUT_DIR, AUDIO_PATH, TRANSCRIPT_JSON, TRANSCRIPT_TXT, ENERGY_JSON, ENERGY_TXT
    OUTPUT_DIR = Path(args.output_dir) if args.output_dir else PROJECT_ROOT / "output" / args.name
    AUDIO_PATH = Path(args.audio_path) if args.audio_path else OUTPUT_DIR / f"{args.name}_audio.wav"
    TRANSCRIPT_JSON = OUTPUT_DIR / "transcript.json"
    TRANSCRIPT_TXT = OUTPUT_DIR / "transcript_timestamped.txt"
    ENERGY_JSON = OUTPUT_DIR / "audio_features.json"
    ENERGY_TXT = OUTPUT_DIR / "energy_map.txt"

    log(f"Decoupled preprocessor - checkpoint-first edition")
    log(f"Name:       {args.name}")
    log(f"Output dir: {OUTPUT_DIR}")
    log(f"Audio:      {AUDIO_PATH}")

    if not AUDIO_PATH.exists():
        log(f"ERROR: Audio file missing - extract audio from MP4 first (ffmpeg or sixth-sense extract)")
        return 1

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    try:
        transcript = step_1_transcribe()
        step_2_write_transcript_txt(transcript)
        features = step_3_audio_features()
        step_4_write_energy_map(features)
    except Exception as e:
        log(f"FATAL ERROR: {type(e).__name__}: {e}")
        log(f"Full traceback:")
        traceback.print_exc(file=sys.stdout)
        sys.stdout.flush()
        return 1

    log("=" * 60)
    log("ALL CORE STEPS COMPLETE")
    log("=" * 60)
    log(f"  Transcript JSON: {TRANSCRIPT_JSON}")
    log(f"  Transcript TXT:  {TRANSCRIPT_TXT}")
    log(f"  Features JSON:   {ENERGY_JSON}")
    log(f"  Energy Map TXT:  {ENERGY_TXT}")
    log(f"")
    log(f"  Ready for: P1-A Content Scanner, P2-A Emotion Scanner")
    log(f"  Still need (parallel track): frames + scenes for P3-A Structure")

    return 0


if __name__ == "__main__":
    sys.exit(main())
