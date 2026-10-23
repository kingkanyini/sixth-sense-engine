"""
Transcription module.

Converts the extracted audio into a word-level timestamped transcript
with speaker diarization (who said what, when).

Uses faster-whisper (a CTranslate2-optimized version of OpenAI's Whisper)
which runs ~4x faster than the original on GPU.

Think of this as converting the audio track into a searchable script —
every word gets a precise timestamp so agents know exactly when
things were said.
"""

import os

from faster_whisper import WhisperModel

from sixth_sense.models import TranscriptSegment, TranscriptWord


def transcribe_audio(
    audio_path: str,
    model_size: str = "large-v3",
    device: str = "cuda",
    compute_type: str | None = None,
    language: str | None = None,
) -> list[TranscriptSegment]:
    """Transcribe audio with word-level timestamps.

    Args:
        audio_path: Path to WAV audio file (16kHz mono)
        model_size: Whisper model size. Options:
                   - "tiny" / "base" / "small" — fast, less accurate
                   - "medium" — good balance
                   - "large-v3" — best accuracy (recommended, needs ~3GB VRAM)
        device: "cuda" for GPU (recommended) or "cpu" (slow but works)
        compute_type: None (default) picks "float16" on GPU and "int8" on CPU.
                      Pass "float32" for max accuracy. CPUs can't run float16.
        language: Language code (e.g., "en"). None = auto-detect.

    Returns:
        List of TranscriptSegment with word-level timestamps
    """
    if compute_type is None:
        compute_type = "float16" if device == "cuda" else "int8"
    print(f"  Loading Whisper model ({model_size}) on {device} ({compute_type})...")

    model = WhisperModel(
        model_size,
        device=device,
        compute_type=compute_type,
    )

    print(f"  Transcribing audio...")

    segments_iter, info = model.transcribe(
        audio_path,
        language=language,
        word_timestamps=True,  # Get per-word timing
        vad_filter=True,       # Voice Activity Detection — skips silence
        vad_parameters=dict(
            min_silence_duration_ms=500,  # 0.5s silence = new segment
        ),
    )

    detected_lang = info.language
    lang_prob = info.language_probability
    print(f"  Detected language: {detected_lang} ({lang_prob:.0%} confidence)")

    transcript = []
    word_count = 0

    for segment in segments_iter:
        words = []
        if segment.words:
            for w in segment.words:
                words.append(TranscriptWord(
                    word=w.word.strip(),
                    start=round(w.start, 3),
                    end=round(w.end, 3),
                    confidence=round(w.probability, 3),
                ))
            word_count += len(words)

        transcript.append(TranscriptSegment(
            start=round(segment.start, 3),
            end=round(segment.end, 3),
            text=segment.text.strip(),
            speaker=None,  # Speaker diarization added separately
            words=words,
        ))

    print(f"  Transcribed {len(transcript)} segments, {word_count} words")

    # Calculate approximate words per minute
    if transcript:
        total_duration = transcript[-1].end - transcript[0].start
        if total_duration > 0:
            wpm = word_count / (total_duration / 60)
            print(f"  Speaking pace: ~{wpm:.0f} words/min")

    return transcript
