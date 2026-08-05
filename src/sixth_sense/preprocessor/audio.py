"""
Audio extraction module.

Extracts the audio track from a video file as a WAV file.
This is the first step in the pipeline — every other module
depends on having a clean audio file to work with.

Uses FFmpeg under the hood (already installed on this system).
"""

import os
from pathlib import Path

import ffmpeg


def extract_audio(video_path: str, output_dir: str) -> str:
    """Extract audio from video as 16kHz mono WAV.

    Why 16kHz mono?
    - WhisperX expects 16kHz sample rate
    - Mono channel is all we need for speech analysis
    - WAV format = no compression artifacts

    Args:
        video_path: Path to the input video file
        output_dir: Directory to save the extracted audio

    Returns:
        Path to the extracted WAV file
    """
    video_name = Path(video_path).stem
    audio_path = os.path.join(output_dir, f"{video_name}_audio.wav")

    # If we already extracted audio for this video, skip it
    if os.path.exists(audio_path):
        print(f"  Audio already extracted: {audio_path}")
        return audio_path

    print(f"  Extracting audio from video...")

    (
        ffmpeg
        .input(video_path)
        .output(
            audio_path,
            acodec="pcm_s16le",  # 16-bit PCM (standard WAV format)
            ac=1,                 # mono channel
            ar=16000,             # 16kHz sample rate (what WhisperX wants)
        )
        .overwrite_output()
        .run(quiet=True)
    )

    file_size_mb = os.path.getsize(audio_path) / (1024 * 1024)
    print(f"  Audio extracted: {audio_path} ({file_size_mb:.1f} MB)")

    return audio_path


def get_video_duration(video_path: str) -> float:
    """Get the duration of a video in seconds using FFmpeg probe.

    Args:
        video_path: Path to the video file

    Returns:
        Duration in seconds
    """
    probe = ffmpeg.probe(video_path)
    duration = float(probe["format"]["duration"])
    return duration
