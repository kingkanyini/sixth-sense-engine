"""
Main preprocessing pipeline.

This is the orchestrator that runs all preprocessing steps in order
and produces a single PreprocessedData "save file" that agents will consume.

The pipeline flow:
  1. Probe video (duration, metadata)
  2. Extract audio (FFmpeg)
  3. Transcribe audio (faster-whisper on GPU)
  4. Sample frames (OpenCV)
  5. Detect scenes (PySceneDetect)
  6. Extract audio features (librosa)
  7. Assign scene IDs to frames
  8. Save the PreprocessedData as JSON

Steps 4, 5, and 6 are independent of each other (but all depend on step 2).
For now we run them sequentially for simplicity.
"""

import os
import json
from datetime import datetime, timezone
from pathlib import Path

from rich.console import Console
from rich.panel import Panel

from sixth_sense.models import PreprocessedData
from sixth_sense.preprocessor.audio import extract_audio, get_video_duration
from sixth_sense.preprocessor.transcript import transcribe_audio
from sixth_sense.preprocessor.frames import sample_frames
from sixth_sense.preprocessor.scenes import detect_scenes
from sixth_sense.preprocessor.features import extract_audio_features

console = Console()


def preprocess_video(
    video_path: str,
    output_dir: str,
    frame_interval: float = 5.0,
    audio_window: float = 30.0,
    scene_threshold: float = 27.0,
    whisper_model: str = "large-v3",
    whisper_device: str = "cuda",
    skip_transcript: bool = False,
) -> PreprocessedData:
    """Run the full preprocessing pipeline on a video.

    This creates the "save file" that all 6 agents will load.

    Args:
        video_path: Path to the input video file
        output_dir: Directory for all output files (audio, frames, JSON)
        frame_interval: Seconds between frame samples (default 5.0)
        audio_window: Seconds per audio analysis window (default 30.0)
        scene_threshold: Scene detection sensitivity (default 27.0)
        whisper_model: Whisper model size (default "large-v3")
        whisper_device: "cuda" for GPU, "cpu" for CPU (default "cuda")
        skip_transcript: Skip transcription step (default False)

    Returns:
        PreprocessedData object with all extracted data
    """
    video_path = os.path.abspath(video_path)
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    video_name = Path(video_path).stem
    total_steps = 6 if not skip_transcript else 5

    console.print(Panel(
        f"[bold]SixthSense Preprocessor[/bold]\n"
        f"Video: {Path(video_path).name}\n"
        f"Output: {output_dir}\n"
        f"Whisper: {whisper_model} on {whisper_device}"
        + (" [dim](skipped)[/dim]" if skip_transcript else ""),
        title="Preprocessing Pipeline",
        border_style="cyan",
    ))

    step = 0

    # Step 1: Get video duration
    step += 1
    console.print(f"\n[bold cyan]Step {step}/{total_steps}:[/bold cyan] Probing video...")
    duration = get_video_duration(video_path)
    minutes = int(duration // 60)
    seconds = int(duration % 60)
    console.print(f"  Duration: {minutes}m {seconds}s ({duration:.1f}s)")

    # Step 2: Extract audio
    step += 1
    console.print(f"\n[bold cyan]Step {step}/{total_steps}:[/bold cyan] Extracting audio...")
    audio_path = extract_audio(video_path, output_dir)

    # Step 3: Transcribe (the big one — runs Whisper on GPU)
    transcript = []
    if not skip_transcript:
        step += 1
        console.print(f"\n[bold cyan]Step {step}/{total_steps}:[/bold cyan] "
                       f"Transcribing with Whisper ({whisper_model})...")
        transcript = transcribe_audio(
            audio_path,
            model_size=whisper_model,
            device=whisper_device,
        )
    else:
        console.print("\n[dim]Skipping transcription[/dim]")

    # Step 4: Sample frames
    step += 1
    console.print(f"\n[bold cyan]Step {step}/{total_steps}:[/bold cyan] Sampling frames...")
    frames = sample_frames(video_path, output_dir, interval_seconds=frame_interval)

    # Step 5: Detect scenes
    step += 1
    console.print(f"\n[bold cyan]Step {step}/{total_steps}:[/bold cyan] Detecting scenes...")
    scenes = detect_scenes(video_path, threshold=scene_threshold)

    # Step 6: Extract audio features
    step += 1
    console.print(f"\n[bold cyan]Step {step}/{total_steps}:[/bold cyan] Extracting audio features...")
    audio_features = extract_audio_features(audio_path, window_seconds=audio_window)

    # Assign scene IDs to frames
    for frame in frames:
        for scene in scenes:
            if scene.start <= frame.timestamp < scene.end:
                frame.scene_id = scene.scene_id
                break

    # Count transcript words
    word_count = sum(len(seg.words) for seg in transcript)

    # Build the save file
    data = PreprocessedData(
        video_path=video_path,
        video_filename=Path(video_path).name,
        duration_seconds=round(duration, 2),
        audio_path=audio_path,
        transcript=transcript,
        frames=frames,
        audio_features=audio_features,
        scenes=scenes,
        frame_interval_seconds=frame_interval,
        audio_window_seconds=audio_window,
        preprocessed_at=datetime.now(timezone.utc).isoformat(),
    )

    # Save as JSON
    json_path = os.path.join(output_dir, f"{video_name}_preprocessed.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data.model_dump(), f, indent=2)

    transcript_line = (
        f"  Transcript: {len(transcript)} segments, {word_count} words"
        if transcript
        else "  Transcript: [dim]skipped[/dim]"
    )

    console.print(Panel(
        f"[bold green]Preprocessing complete![/bold green]\n\n"
        f"  Frames:        {len(frames)}\n"
        f"  Scenes:        {len(scenes)}\n"
        f"  Audio windows: {len(audio_features)}\n"
        f"{transcript_line}\n\n"
        f"  Save file: {json_path}",
        title="Results",
        border_style="green",
    ))

    return data
