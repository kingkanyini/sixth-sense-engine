"""
Build P3 Structure Scanner inputs from a LOCAL MP4 copy.

This script handles the parts that the original pipeline.py was crashing on:
- Frame sampling (uses FFmpeg directly, NOT cv2 frame-by-frame - much faster + reliable)
- Scene detection (PySceneDetect on the local copy)

Outputs:
- frames/frame_NNNN.jpg            (one frame every 5s)
- scenes.json                       (scene boundaries + IDs)
- <name>_preprocessed.json         (combined save file with transcript + features
                                     + frames + scenes, ready for P3-A/P3-B agents)

Run from sixth-sense project root AFTER decoupled_preprocess.py has finished
(needs transcript.json + audio_features.json on disk):

    uv run python scripts/build_p3_inputs.py \\
        --name "my-video" \\
        --local-mp4 "/tmp/sixth-sense/my-video.mp4" \\
        --original-mp4 "/path/to/original/my-video.mp4" \\
        --video-filename "my-video.mp4"

All paths default to ./output/<name>/ relative to the project root.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import traceback
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# === Configuration ===
# Derived from this file's location so the script is machine-independent.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_NAME = "my-video"
OUTPUT_DIR = PROJECT_ROOT / "output" / DEFAULT_NAME

LOCAL_MP4 = Path(f"/tmp/sixth-sense/{DEFAULT_NAME}.mp4")
ORIGINAL_DROPBOX_MP4 = Path(f"/path/to/original/{DEFAULT_NAME}.mp4")

FRAMES_DIR = OUTPUT_DIR / "frames"
SCENES_JSON = OUTPUT_DIR / "scenes.json"
SAVE_FILE_JSON = OUTPUT_DIR / f"{DEFAULT_NAME}_preprocessed.json"

TRANSCRIPT_JSON = OUTPUT_DIR / "transcript.json"
ENERGY_JSON = OUTPUT_DIR / "audio_features.json"

# These two get reassigned in main() from CLI args.
VIDEO_FILENAME = f"{DEFAULT_NAME}.mp4"
AUDIO_PATH = OUTPUT_DIR / f"{DEFAULT_NAME}_audio.wav"

FRAME_INTERVAL_SECONDS = 5.0
SCENE_THRESHOLD = 27.0


def ffprobe_duration(path: Path) -> float:
    """Get video duration in seconds via ffprobe."""
    result = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        capture_output=True, text=True, check=True,
    )
    return float(result.stdout.strip())


def step_1_extract_frames(video_path: Path) -> list[dict]:
    """Use FFmpeg to extract one frame every FRAME_INTERVAL_SECONDS.

    This replaces the cv2 frame-by-frame approach that was crashing.
    FFmpeg seeks directly without reading every frame.
    """
    log("=" * 60)
    log("STEP 1: Extract frames via FFmpeg (replaces broken cv2)")
    log("=" * 60)

    FRAMES_DIR.mkdir(parents=True, exist_ok=True)
    duration = ffprobe_duration(video_path)
    expected_frames = int(duration / FRAME_INTERVAL_SECONDS)
    log(f"  Video duration: {duration:.0f}s")
    log(f"  Sampling every {FRAME_INTERVAL_SECONDS}s -> ~{expected_frames} frames")

    # FFmpeg: -vf fps=1/5 means 1 frame every 5 seconds.
    # JPEG quality 3 is high (range 1=best, 31=worst).
    fps_value = 1.0 / FRAME_INTERVAL_SECONDS
    output_pattern = str(FRAMES_DIR / "frame_%04d.jpg")

    log(f"  Running FFmpeg...")
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(video_path),
        "-vf", f"fps={fps_value}",
        "-q:v", "3",
        output_pattern,
    ]
    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - t0

    if result.returncode != 0:
        log(f"  FFmpeg stderr: {result.stderr}")
        raise RuntimeError(f"FFmpeg frame extraction failed (exit {result.returncode})")

    frame_files = sorted(FRAMES_DIR.glob("frame_*.jpg"))
    log(f"  Extracted {len(frame_files)} frames in {elapsed:.1f}s")

    # Build frame metadata: timestamp = i * FRAME_INTERVAL_SECONDS
    samples = []
    for i, fpath in enumerate(frame_files):
        timestamp = round(i * FRAME_INTERVAL_SECONDS, 2)
        samples.append({
            "timestamp": timestamp,
            "frame_path": str(fpath),
            "scene_id": None,
        })

    return samples


def step_2_detect_scenes(video_path: Path) -> list[dict]:
    """Run PySceneDetect on the LOCAL copy to get scene boundaries."""
    log("=" * 60)
    log("STEP 2: Detect scenes via PySceneDetect")
    log("=" * 60)

    log(f"  Importing scenedetect...")
    from scenedetect import detect, ContentDetector

    log(f"  Running content detection (threshold={SCENE_THRESHOLD})...")
    t0 = time.time()
    scene_list = detect(
        str(video_path),
        ContentDetector(threshold=SCENE_THRESHOLD),
        show_progress=False,
    )
    elapsed = time.time() - t0
    log(f"  Detected {len(scene_list)} scenes in {elapsed:.1f}s")

    scenes = []
    for i, (start, end) in enumerate(scene_list):
        scenes.append({
            "scene_id": i,
            "start": round(start.get_seconds(), 3),
            "end": round(end.get_seconds(), 3),
            "duration": round(end.get_seconds() - start.get_seconds(), 3),
        })

    # Checkpoint
    with open(SCENES_JSON, "w", encoding="utf-8") as f:
        json.dump(scenes, f, indent=2)
    log(f"  Saved {SCENES_JSON}")

    return scenes


def step_3_assign_scene_ids(frames: list[dict], scenes: list[dict]) -> None:
    """Assign each frame to its containing scene (in-place)."""
    for frame in frames:
        for scene in scenes:
            if scene["start"] <= frame["timestamp"] < scene["end"]:
                frame["scene_id"] = scene["scene_id"]
                break


def step_4_build_save_file(frames: list[dict], scenes: list[dict]) -> None:
    """Combine transcript + audio_features + frames + scenes into a single
    save file matching the original pipeline's output schema."""
    log("=" * 60)
    log("STEP 4: Build combined save_file.json")
    log("=" * 60)

    if not TRANSCRIPT_JSON.exists():
        log(f"  ERROR: {TRANSCRIPT_JSON} not found. Run decoupled_preprocess.py first.")
        raise FileNotFoundError(str(TRANSCRIPT_JSON))
    if not ENERGY_JSON.exists():
        log(f"  ERROR: {ENERGY_JSON} not found. Run decoupled_preprocess.py first.")
        raise FileNotFoundError(str(ENERGY_JSON))

    log(f"  Loading transcript + features...")
    with open(TRANSCRIPT_JSON, "r", encoding="utf-8") as f:
        transcript = json.load(f)
    with open(ENERGY_JSON, "r", encoding="utf-8") as f:
        audio_features = json.load(f)

    duration = ffprobe_duration(LOCAL_MP4)
    from datetime import datetime, timezone

    save_data = {
        "video_path": str(ORIGINAL_DROPBOX_MP4),
        "video_filename": VIDEO_FILENAME,
        "duration_seconds": round(duration, 2),
        "audio_path": str(AUDIO_PATH),
        "transcript": transcript,
        "frames": frames,
        "audio_features": audio_features,
        "scenes": scenes,
        "frame_interval_seconds": FRAME_INTERVAL_SECONDS,
        "audio_window_seconds": 30.0,
        "preprocessed_at": datetime.now(timezone.utc).isoformat(),
    }

    with open(SAVE_FILE_JSON, "w", encoding="utf-8") as f:
        json.dump(save_data, f, indent=2)

    size_mb = SAVE_FILE_JSON.stat().st_size / 1e6
    log(f"  Saved {SAVE_FILE_JSON} ({size_mb:.2f} MB)")
    log(f"  Contents: {len(transcript)} transcript segs, {len(frames)} frames, {len(scenes)} scenes, {len(audio_features)} audio windows")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build P3 inputs (frames + scenes + combined save file) for SixthSense"
    )
    parser.add_argument(
        "--name",
        default=DEFAULT_NAME,
        help="Video display name. Drives OUTPUT_DIR and save-file naming.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Override OUTPUT_DIR (default: PROJECT_ROOT/output/<name>)",
    )
    parser.add_argument(
        "--local-mp4",
        default=None,
        help="Path to LOCAL MP4 copy (avoids cv2 crashes on cloud-synced files).",
    )
    parser.add_argument(
        "--original-mp4",
        default=None,
        help="Path to ORIGINAL MP4 (recorded in save file metadata).",
    )
    parser.add_argument(
        "--video-filename",
        default=None,
        help="Filename to record in save file (default: <name>.mp4)",
    )
    parser.add_argument(
        "--audio-path",
        default=None,
        help="Path to extracted audio (recorded in save file metadata). Default: OUTPUT_DIR/<name>_audio.wav",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    global OUTPUT_DIR, FRAMES_DIR, SCENES_JSON, SAVE_FILE_JSON
    global TRANSCRIPT_JSON, ENERGY_JSON
    global LOCAL_MP4, ORIGINAL_DROPBOX_MP4, VIDEO_FILENAME, AUDIO_PATH

    if args.output_dir:
        OUTPUT_DIR = Path(args.output_dir)
    else:
        OUTPUT_DIR = PROJECT_ROOT / "output" / args.name

    FRAMES_DIR = OUTPUT_DIR / "frames"
    SCENES_JSON = OUTPUT_DIR / "scenes.json"
    SAVE_FILE_JSON = OUTPUT_DIR / f"{args.name}_preprocessed.json"
    TRANSCRIPT_JSON = OUTPUT_DIR / "transcript.json"
    ENERGY_JSON = OUTPUT_DIR / "audio_features.json"

    if args.local_mp4:
        LOCAL_MP4 = Path(args.local_mp4)
    if args.original_mp4:
        ORIGINAL_DROPBOX_MP4 = Path(args.original_mp4)

    VIDEO_FILENAME = args.video_filename if args.video_filename else f"{args.name}.mp4"
    AUDIO_PATH = Path(args.audio_path) if args.audio_path else OUTPUT_DIR / f"{args.name}_audio.wav"

    log(f"Build P3 inputs - frames + scenes + combined save file")
    log(f"Name:        {args.name}")
    log(f"Output dir:  {OUTPUT_DIR}")
    log(f"Local MP4:   {LOCAL_MP4}")
    log(f"Original:    {ORIGINAL_DROPBOX_MP4}")
    log(f"Save file:   {SAVE_FILE_JSON}")

    if not LOCAL_MP4.exists():
        log(f"ERROR: Local MP4 not found at {LOCAL_MP4}")
        log(f"Copy from {ORIGINAL_DROPBOX_MP4} first.")
        return 1

    log(f"Local MP4: {LOCAL_MP4} ({LOCAL_MP4.stat().st_size / 1e6:.0f} MB)")

    try:
        frames = step_1_extract_frames(LOCAL_MP4)
        scenes = step_2_detect_scenes(LOCAL_MP4)
        step_3_assign_scene_ids(frames, scenes)
        step_4_build_save_file(frames, scenes)
    except Exception as e:
        log(f"FATAL ERROR: {type(e).__name__}: {e}")
        log(f"Full traceback:")
        traceback.print_exc(file=sys.stdout)
        sys.stdout.flush()
        return 1

    log("=" * 60)
    log("P3 INPUTS READY")
    log("=" * 60)
    log(f"  P3-A Structure Scanner can now run with full save_file.json")

    return 0


if __name__ == "__main__":
    sys.exit(main())
