"""
Frame sampling module.

Extracts frames from the video at regular intervals.
These frame images are what the visual analysis agents will look at.

Think of it like taking screenshots of the video every few seconds —
the agents can then "see" what's happening at each point.
"""

import os
from pathlib import Path

import cv2

from sixth_sense.models import FrameSample


def sample_frames(
    video_path: str,
    output_dir: str,
    interval_seconds: float = 5.0,
) -> list[FrameSample]:
    """Extract frames from video at regular intervals.

    Args:
        video_path: Path to the input video file
        output_dir: Directory to save frame images
        interval_seconds: How often to grab a frame (default: every 5 seconds).
                         Lower = more frames = more detail but more data.
                         5 seconds is a good balance for 30-60 min videos.

    Returns:
        List of FrameSample objects with timestamps and file paths
    """
    frames_dir = os.path.join(output_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    video_name = Path(video_path).stem
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    frame_interval = int(fps * interval_seconds)

    print(f"  Video: {fps:.1f} FPS, {duration:.0f}s duration")
    print(f"  Sampling every {interval_seconds}s ({int(duration / interval_seconds)} frames)")

    samples = []
    frame_number = 0
    sample_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_number % frame_interval == 0:
            timestamp = frame_number / fps
            frame_filename = f"{video_name}_frame_{sample_count:04d}_{timestamp:.1f}s.jpg"
            frame_path = os.path.join(frames_dir, frame_filename)

            # Save as JPEG (good quality, reasonable file size)
            cv2.imwrite(frame_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 85])

            samples.append(FrameSample(
                timestamp=round(timestamp, 2),
                frame_path=frame_path,
            ))
            sample_count += 1

        frame_number += 1

    cap.release()
    print(f"  Extracted {sample_count} frames to {frames_dir}")

    return samples
