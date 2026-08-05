"""
Scene detection module.

Detects scene boundaries (cuts, transitions) in the video.
A "scene" is a continuous visual segment — when the camera angle changes,
a new graphic appears, or there's a visual transition, that's a new scene.

This structural data helps agents understand the video's natural segments.
Think of scenes like "chapters" in a game — they give structure to the content.
"""

from scenedetect import detect, ContentDetector

from sixth_sense.models import Scene


def detect_scenes(
    video_path: str,
    threshold: float = 27.0,
) -> list[Scene]:
    """Detect scene boundaries in a video.

    Uses content-aware detection — it looks for significant changes between
    consecutive frames (color shifts, movement, new graphics appearing).

    Args:
        video_path: Path to the input video file
        threshold: Sensitivity for scene detection (lower = more sensitive).
                  27.0 is a good default. Lower it to 20-22 if you're
                  missing cuts, raise it to 30+ if detecting too many.

    Returns:
        List of Scene objects with start/end timestamps
    """
    print(f"  Detecting scenes (threshold={threshold})...")

    scene_list = detect(video_path, ContentDetector(threshold=threshold))

    scenes = []
    for i, scene in enumerate(scene_list):
        start = scene[0].get_seconds()
        end = scene[1].get_seconds()
        scenes.append(Scene(
            scene_id=i,
            start=round(start, 2),
            end=round(end, 2),
            duration=round(end - start, 2),
        ))

    print(f"  Detected {len(scenes)} scenes")

    if scenes:
        durations = [s.duration for s in scenes]
        avg_duration = sum(durations) / len(durations)
        shortest = min(durations)
        longest = max(durations)
        print(f"  Scene durations: avg={avg_duration:.1f}s, "
              f"shortest={shortest:.1f}s, longest={longest:.1f}s")

    return scenes
