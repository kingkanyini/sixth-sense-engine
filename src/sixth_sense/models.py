"""
Data models for SixthSense.

These Pydantic models define the contract between the preprocessing pipeline
and the 6 scanning agents. The preprocessor WRITES this data, agents READ it.

Think of it like a save file — the preprocessor creates it once, and every
agent loads the same save file to do their analysis.
"""

from pydantic import BaseModel
from typing import Optional


# ── Scan Configuration ─────────────────────────────────────────────

class ClipTarget(BaseModel):
    """A target clip length to scan for.

    The agents use these targets to calibrate what they're looking for.
    A 30-second clip needs a quick hook + punchline.
    A 5-minute clip needs a full story arc.
    """
    min_seconds: int       # minimum clip length
    max_seconds: int       # maximum clip length
    label: str             # e.g. "Reels/TikTok", "YouTube Short", "Power Clip"
    priority: int = 1      # higher = agents weight this target more (1-3)


class ScanConfig(BaseModel):
    """Configuration that tells agents WHAT to scan for.

    This gets set BEFORE scanning begins. It answers:
    - How long should the clips be?
    - What platforms are we targeting?
    - Any specific content to look for?

    Think of it like the mission briefing before agents deploy.
    """
    clip_targets: list[ClipTarget]                # what clip lengths to find
    platforms: list[str] = []                      # e.g. ["tiktok", "youtube", "instagram"]
    content_focus: list[str] = []                  # e.g. ["teaching", "stories", "hooks"]
    max_clips_per_target: int = 10                 # max clips to find per target length
    min_clip_score: float = 0.5                    # minimum quality score (0-1) to include


# ── Transcript Models ──────────────────────────────────────────────

class TranscriptWord(BaseModel):
    """A single word with its exact timestamp in the video."""
    word: str
    start: float       # seconds from video start
    end: float         # seconds from video start
    confidence: float  # 0.0 to 1.0, how sure the transcriber is


class TranscriptSegment(BaseModel):
    """A chunk of speech (usually a sentence or phrase)."""
    start: float
    end: float
    text: str
    speaker: Optional[str] = None  # speaker ID if diarization is available
    words: list[TranscriptWord] = []


# ── Frame Models ───────────────────────────────────────────────────

class FrameSample(BaseModel):
    """A single frame extracted from the video."""
    timestamp: float     # seconds from video start
    frame_path: str      # path to the saved image file
    scene_id: int = -1   # which scene this frame belongs to (-1 = unassigned)


# ── Audio Feature Models ───────────────────────────────────────────

class AudioFeatures(BaseModel):
    """Audio characteristics for a time window in the video.

    Think of this like the "energy meter" in a game — it tells you
    how loud, fast, and dynamic each section of the video is.
    """
    start: float
    end: float
    energy: float           # volume/loudness (RMS energy, normalized 0-1)
    tempo: float            # estimated speaking pace or musical tempo (BPM)
    pitch_mean: float       # average pitch in Hz
    pitch_variance: float   # how much pitch varies (monotone vs expressive)
    silence_ratio: float    # percentage of this window that's silence (0.0-1.0)


# ── Scene Models ───────────────────────────────────────────────────

class Scene(BaseModel):
    """A detected scene — a continuous visual segment between cuts/transitions."""
    scene_id: int
    start: float   # seconds
    end: float     # seconds
    duration: float


# ── The Main Save File ─────────────────────────────────────────────

class PreprocessedData(BaseModel):
    """The complete preprocessed output for a video.

    This is the "save file" that gets created once by the preprocessing
    pipeline and then loaded by all 6 agents for their analysis.
    Every field is immutable after creation — agents read, never write.
    """
    # Source info
    video_path: str
    video_filename: str
    duration_seconds: float

    # Audio
    audio_path: str  # path to extracted WAV file

    # Transcript (populated by WhisperX)
    transcript: list[TranscriptSegment] = []

    # Sampled frames (populated by OpenCV)
    frames: list[FrameSample] = []

    # Audio features per time window (populated by librosa)
    audio_features: list[AudioFeatures] = []

    # Scene boundaries (populated by PySceneDetect)
    scenes: list[Scene] = []

    # Scan configuration (set by user before agents deploy)
    scan_config: Optional[ScanConfig] = None

    # Processing metadata
    frame_interval_seconds: float = 5.0  # how often frames were sampled
    audio_window_seconds: float = 30.0   # size of audio analysis windows
    preprocessed_at: str = ""             # ISO timestamp of when this was created
