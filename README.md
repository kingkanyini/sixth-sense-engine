# sixth-sense

Video preprocessing engine for the SixthSense skill family. Takes a long-form video and produces a single `_preprocessed.json` — transcript with word-level timestamps, audio-energy map, sampled frames, and scene boundaries — that the scanning agents read.

This repo is the **engine only**. It does the ffmpeg / Whisper / scene-detection work. The agent prompts, scoring rubrics, and consensus logic live in the `sixth-sense` Claude Code skill.

## Install

```bash
uv tool install git+https://github.com/kingkanyini/sixth-sense-engine
```

Verify:

```bash
sixth-sense --help
```

> **Do not run `pip install sixth-sense`.** That name belongs to an unrelated package on PyPI and will install a stranger's code. Install from this git URL only.

## Requirements

| | |
|---|---|
| Python | 3.13+ |
| ffmpeg | on `PATH` — `ffmpeg -version` must work |
| GPU | Optional. NVIDIA + CUDA 12.4 accelerates transcription; CPU and Apple Silicon (MPS) both work, slower. |

Platform notes:

- **Windows + NVIDIA** — installs CUDA wheels automatically.
- **macOS / Linux / non-NVIDIA Windows** — resolves CPU or MPS wheels from PyPI. No CUDA download.

The transcription model (`large-v3`) downloads on first run and is several GB. Budget disk space and time for that first invocation.

## Usage

```bash
# Full preprocess
sixth-sense preprocess "path/to/video.mp4"

# Outputs land in ./output/<video_name>/
```

Two scripts handle the checkpoint-first path, which survives a crash mid-transcription:

```bash
# 1. Transcript + audio features (checkpoints transcript.json the moment it exists)
uv run python scripts/decoupled_preprocess.py --name "my-video"

# 2. Frames + scenes + combined save file
uv run python scripts/build_p3_inputs.py --name "my-video"
```

Run these from the project root. Both accept `--output-dir` and related overrides; see `--help`.

## Output

`output/<name>/` contains:

| File | What |
|---|---|
| `transcript.json` | Segments with word-level timestamps and confidence |
| `transcript_timestamped.txt` | Human-readable, `[MM:SS] text` per line |
| `audio_features.json` | Energy, tempo, pitch per 30s window |
| `energy_map.txt` | Readable energy map |
| `scenes.json` | Scene boundaries |
| `frames/` | One JPEG every 5s |
| `<name>_preprocessed.json` | Combined save file — this is what the agents read |

`output/` is gitignored. It holds source-video derivatives and should never be committed.

## License

MIT — see [LICENSE](LICENSE).
