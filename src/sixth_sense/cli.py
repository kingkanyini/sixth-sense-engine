"""
SixthSense CLI.

Usage:
    sixth-sense preprocess <video_path> [options]
    sixth-sense configure <save_file> [options]

This is the command-line interface for running the preprocessing pipeline
and configuring scan parameters before agents deploy.
"""

import os
import json
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.prompt import Prompt, IntPrompt

from sixth_sense.models import ScanConfig, ClipTarget

console = Console()

# ── Clip Target Presets ────────────────────────────────────────────
# These are common clip length targets. Users pick which ones they want.

CLIP_PRESETS = {
    "reels": ClipTarget(min_seconds=15, max_seconds=60, label="Reels / TikTok"),
    "shorts": ClipTarget(min_seconds=60, max_seconds=180, label="YouTube Shorts"),
    "power": ClipTarget(min_seconds=180, max_seconds=300, label="Power Clips (3-5 min)"),
    "youtube": ClipTarget(min_seconds=300, max_seconds=600, label="YouTube (5-10 min)"),
    "highlight": ClipTarget(min_seconds=30, max_seconds=120, label="Highlight Reel (30s-2min)"),
}


@click.group()
@click.version_option(version="0.1.0", prog_name="SixthSense")
def main():
    """SixthSense — 6-agent video intelligence scanner."""
    pass


@main.command()
@click.argument("video_path", type=click.Path(exists=True))
@click.option(
    "--output", "-o",
    default=None,
    help="Output directory (default: ./output/<video_name>/)",
)
@click.option(
    "--frame-interval", "-f",
    default=5.0,
    type=float,
    help="Seconds between frame samples (default: 5.0)",
)
@click.option(
    "--audio-window", "-a",
    default=30.0,
    type=float,
    help="Audio analysis window size in seconds (default: 30.0)",
)
@click.option(
    "--scene-threshold", "-s",
    default=27.0,
    type=float,
    help="Scene detection sensitivity, lower=more sensitive (default: 27.0)",
)
@click.option(
    "--whisper-model", "-w",
    default="large-v3",
    type=click.Choice(["tiny", "base", "small", "medium", "large-v3"]),
    help="Whisper model size (default: large-v3, best accuracy)",
)
@click.option(
    "--cpu",
    is_flag=True,
    default=False,
    help="Run Whisper on CPU instead of GPU (slower but works without CUDA)",
)
@click.option(
    "--skip-transcript",
    is_flag=True,
    default=False,
    help="Skip transcription step (useful for re-running other steps)",
)
def preprocess(video_path, output, frame_interval, audio_window, scene_threshold,
               whisper_model, cpu, skip_transcript):
    """Preprocess a video file for agent analysis.

    Extracts audio, transcribes speech, samples frames, detects scenes,
    and analyzes audio features. Saves a JSON "save file" that the
    scanning agents will consume.

    Example:
        sixth-sense preprocess my_video.mp4
        sixth-sense preprocess my_video.mp4 --whisper-model medium --cpu
    """
    from sixth_sense.preprocessor.pipeline import preprocess_video

    video_path = os.path.abspath(video_path)
    video_name = Path(video_path).stem

    if output is None:
        output = os.path.join(os.getcwd(), "output", video_name)

    preprocess_video(
        video_path=video_path,
        output_dir=output,
        frame_interval=frame_interval,
        audio_window=audio_window,
        scene_threshold=scene_threshold,
        whisper_model=whisper_model,
        whisper_device="cpu" if cpu else "cuda",
        skip_transcript=skip_transcript,
    )


@main.command()
@click.argument("save_file", type=click.Path(exists=True))
def configure(save_file):
    """Configure scan parameters for agent analysis.

    Opens an interactive session to set clip targets, platforms,
    and content focus BEFORE agents deploy. Updates the save file in place.

    Example:
        sixth-sense configure output/my_video/my_video_preprocessed.json
    """
    save_file = os.path.abspath(save_file)

    # Load the save file
    with open(save_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    duration = data.get("duration_seconds", 0)
    minutes = int(duration // 60)
    seconds = int(duration % 60)

    console.print(Panel(
        f"[bold]SixthSense — Mission Briefing[/bold]\n"
        f"Video: {data.get('video_filename', 'unknown')}\n"
        f"Duration: {minutes}m {seconds}s",
        title="Configure Scan",
        border_style="cyan",
    ))

    # Step 1: Pick clip targets
    console.print("\n[bold cyan]Step 1:[/bold cyan] What clip lengths are you scanning for?\n")

    table = Table(show_header=True, header_style="bold")
    table.add_column("#", style="dim", width=3)
    table.add_column("Preset", width=20)
    table.add_column("Length", width=15)

    for i, (key, preset) in enumerate(CLIP_PRESETS.items(), 1):
        mins_min = preset.min_seconds // 60
        secs_min = preset.min_seconds % 60
        mins_max = preset.max_seconds // 60
        secs_max = preset.max_seconds % 60
        length_str = f"{mins_min}:{secs_min:02d} - {mins_max}:{secs_max:02d}"
        table.add_row(str(i), preset.label, length_str)

    table.add_row("6", "Custom", "You define")
    console.print(table)

    choices = Prompt.ask(
        "\nPick target(s) — comma-separated",
        default="1,2",
    )

    selected_targets = []
    preset_keys = list(CLIP_PRESETS.keys())

    for choice in choices.split(","):
        choice = choice.strip()
        if choice == "6":
            # Custom clip target
            min_s = IntPrompt.ask("  Min clip length (seconds)")
            max_s = IntPrompt.ask("  Max clip length (seconds)")
            label = Prompt.ask("  Label for this target", default="Custom")
            selected_targets.append(ClipTarget(
                min_seconds=min_s, max_seconds=max_s, label=label,
            ))
        elif choice.isdigit() and 1 <= int(choice) <= 5:
            idx = int(choice) - 1
            selected_targets.append(CLIP_PRESETS[preset_keys[idx]])

    if not selected_targets:
        console.print("[yellow]No targets selected — using Reels + Shorts as default[/yellow]")
        selected_targets = [CLIP_PRESETS["reels"], CLIP_PRESETS["shorts"]]

    # Step 2: Platforms
    console.print("\n[bold cyan]Step 2:[/bold cyan] Target platforms? (helps agents optimize)\n")
    platforms_input = Prompt.ask(
        "Platforms (comma-separated)",
        default="tiktok, youtube, instagram",
    )
    platforms = [p.strip().lower() for p in platforms_input.split(",") if p.strip()]

    # Step 3: Content focus
    console.print("\n[bold cyan]Step 3:[/bold cyan] What should agents prioritize finding?\n")
    console.print("  Options: teaching, stories, hooks, vulnerability, humor, quotes, energy, all")
    focus_input = Prompt.ask(
        "Content focus (comma-separated)",
        default="all",
    )
    if focus_input.strip().lower() == "all":
        content_focus = ["teaching", "stories", "hooks", "vulnerability", "humor", "quotes", "energy"]
    else:
        content_focus = [f.strip().lower() for f in focus_input.split(",") if f.strip()]

    # Step 4: Max clips per target
    max_clips = IntPrompt.ask(
        "\nMax clips to find per target length",
        default=10,
    )

    # Build the config
    scan_config = ScanConfig(
        clip_targets=selected_targets,
        platforms=platforms,
        content_focus=content_focus,
        max_clips_per_target=max_clips,
    )

    # Save back to the JSON file
    data["scan_config"] = scan_config.model_dump()

    with open(save_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    # Summary
    console.print(Panel(
        f"[bold green]Mission briefing saved![/bold green]\n\n"
        + "".join(
            f"  Target: {t.label} ({t.min_seconds}s - {t.max_seconds}s)\n"
            for t in selected_targets
        )
        + f"\n  Platforms: {', '.join(platforms)}\n"
        + f"  Focus: {', '.join(content_focus)}\n"
        + f"  Max clips/target: {max_clips}\n\n"
        + f"  Updated: {save_file}",
        title="Configuration Complete",
        border_style="green",
    ))


if __name__ == "__main__":
    main()
