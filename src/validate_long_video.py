"""
src/validate_long_video.py
==========================
Orchestrates the complete VisionClass pipeline for long-duration validation.
Measures runtime performance (Processing FPS, Real-time Factor) and memory safety.

Usage:
  python src/validate_long_video.py --video "path/to/long_video.mp4"
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import cv2
import psutil

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def _get_vram_mb() -> float:
    if HAS_TORCH and torch.cuda.is_available():
        try:
            return torch.cuda.memory_allocated() / (1024 * 1024)
        except Exception:
            return 0.0
    return 0.0


def _get_ram_mb() -> float:
    process = psutil.Process(os.getpid())
    # Note: subprocess memory won't be captured accurately by just getting THIS process's memory.
    # To be accurate, we should capture the total system memory or track children.
    # We will track the child process memory.
    return 0.0 # Handled in monitor


def monitor_memory(proc: subprocess.Popen, stop_event: threading.Event, stats: dict) -> None:
    try:
        ps_proc = psutil.Process(proc.pid)
    except psutil.NoSuchProcess:
        return

    peak_ram = 0.0
    peak_vram = 0.0

    while not stop_event.is_set():
        try:
            # RAM of subprocess + children
            ram = ps_proc.memory_info().rss
            for child in ps_proc.children(recursive=True):
                ram += child.memory_info().rss

            ram_mb = ram / (1024 * 1024)
            if ram_mb > peak_ram:
                peak_ram = ram_mb

            vram_mb = _get_vram_mb()
            if vram_mb > peak_vram:
                peak_vram = vram_mb

        except (psutil.NoSuchProcess, psutil.AccessDenied):
            break

        time.sleep(0.5)

    stats["peak_ram_mb"] = peak_ram
    stats["peak_vram_mb"] = peak_vram


def get_video_info(video_path: Path) -> tuple[int, float]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return 0, 0.0
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return frames, fps


def main() -> None:
    parser = argparse.ArgumentParser(description="VisionClass Long Video Validator")
    parser.add_argument("--video", type=str, required=True, help="Path to input video")
    parser.add_argument("--outdir", type=str, default="data/long_validation", help="Output directory")
    args = parser.parse_args()

    video_path = Path(args.video).resolve()
    outdir = Path(args.outdir).resolve()

    if not video_path.exists():
        print(f"ERROR: Video not found: {video_path}")
        sys.exit(1)

    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "videos").mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("VISIONCLASS - LONG VIDEO VALIDATION")
    print("=" * 60)
    print(f"Input: {video_path}")

    total_frames, video_fps = get_video_info(video_path)
    if video_fps <= 0:
        video_fps = 25.0

    video_duration = total_frames / video_fps if video_fps else 0
    print(f"Frames: {total_frames}")
    print(f"FPS:    {video_fps:.2f}")
    print(f"Length: {video_duration:.2f} seconds")
    print("-" * 60)

    # Set up environment variables to redirect outputs
    env = os.environ.copy()
    env["VISIONCLASS_VIDEO"] = str(video_path)

    history_video = outdir / "videos" / f"{video_path.stem}_history.mp4"
    history_csv = outdir / f"{video_path.stem}_history.csv"
    summary_csv = outdir / f"{video_path.stem}_summary.csv"
    analytics_md = outdir / f"{video_path.stem}_analytics.md"
    dashboard_html = outdir / f"{video_path.stem}_dashboard.html"

    env["VISIONCLASS_HISTORY_VIDEO_OUTPUT"] = str(history_video)
    env["VISIONCLASS_HISTORY_CSV"] = str(history_csv)
    env["VISIONCLASS_SUMMARY_CSV"] = str(summary_csv)

    # 1. Run Pipeline
    print("[1/3] Running Behavior History Pipeline...")
    start_time = time.time()

    python_exe = sys.executable
    cmd = [python_exe, "src/behavior_history.py"]

    # Start subprocess
    proc = subprocess.Popen(cmd, env=env)

    # Monitor memory in background
    stop_event = threading.Event()
    mem_stats = {"peak_ram_mb": 0.0, "peak_vram_mb": 0.0}

    # Get initial RAM
    init_ram = psutil.virtual_memory().used / (1024 * 1024)
    mem_stats["start_ram_mb"] = init_ram

    monitor_thread = threading.Thread(target=monitor_memory, args=(proc, stop_event, mem_stats))
    monitor_thread.start()

    # Wait for completion
    proc.wait()

    stop_event.set()
    monitor_thread.join()

    end_time = time.time()

    if proc.returncode != 0:
        print("ERROR: Pipeline failed.")
        sys.exit(proc.returncode)

    processing_time = end_time - start_time
    processing_fps = total_frames / processing_time if processing_time > 0 else 0
    rtf = processing_fps / video_fps if video_fps > 0 else 0

    end_ram = psutil.virtual_memory().used / (1024 * 1024)
    mem_stats["end_ram_mb"] = end_ram

    # 2. Run Analytics
    print("[2/3] Generating Analytics Report...")
    subprocess.run([
        python_exe, "src/generate_analytics_report.py",
        "--history", str(history_csv),
        "--summary", str(summary_csv),
        "--output", str(analytics_md)
    ], env=env, check=True)

    # 3. Build Dashboard
    print("[3/3] Building Dashboard...")
    subprocess.run([
        python_exe, "src/build_dashboard.py",
        "--history", str(history_csv),
        "--summary", str(summary_csv),
        "--output", str(dashboard_html)
    ], env=env, check=True)

    print("-" * 60)
    print("VALIDATION COMPLETE")
    print("-" * 60)
    print("PERFORMANCE METRICS:")
    print(f"Total time     : {processing_time:.2f} seconds")
    print(f"Processing FPS : {processing_fps:.2f} FPS")
    print(f"Video FPS      : {video_fps:.2f} FPS")
    print(f"Real-Time Fact.: {rtf:.2f}x")
    print()
    print("MEMORY SAFETY:")
    print(f"System RAM (Start) : {mem_stats['start_ram_mb']:.0f} MB")
    print(f"Pipeline Peak RAM  : {mem_stats.get('peak_ram_mb', 0):.0f} MB")
    print(f"System RAM (End)   : {mem_stats['end_ram_mb']:.0f} MB")
    if HAS_TORCH and torch.cuda.is_available():
        print(f"Peak VRAM          : {mem_stats.get('peak_vram_mb', 0):.0f} MB")

    print("-" * 60)
    print("OUTPUTS:")
    print(f"History Video : {history_video}")
    print(f"History CSV   : {history_csv}")
    print(f"Summary CSV   : {summary_csv}")
    print(f"Analytics     : {analytics_md}")
    print(f"Dashboard     : {dashboard_html}")
    print("=" * 60)

if __name__ == "__main__":
    main()
