#!/usr/bin/env python3
"""Render a CO-FLIP animation from text outputs produced by CO-FLIP_periodic.jl.

Expected input folder layout:
- metadata.txt
- step_0001.txt
- step_0002.txt
- ...

Each step file must contain 5 whitespace-separated columns:
x, y, u, v, |u|
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from matplotlib.animation import FuncAnimation


def read_metadata(metadata_path: Path) -> Dict[str, float]:
    meta: Dict[str, float] = {}
    with metadata_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            key, value = line.split(maxsplit=1)
            meta[key] = float(value)
    return meta


def load_step(step_path: Path, nx: int, ny: int) -> np.ndarray:
    data = np.loadtxt(step_path)
    if data.ndim != 2 or data.shape[1] < 5:
        raise ValueError(f"Unexpected data format in {step_path}")
    if data.shape[0] != nx * ny:
        raise ValueError(
            f"Unexpected number of points in {step_path}: "
            f"got {data.shape[0]}, expected {nx * ny}"
        )

    # Column 5 is |u|. Julia writes points with x varying fastest, then y,
    # which maps naturally to a (ny, nx) image.
    mag = data[:, 4].reshape((ny, nx))
    return mag


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render CO-FLIP animation from per-step text files."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("coflip_output"),
        help="Directory with metadata.txt and step_XXXX.txt files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("co_flip_evolution.mp4"),
        help="Output animation file (e.g. .mp4 or .gif).",
    )
    parser.add_argument("--fps", type=int, default=10, help="Frames per second.")
    parser.add_argument(
        "--cmap", type=str, default="viridis", help="Matplotlib colormap name."
    )
    parser.add_argument(
        "--vmin",
        type=float,
        default=None,
        help="Optional fixed lower color limit.",
    )
    parser.add_argument(
        "--vmax",
        type=float,
        default=None,
        help="Optional fixed upper color limit.",
    )
    args = parser.parse_args()

    metadata_path = args.input_dir / "metadata.txt"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    meta = read_metadata(metadata_path)
    nx = int(meta["nx_plot"])
    ny = int(meta["ny_plot"])
    n_steps = int(meta["n_steps"])
    dt = float(meta["dt"])
    lx = float(meta["Lx"])
    ly = float(meta["Ly"])

    step_files = [args.input_dir / f"step_{i:04d}.txt" for i in range(1, n_steps + 1)]
    missing = [p for p in step_files if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing step files, first missing: {missing[0]}")

    print(f"Reading {n_steps} frames from {args.input_dir} ...")
    frames = [load_step(p, nx, ny) for p in step_files]

    if args.vmin is None:
        vmin = min(float(np.min(f)) for f in frames)
    else:
        vmin = args.vmin

    if args.vmax is None:
        vmax = max(float(np.max(f)) for f in frames)
    else:
        vmax = args.vmax

    fig, ax = plt.subplots(figsize=(6, 6), dpi=120)
    im = ax.imshow(
        frames[0],
        origin="lower",
        extent=[0.0, lx, 0.0, ly],
        cmap=args.cmap,
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
        aspect="equal",
    )
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("|u|")
    title = ax.set_title(f"CO-FLIP Velocity |u|: t = {dt:.2f}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    def update(frame_idx: int):
        im.set_data(frames[frame_idx])
        title.set_text(f"CO-FLIP Velocity |u|: t = {(frame_idx + 1) * dt:.2f}")
        return (im, title)

    anim = FuncAnimation(fig, update, frames=n_steps, interval=1000 / args.fps, blit=False)

    output_path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ext = output_path.suffix.lower()
    ffmpeg_available = animation.writers.is_available("ffmpeg")

    # Keep writer and extension aligned to avoid Pillow attempting to write .mp4.
    if ext == ".gif":
        writer = "pillow"
        save_path = output_path
    elif ext == ".mp4":
        if ffmpeg_available:
            writer = "ffmpeg"
            save_path = output_path
        else:
            save_path = output_path.with_suffix(".gif")
            writer = "pillow"
            print(
                "ffmpeg is unavailable; saving GIF instead at "
                f"{save_path}"
            )
    else:
        if ffmpeg_available:
            writer = "ffmpeg"
            save_path = output_path
        else:
            save_path = output_path.with_suffix(".gif")
            writer = "pillow"
            print(
                "ffmpeg is unavailable; output extension is not GIF. "
                f"Saving GIF instead at {save_path}"
            )

    print(f"Saving animation to {save_path} with writer={writer} ...")
    anim.save(save_path, writer=writer, fps=args.fps)

    print("Done.")


if __name__ == "__main__":
    main()
