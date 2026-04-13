#!/usr/bin/env python3
"""Plot convergence history and estimate slopes from mesh refinement data.

Input file format (whitespace separated, optional comment header):
    nel_x nel_y nparticles p2g_err l2_err proj_err

Example:
    python tests_output/plot_mesh_convergence.py --input tests_output/coflip_p2g_mesh_convergence.txt --output tests_output/coflip_p2g_mesh_convergence.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot convergence errors and report log-log slopes."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("tests_output/coflip_p2g_mesh_convergence.txt"),
        help="Path to convergence text file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("tests_output/coflip_p2g_mesh_convergence.png"),
        help="Path to output plot image.",
    )
    return parser.parse_args()


def load_convergence_table(
    path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    data = np.loadtxt(path, comments="#")
    if data.ndim == 1:
        data = data[None, :]
    if data.shape[1] < 6:
        raise ValueError(
            f"Expected at least 6 columns in {path}, found {data.shape[1]}"
        )
    nel_x = data[:, 0]
    nel_y = data[:, 1]
    nparticles = data[:, 2]
    p2g_err = data[:, 3]
    l2_err = data[:, 4]
    proj_err = data[:, 5]
    return nel_x, nel_y, nparticles, p2g_err, l2_err, proj_err


def pairwise_slopes(h: np.ndarray, err: np.ndarray) -> np.ndarray:
    # Local order between successive refinements: p = d(log e) / d(log h)
    return np.diff(np.log(err)) / np.diff(np.log(h))


def fit_slope(h: np.ndarray, err: np.ndarray) -> tuple[float, float]:
    x = np.log(h)
    y = np.log(err)
    slope, intercept = np.polyfit(x, y, 1)
    return slope, intercept


def main() -> None:
    args = parse_args()
    nel_x, nel_y, nparticles, p2g_err, l2_err, proj_err = load_convergence_table(
        args.input
    )

    # Characteristic mesh size for square grids: h ~ 1 / nel_x.
    h = 1.0 / nel_x

    p2g_slope, p2g_intercept = fit_slope(h, p2g_err)
    l2_slope, l2_intercept = fit_slope(h, l2_err)
    proj_slope, proj_intercept = fit_slope(h, proj_err)

    p2g_pairwise = pairwise_slopes(h, p2g_err)
    l2_pairwise = pairwise_slopes(h, l2_err)
    proj_pairwise = pairwise_slopes(h, proj_err)

    fig, ax = plt.subplots(figsize=(8, 5.5))

    ax.loglog(
        h,
        p2g_err,
        "o-",
        linewidth=2,
        markersize=6,
        label="P2G vs proxy form",
    )
    ax.loglog(
        h,
        l2_err,
        "s-",
        linewidth=2,
        markersize=6,
        label="L2 projection vs proxy form",
    )
    ax.loglog(
        h,
        proj_err,
        "^-",
        linewidth=2,
        markersize=6,
        label="Divergence projection vs proxy form",
    )

    # Overlay fitted trend lines in log space.
    h_fit = np.linspace(h.min(), h.max(), 200)
    p2g_fit = np.exp(p2g_intercept) * h_fit**p2g_slope
    l2_fit = np.exp(l2_intercept) * h_fit**l2_slope
    proj_fit = np.exp(proj_intercept) * h_fit**proj_slope
    ax.loglog(
        h_fit,
        p2g_fit,
        "--",
        linewidth=1.5,
        label=f"P2G fit slope = {p2g_slope:.3f}",
    )
    ax.loglog(
        h_fit,
        l2_fit,
        "--",
        linewidth=1.5,
        label=f"L2 fit slope = {l2_slope:.3f}",
    )
    ax.loglog(
        h_fit,
        proj_fit,
        "--",
        linewidth=1.5,
        label=f"Divergence fit slope = {proj_slope:.3f}",
    )

    ax.set_xlabel("Mesh size h (1/nel_x)")
    ax.set_ylabel("Error")
    ax.set_title("Mesh Convergence of Proxy-Form Error Metrics")
    ax.grid(True, which="both", linestyle=":", alpha=0.6)
    ax.legend()

    # Most refined mesh at right is often preferred in convergence plots.
    ax.invert_xaxis()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.output, dpi=200)
    plt.close(fig)

    print(f"Loaded {len(h)} refinement levels from {args.input}")
    print(f"Saved convergence plot to {args.output}")
    print()
    print("Global fitted slopes (log(error) vs log(h)):")
    print(f"  P2G error slope:        {p2g_slope:.6f}")
    print(f"  L2 projection slope:    {l2_slope:.6f}")
    print(f"  Divergence slope:       {proj_slope:.6f}")
    print()
    print("Pairwise slopes between successive refinements:")
    for i in range(len(p2g_pairwise)):
        print(
            "  "
            f"level {i} -> {i+1} | "
            f"nel: ({int(nel_x[i])},{int(nel_y[i])}) -> ({int(nel_x[i+1])},{int(nel_y[i+1])}) | "
            f"nparticles: {int(nparticles[i])} -> {int(nparticles[i+1])} | "
            f"p2g slope = {p2g_pairwise[i]:.6f}, "
            f"l2 slope = {l2_pairwise[i]:.6f}, "
            f"divergence slope = {proj_pairwise[i]:.6f}"
        )


if __name__ == "__main__":
    main()
