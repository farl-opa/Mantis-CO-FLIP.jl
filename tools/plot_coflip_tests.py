#!/usr/bin/env python3
"""Plot diagnostics for CO-FLIP convergence test outputs.

Reads these files from --input-dir (default: tests_output):
- coflip_projection_convergence.txt
- coflip_p2g_mesh_convergence.txt
- coflip_p2g_particle_convergence.txt

Writes figures to --output-dir (default: tests_output/plots).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


def read_table(path: Path) -> Tuple[List[str], np.ndarray]:
    """Read a whitespace table with a header line starting with '#'."""
    if not path.exists():
        raise FileNotFoundError(f"Missing input file: {path}")

    header_cols = None
    rows: List[List[float]] = []

    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if line.startswith("#"):
                if "last_column" in line:
                    continue
                if header_cols is None:
                    header_cols = line.lstrip("#").strip().split()
                continue

            parts = line.split()
            values = []
            for p in parts:
                if p.lower() == "nan":
                    values.append(np.nan)
                else:
                    values.append(float(p))
            rows.append(values)

    if header_cols is None:
        raise ValueError(f"No header found in {path}")
    if not rows:
        raise ValueError(f"No data rows found in {path}")

    data = np.array(rows, dtype=float)
    if data.shape[1] != len(header_cols):
        raise ValueError(
            f"Column mismatch in {path}: header has {len(header_cols)}, data has {data.shape[1]}"
        )

    return header_cols, data


def as_dict(cols: List[str], data: np.ndarray) -> Dict[str, np.ndarray]:
    return {c: data[:, i] for i, c in enumerate(cols)}


def finite_mask(*arrays: np.ndarray) -> np.ndarray:
    mask = np.ones_like(arrays[0], dtype=bool)
    for arr in arrays:
        mask &= np.isfinite(arr)
    return mask


def save_projection_plots(d: Dict[str, np.ndarray], out_dir: Path) -> None:
    nx = d["nx"]
    div_l2 = d["div_l2"]
    vel_l2 = d["vel_l2"]
    vel_rate = d.get("vel_rate", np.full_like(nx, np.nan))

    fig, ax = plt.subplots(figsize=(7, 5))
    m = finite_mask(nx, div_l2)
    ax.semilogy(nx[m], div_l2[m], "o-", lw=2, ms=6, label="div_l2")
    ax.set_xlabel("nx (= ny)")
    ax.set_ylabel("Divergence L2")
    ax.set_title("Projection Divergence Defect")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "projection_divergence_l2.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    m = finite_mask(nx, vel_l2)
    ax.loglog(nx[m], vel_l2[m], "o-", lw=2, ms=6, label="vel_l2")

    # Add an empirical O(h) guide using the first point.
    if np.count_nonzero(m) >= 2:
        xref = nx[m]
        yref = vel_l2[m][0] * (xref[0] / xref)
        ax.loglog(xref, yref, "--", lw=1.5, label="O(h)")

    for i in range(len(nx) - 1):
        if np.isfinite(vel_rate[i]):
            ax.annotate(
                f"r={vel_rate[i]:.2f}",
                xy=(nx[i + 1], vel_l2[i + 1]),
                xytext=(5, 4),
                textcoords="offset points",
                fontsize=8,
            )

    ax.set_xlabel("nx (= ny)")
    ax.set_ylabel("Velocity L2")
    ax.set_title("Projection Velocity Convergence")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "projection_velocity_convergence.png", dpi=180)
    plt.close(fig)


def save_p2g_mesh_plots(d: Dict[str, np.ndarray], out_dir: Path) -> None:
    nx = d["nx"]

    fig, ax = plt.subplots(figsize=(7, 5))
    for key, label in [
        ("vel_l2", "vel_l2"),
        ("coeff_rel_l2", "coeff_rel_l2"),
        ("sampled_ref_rel_l2", "sampled_ref_rel_l2"),
    ]:
        if key in d:
            m = finite_mask(nx, d[key])
            ax.loglog(nx[m], d[key][m], "o-", lw=2, ms=5, label=label)

    ax.set_xlabel("nx (= ny)")
    ax.set_ylabel("Relative / L2 Error")
    ax.set_title("P2G Mesh Convergence")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "p2g_mesh_errors.png", dpi=180)
    plt.close(fig)

    if "rel_resid" in d and "rel_normal_resid" in d:
        fig, ax = plt.subplots(figsize=(7, 5))
        m1 = finite_mask(nx, d["rel_resid"])
        m2 = finite_mask(nx, d["rel_normal_resid"])
        ax.semilogy(nx[m1], d["rel_resid"][m1], "o-", lw=2, ms=5, label="rel_resid")
        ax.semilogy(
            nx[m2],
            d["rel_normal_resid"][m2],
            "s-",
            lw=2,
            ms=5,
            label="rel_normal_resid",
        )
        ax.set_xlabel("nx (= ny)")
        ax.set_ylabel("Residual")
        ax.set_title("P2G Mesh Residual Diagnostics")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / "p2g_mesh_residuals.png", dpi=180)
        plt.close(fig)


def save_p2g_ppc_plots(d: Dict[str, np.ndarray], out_dir: Path) -> None:
    ppc = d["ppc"]

    fig, ax = plt.subplots(figsize=(7, 5))

    def add_errbar(mean_key: str, std_key: str, label: str, marker: str) -> None:
        if mean_key not in d:
            return
        y = d[mean_key]
        yerr = d[std_key] if std_key in d else np.zeros_like(y)
        m = finite_mask(ppc, y, yerr)
        ax.errorbar(
            ppc[m],
            y[m],
            yerr=yerr[m],
            fmt=marker,
            lw=1.8,
            ms=5,
            capsize=3,
            label=label,
        )

    add_errbar("vel_l2_mean", "vel_l2_std", "vel_l2_mean", "o-")
    add_errbar("coeff_rel_l2_mean", "coeff_rel_l2_std", "coeff_rel_l2_mean", "s-")
    add_errbar(
        "sampled_ref_rel_l2_mean",
        "sampled_ref_rel_l2_std",
        "sampled_ref_rel_l2_mean",
        "^-",
    )

    ax.set_xscale("log", base=2)
    ax.set_xlabel("PPC")
    ax.set_ylabel("Error")
    ax.set_title("P2G PPC Error Trends")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "p2g_ppc_errors_with_std.png", dpi=180)
    plt.close(fig)

    if "rel_resid_mean" in d and "rel_normal_resid_mean" in d:
        fig, ax = plt.subplots(figsize=(7, 5))
        m1 = finite_mask(ppc, d["rel_resid_mean"])
        m2 = finite_mask(ppc, d["rel_normal_resid_mean"])
        ax.semilogy(
            ppc[m1],
            d["rel_resid_mean"][m1],
            "o-",
            lw=2,
            ms=5,
            label="rel_resid_mean",
        )
        ax.semilogy(
            ppc[m2],
            d["rel_normal_resid_mean"][m2],
            "s-",
            lw=2,
            ms=5,
            label="rel_normal_resid_mean",
        )
        ax.set_xscale("log", base=2)
        ax.set_xlabel("PPC")
        ax.set_ylabel("Residual")
        ax.set_title("P2G PPC Residual Diagnostics")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / "p2g_ppc_residuals.png", dpi=180)
        plt.close(fig)


def save_summary_figure(
    projection: Dict[str, np.ndarray],
    p2g_mesh: Dict[str, np.ndarray],
    p2g_ppc: Dict[str, np.ndarray],
    out_dir: Path,
) -> None:
    """Create a compact one-page diagnostics overview."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))

    # Panel 1: projection divergence
    ax = axes[0, 0]
    nx = projection["nx"]
    div_l2 = projection["div_l2"]
    m = finite_mask(nx, div_l2)
    ax.semilogy(nx[m], div_l2[m], "o-", lw=2, ms=5)
    ax.set_title("Projection: div_l2")
    ax.set_xlabel("nx")
    ax.set_ylabel("Divergence L2")
    ax.grid(True, alpha=0.3)

    # Panel 2: projection velocity convergence
    ax = axes[0, 1]
    vel_l2 = projection["vel_l2"]
    m = finite_mask(nx, vel_l2)
    ax.loglog(nx[m], vel_l2[m], "o-", lw=2, ms=5, label="vel_l2")
    if np.count_nonzero(m) >= 2:
        xref = nx[m]
        yref = vel_l2[m][0] * (xref[0] / xref)
        ax.loglog(xref, yref, "--", lw=1.4, label="O(h)")
    ax.set_title("Projection: velocity error")
    ax.set_xlabel("nx")
    ax.set_ylabel("Velocity L2")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)

    # Panel 3: mesh errors
    ax = axes[0, 2]
    nxm = p2g_mesh["nx"]
    for key, label in [
        ("vel_l2", "vel_l2"),
        ("coeff_rel_l2", "coeff_rel_l2"),
        ("sampled_ref_rel_l2", "sampled_ref_rel_l2"),
    ]:
        if key in p2g_mesh:
            m = finite_mask(nxm, p2g_mesh[key])
            ax.loglog(nxm[m], p2g_mesh[key][m], "o-", lw=1.8, ms=4, label=label)
    ax.set_title("P2G mesh: errors")
    ax.set_xlabel("nx")
    ax.set_ylabel("Error")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)

    # Panel 4: mesh residuals
    ax = axes[1, 0]
    if "rel_resid" in p2g_mesh:
        m = finite_mask(nxm, p2g_mesh["rel_resid"])
        ax.semilogy(nxm[m], p2g_mesh["rel_resid"][m], "o-", lw=2, ms=5, label="rel_resid")
    if "rel_normal_resid" in p2g_mesh:
        m = finite_mask(nxm, p2g_mesh["rel_normal_resid"])
        ax.semilogy(
            nxm[m],
            p2g_mesh["rel_normal_resid"][m],
            "s-",
            lw=2,
            ms=5,
            label="rel_normal_resid",
        )
    ax.set_title("P2G mesh: residuals")
    ax.set_xlabel("nx")
    ax.set_ylabel("Residual")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)

    # Panel 5: PPC errors
    ax = axes[1, 1]
    ppc = p2g_ppc["ppc"]
    for mean_key, std_key, label, marker in [
        ("vel_l2_mean", "vel_l2_std", "vel_l2", "o-"),
        ("coeff_rel_l2_mean", "coeff_rel_l2_std", "coeff_rel_l2", "s-"),
        (
            "sampled_ref_rel_l2_mean",
            "sampled_ref_rel_l2_std",
            "sampled_ref_rel_l2",
            "^-",
        ),
    ]:
        if mean_key in p2g_ppc:
            y = p2g_ppc[mean_key]
            yerr = p2g_ppc[std_key] if std_key in p2g_ppc else np.zeros_like(y)
            m = finite_mask(ppc, y, yerr)
            ax.errorbar(ppc[m], y[m], yerr=yerr[m], fmt=marker, lw=1.6, ms=4, capsize=2, label=label)
    ax.set_xscale("log", base=2)
    ax.set_title("P2G PPC: errors")
    ax.set_xlabel("ppc")
    ax.set_ylabel("Error")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)

    # Panel 6: PPC residuals
    ax = axes[1, 2]
    if "rel_resid_mean" in p2g_ppc:
        m = finite_mask(ppc, p2g_ppc["rel_resid_mean"])
        ax.semilogy(
            ppc[m],
            p2g_ppc["rel_resid_mean"][m],
            "o-",
            lw=2,
            ms=5,
            label="rel_resid_mean",
        )
    if "rel_normal_resid_mean" in p2g_ppc:
        m = finite_mask(ppc, p2g_ppc["rel_normal_resid_mean"])
        ax.semilogy(
            ppc[m],
            p2g_ppc["rel_normal_resid_mean"][m],
            "s-",
            lw=2,
            ms=5,
            label="rel_normal_resid_mean",
        )
    ax.set_xscale("log", base=2)
    ax.set_title("P2G PPC: residuals")
    ax.set_xlabel("ppc")
    ax.set_ylabel("Residual")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)

    fig.suptitle("CO-FLIP Convergence Diagnostics Summary", fontsize=14)
    fig.tight_layout(rect=[0, 0.02, 1, 0.97])
    fig.savefig(out_dir / "coflip_convergence_summary.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot CO-FLIP test outputs")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("tests_output"),
        help="Directory containing convergence .txt files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("tests_output") / "plots",
        help="Directory where plots are saved",
    )
    args = parser.parse_args()

    input_dir = args.input_dir
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    projection_cols, projection_data = read_table(input_dir / "coflip_projection_convergence.txt")
    p2g_mesh_cols, p2g_mesh_data = read_table(input_dir / "coflip_p2g_mesh_convergence.txt")
    p2g_ppc_cols, p2g_ppc_data = read_table(input_dir / "coflip_p2g_particle_convergence.txt")

    save_projection_plots(as_dict(projection_cols, projection_data), out_dir)
    save_p2g_mesh_plots(as_dict(p2g_mesh_cols, p2g_mesh_data), out_dir)
    save_p2g_ppc_plots(as_dict(p2g_ppc_cols, p2g_ppc_data), out_dir)
    save_summary_figure(
        as_dict(projection_cols, projection_data),
        as_dict(p2g_mesh_cols, p2g_mesh_data),
        as_dict(p2g_ppc_cols, p2g_ppc_data),
        out_dir,
    )

    print("Saved plots to", out_dir)


if __name__ == "__main__":
    main()
