#!/usr/bin/env python3
"""
Visualise the lid-driven-cavity validation results.

Reads the CSV files written by `test_lid_driven_cavity` in this folder
and produces five figures plus a refreshed summary printed to stdout:

  - ghia_centerline_u.png   : u(y) at x = L/2 — dense CO-FLIP curve overlaid
                              with the 17 Ghia (1982) reference points
                              (interior + wall, with wall points greyed out).
  - ghia_centerline_v.png   : v(x) at y = L/2 — same overlay.
  - ghia_per_point_error.png: pointwise abs and rel errors at the 17 Ghia
                              y/L (left subplot) and x/L (right subplot)
                              points. Wall endpoints flagged.
  - steady_state.png        : ||Δu||_M / ||u||_M against t on log-y, so
                              the user can see how close to steady-state
                              the run actually got.
  - conservation.png        : energy, enstrophy, circulation against t.
                              Energy growing = lid doing work; enstrophy
                              settling = corner singularity damping;
                              circulation ≈ 0 = consistency check for
                              closed no-slip cavity.

Plus `summary.txt` with all the key numbers.

Usage: `python plot_cavity.py` from the lid_cavity folder.

Dependencies: numpy, pandas, matplotlib.
"""
from __future__ import annotations

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Windows defaults stdout to cp1252; force UTF-8 so π / Δ print cleanly.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

HERE       = os.path.dirname(os.path.abspath(__file__))
CASE_LABEL = "lid_re100"
OUT_DIR    = HERE

# Cavity geometry / parameters (must match the Julia config).
RE        = 100.0
U_LID     = 1.0
L_BOX     = 1.0


# ---------- data loading ----------------------------------------------------

def load_csv(path: str) -> pd.DataFrame:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing CSV: {path}")
    return pd.read_csv(path)


def main():
    paths = {
        "ghia_u":  os.path.join(HERE, f"{CASE_LABEL}_ghia_u_centerline.csv"),
        "ghia_v":  os.path.join(HERE, f"{CASE_LABEL}_ghia_v_centerline.csv"),
        "dense_u": os.path.join(HERE, f"{CASE_LABEL}_centerline_u_dense.csv"),
        "dense_v": os.path.join(HERE, f"{CASE_LABEL}_centerline_v_dense.csv"),
        "history": os.path.join(HERE, f"{CASE_LABEL}_history.csv"),
        "ss":      os.path.join(HERE, f"{CASE_LABEL}_steady_state.csv"),
    }
    data = {k: load_csv(v) for k, v in paths.items()}

    plot_ghia_centerline_u(data["dense_u"], data["ghia_u"])
    plot_ghia_centerline_v(data["dense_v"], data["ghia_v"])
    plot_ghia_per_point_error(data["ghia_u"], data["ghia_v"])
    plot_steady_state(data["ss"])
    plot_conservation(data["history"])
    write_summary(data)

    print("\nFigures written to", OUT_DIR + ":")
    for name in ("ghia_centerline_u.png",
                 "ghia_centerline_v.png",
                 "ghia_per_point_error.png",
                 "steady_state.png",
                 "conservation.png",
                 "summary.txt"):
        print(f"  {name}")


# ---------- centerline plots ------------------------------------------------

def plot_ghia_centerline_u(dense: pd.DataFrame, ghia: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(7, 6.5), constrained_layout=True)

    # Dense CO-FLIP curve (normalised by U_lid).
    y_d = dense["y"].values / L_BOX
    u_d = dense["u_over_U_lid"].values
    ax.plot(u_d, y_d, "-", color="C0", lw=2.2,
            label="CO-FLIP (dense)", zorder=2)

    interior = ghia["is_wall_endpoint"].values == 0
    wall     = ~interior
    # Ghia interior points
    ax.plot(ghia.loc[interior, "u_ghia"],
            ghia.loc[interior, "y_over_L"],
            "o", mfc="white", mec="black", mew=1.3, ms=8,
            label="Ghia (1982), interior", zorder=3)
    # Ghia wall endpoints (greyed)
    ax.plot(ghia.loc[wall, "u_ghia"],
            ghia.loc[wall, "y_over_L"],
            "s", mfc="lightgrey", mec="grey", mew=1.0, ms=8,
            label="Ghia, wall (excluded)", zorder=3)

    # CO-FLIP at the Ghia y/L (interior only) for direct comparison markers.
    ax.plot(ghia.loc[interior, "u_num"],
            ghia.loc[interior, "y_over_L"],
            "x", color="C3", mew=1.5, ms=9,
            label="CO-FLIP at Ghia y/L", zorder=4)

    ax.axhline(1.0, color="k", lw=0.6, ls=":")
    ax.axvline(0.0, color="k", lw=0.6, ls=":")
    ax.set_xlabel(r"$u / U_{\mathrm{lid}}$")
    ax.set_ylabel(r"$y / L$")
    ax.set_title(f"Lid-driven cavity (Re = {RE:g}): u along vertical centerline")
    ax.set_xlim(-0.4, 1.05)
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.95)

    annotate_ghia_l2(ax, ghia, axis="u")
    fig.savefig(os.path.join(OUT_DIR, "ghia_centerline_u.png"), dpi=150)
    plt.close(fig)


def plot_ghia_centerline_v(dense: pd.DataFrame, ghia: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(7.5, 6), constrained_layout=True)

    x_d = dense["x"].values / L_BOX
    v_d = dense["v_over_U_lid"].values
    ax.plot(x_d, v_d, "-", color="C0", lw=2.2,
            label="CO-FLIP (dense)", zorder=2)

    interior = ghia["is_wall_endpoint"].values == 0
    wall     = ~interior
    ax.plot(ghia.loc[interior, "x_over_L"],
            ghia.loc[interior, "v_ghia"],
            "o", mfc="white", mec="black", mew=1.3, ms=8,
            label="Ghia (1982), interior", zorder=3)
    ax.plot(ghia.loc[wall, "x_over_L"],
            ghia.loc[wall, "v_ghia"],
            "s", mfc="lightgrey", mec="grey", mew=1.0, ms=8,
            label="Ghia, wall (excluded)", zorder=3)
    ax.plot(ghia.loc[interior, "x_over_L"],
            ghia.loc[interior, "v_num"],
            "x", color="C3", mew=1.5, ms=9,
            label="CO-FLIP at Ghia x/L", zorder=4)

    ax.axhline(0.0, color="k", lw=0.6, ls=":")
    ax.set_xlabel(r"$x / L$")
    ax.set_ylabel(r"$v / U_{\mathrm{lid}}$")
    ax.set_title(f"Lid-driven cavity (Re = {RE:g}): v along horizontal centerline")
    ax.set_xlim(0.0, 1.0)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left", fontsize=9, framealpha=0.95)

    annotate_ghia_l2(ax, ghia, axis="v")
    fig.savefig(os.path.join(OUT_DIR, "ghia_centerline_v.png"), dpi=150)
    plt.close(fig)


def annotate_ghia_l2(ax, ghia: pd.DataFrame, axis: str):
    """Add a textbox with the interior L2 / max errors to the plot."""
    interior = ghia["is_wall_endpoint"].values == 0
    if axis == "u":
        e_num   = ghia.loc[interior, "u_num"].values
        e_ghia  = ghia.loc[interior, "u_ghia"].values
        coord   = ghia.loc[interior, "y_over_L"].values
    else:
        e_num   = ghia.loc[interior, "v_num"].values
        e_ghia  = ghia.loc[interior, "v_ghia"].values
        coord   = ghia.loc[interior, "x_over_L"].values
    diff = e_num - e_ghia
    # Trapezoidal L2 on the (non-uniform) Ghia grid
    def trapz_norm(c, vals):
        s = 0.0
        for i in range(len(c) - 1):
            s += 0.5 * (c[i + 1] - c[i]) * (vals[i] ** 2 + vals[i + 1] ** 2)
        return float(np.sqrt(s))
    l2_abs = trapz_norm(coord, diff)
    l2_ref = trapz_norm(coord, e_ghia)
    rel_l2 = l2_abs / max(l2_ref, 1e-30)
    max_abs = float(np.max(np.abs(diff)))
    txt = (f"Interior only (Ghia wall pts excluded):\n"
           f"  L2 rel  = {rel_l2:.3%}\n"
           f"  L2 abs  = {l2_abs:.3e}\n"
           f"  max err = {max_abs:.3e}")
    ax.text(0.02, 0.98, txt, transform=ax.transAxes,
            fontsize=9, va="top", ha="left",
            bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.9))


def plot_ghia_per_point_error(ghia_u: pd.DataFrame, ghia_v: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)

    def panel(ax, df, coord_col, num_col, ref_col, title):
        interior = df["is_wall_endpoint"].values == 0
        wall     = ~interior

        abs_err = df["abs_err"].values
        coord   = df[coord_col].values

        # Bar plot of absolute pointwise errors.
        colors  = ["C0" if i else "lightgrey" for i in interior]
        ax.bar(coord, abs_err, width=0.025, color=colors,
               edgecolor="black", linewidth=0.4,
               label=None)

        # Mark wall-endpoint bars explicitly
        for x, e, is_wall in zip(coord, abs_err, wall):
            if is_wall:
                ax.text(x, e * 1.02, "wall\n(excluded)",
                        ha="center", va="bottom", fontsize=7, color="dimgrey")

        # Overlay both Ghia and num curves for context (right y-axis).
        ax2 = ax.twinx()
        ax2.plot(coord, df[ref_col].values, "o-", color="black",
                 ms=4, lw=1.2, label="Ghia", alpha=0.7)
        ax2.plot(coord, df[num_col].values, "x-", color="C3",
                 ms=5, lw=1.2, label="CO-FLIP", alpha=0.9)
        ax2.axhline(0.0, color="k", lw=0.4, ls=":")
        ax2.set_ylabel("velocity (normalised)", fontsize=9)
        ax2.legend(loc="upper right", fontsize=8)

        ax.set_xlabel(coord_col.replace("_over_", "/").upper().replace("Y/L", "y/L").replace("X/L", "x/L"))
        ax.set_ylabel("absolute error")
        ax.set_title(title)
        ax.grid(True, alpha=0.3, axis="y")

    panel(axes[0], ghia_u, "y_over_L", "u_num", "u_ghia",
          "u centerline (x = L/2): pointwise error")
    panel(axes[1], ghia_v, "x_over_L", "v_num", "v_ghia",
          "v centerline (y = L/2): pointwise error")
    fig.suptitle(f"Pointwise comparison to Ghia (1982), Re = {RE:g}",
                 fontsize=12, weight="bold")
    fig.savefig(os.path.join(OUT_DIR, "ghia_per_point_error.png"), dpi=150)
    plt.close(fig)


# ---------- diagnostic plots ------------------------------------------------

def plot_steady_state(ss: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)

    ax = axes[0]
    ax.semilogy(ss["t"], ss["abs_diff_L2"], "-", color="C0", lw=1.6,
                label=r"$\Vert u_h - u_h^{prev}\Vert_M$")
    ax.semilogy(ss["t"], ss["norm_u_L2"], "-", color="C2", lw=1.6,
                label=r"$\Vert u_h \Vert_M$")
    ax.set_xlabel("t")
    ax.set_ylabel("mass-weighted L2 norm")
    ax.set_title("Steady-state convergence: absolute norms")
    ax.legend(fontsize=10, loc="lower right")
    ax.grid(True, which="both", alpha=0.3)

    ax = axes[1]
    ax.semilogy(ss["t"], ss["rel_diff"], "-", color="C3", lw=1.6)
    ax.set_xlabel("t")
    ax.set_ylabel(r"$\Vert \Delta u\Vert_M / \Vert u \Vert_M$")
    ax.set_title("Relative change between consecutive snapshots")
    ax.axhline(1e-3, color="k", lw=0.6, ls="--", alpha=0.5)
    ax.text(ax.get_xlim()[1], 1e-3, "  1e-3 (typical SS threshold)",
            ha="right", va="bottom", fontsize=8, color="dimgrey")
    ax.grid(True, which="both", alpha=0.3)

    fig.suptitle(f"Lid cavity (Re = {RE:g}): approach to steady state",
                 fontsize=12, weight="bold")
    fig.savefig(os.path.join(OUT_DIR, "steady_state.png"), dpi=150)
    plt.close(fig)


def plot_conservation(hist: pd.DataFrame):
    fig, axes = plt.subplots(3, 1, figsize=(9, 8.5), constrained_layout=True,
                             sharex=True)

    ax = axes[0]
    ax.plot(hist["t"], hist["energy"], "-", color="C0", lw=1.4)
    ax.set_ylabel("energy")
    ax.set_title("Kinetic energy (grows as the lid does work on the fluid)")
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.semilogy(hist["t"], hist["enstrophy"].clip(lower=1e-12), "-", color="C1", lw=1.4)
    ax.set_ylabel("enstrophy")
    ax.set_title("Enstrophy (initial spike = corner singularity, then settles)")
    ax.grid(True, which="both", alpha=0.3)

    ax = axes[2]
    ax.plot(hist["t"], hist["circulation"], "-", color="C2", lw=1.4)
    ax.axhline(0.0, color="k", lw=0.6, ls=":")
    ax.set_ylabel("circulation")
    ax.set_xlabel("t")
    ax.set_title("Circulation (should approach 0 — closed cavity, Brinkman walls)")
    ax.grid(True, alpha=0.3)

    fig.suptitle(f"Lid cavity (Re = {RE:g}): conservation diagnostics",
                 fontsize=12, weight="bold")
    fig.savefig(os.path.join(OUT_DIR, "conservation.png"), dpi=150)
    plt.close(fig)


# ---------- summary ---------------------------------------------------------

def write_summary(data):
    ghia_u = data["ghia_u"]
    ghia_v = data["ghia_v"]
    ss     = data["ss"]
    hist   = data["history"]

    interior_u = ghia_u["is_wall_endpoint"].values == 0
    interior_v = ghia_v["is_wall_endpoint"].values == 0

    def trapz_norm(c, vals):
        s = 0.0
        for i in range(len(c) - 1):
            s += 0.5 * (c[i + 1] - c[i]) * (vals[i] ** 2 + vals[i + 1] ** 2)
        return float(np.sqrt(s))

    eu = (ghia_u.loc[interior_u, "u_num"].values
          - ghia_u.loc[interior_u, "u_ghia"].values)
    ev = (ghia_v.loc[interior_v, "v_num"].values
          - ghia_v.loc[interior_v, "v_ghia"].values)
    cu = ghia_u.loc[interior_u, "y_over_L"].values
    cv = ghia_v.loc[interior_v, "x_over_L"].values
    refu = ghia_u.loc[interior_u, "u_ghia"].values
    refv = ghia_v.loc[interior_v, "v_ghia"].values

    rel_l2_u = trapz_norm(cu, eu) / max(trapz_norm(cu, refu), 1e-30)
    rel_l2_v = trapz_norm(cv, ev) / max(trapz_norm(cv, refv), 1e-30)
    max_u    = float(np.max(np.abs(eu)))
    max_v    = float(np.max(np.abs(ev)))

    lines = []
    lines.append(f"Lid-driven cavity at Re = {RE:g} — analysis summary")
    lines.append("=" * 60)
    lines.append("")
    lines.append("Ghia (1982) centerline validation (interior points only):")
    lines.append(f"  u(x=L/2, y) :  rel L2 = {rel_l2_u:.3%},  max err = {max_u:.3e}")
    lines.append(f"  v(x, y=L/2) :  rel L2 = {rel_l2_v:.3%},  max err = {max_v:.3e}")
    lines.append("")
    lines.append("Steady-state convergence:")
    if len(ss) > 0:
        lines.append(f"  t_final         = {ss['t'].iloc[-1]:.3f}")
        lines.append(f"  rel_diff(start) = {ss['rel_diff'].iloc[0]:.3e}")
        lines.append(f"  rel_diff(end)   = {ss['rel_diff'].iloc[-1]:.3e}")
        lines.append(f"  ||u||_M (end)   = {ss['norm_u_L2'].iloc[-1]:.3e}")
    lines.append("")
    lines.append("Conservation history:")
    if len(hist) > 0:
        lines.append(f"  E(t=0)    = {hist['energy'].iloc[0]:.3e}")
        lines.append(f"  E(end)    = {hist['energy'].iloc[-1]:.3e}   (growth = lid work)")
        lines.append(f"  Z(t=0)    = {hist['enstrophy'].iloc[0]:.3e}")
        lines.append(f"  Z(end)    = {hist['enstrophy'].iloc[-1]:.3e}")
        lines.append(f"  Γ(t=0)    = {hist['circulation'].iloc[0]:.3e}")
        lines.append(f"  Γ(end)    = {hist['circulation'].iloc[-1]:.3e}   (should ≈ 0)")
    lines.append("")
    lines.append("Notes:")
    lines.append("  • The y/L=0 and y/L=1 (and x/L=0, 1) Ghia rows are wall")
    lines.append("    endpoints — excluded from the L2 metric because the FEEC")
    lines.append("    dy-form derivative space `D_y` vanishes structurally at")
    lines.append("    y-walls (and same for `D_x` at x-walls). The Brinkman lid")
    lines.append("    drives `u ≈ U_lid` half a cell INSIDE the boundary, not")
    lines.append("    AT the boundary itself. Compare with Ghia at y/L = 0.977")
    lines.append("    (the topmost interior reference point) rather than y/L = 1.")
    lines.append("  • Energy growth = lid does work on the fluid → "
                 "`enable_energy_correction=false` is required.")
    lines.append("  • Circulation ≈ 0 at long time = closed-cavity consistency.")
    text = "\n".join(lines)
    print(text)
    with open(os.path.join(OUT_DIR, "summary.txt"), "w", encoding="utf-8") as f:
        f.write(text + "\n")


if __name__ == "__main__":
    main()
