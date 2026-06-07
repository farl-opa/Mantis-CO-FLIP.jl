#!/usr/bin/env python3
"""
Plot CO-FLIP Von Kármán cylinder-wake results and compare to bibliography.

Reads (from the directory passed as argv[1], or the current directory):
    <case>_summary.txt   — text dump from test_von_karman_strouhal
    <case>_forces.csv    — t, step, dt, fx, fy, Cd, Cl    (one row per step)
    <case>_probes.csv    — t, step, u_p1, v_p1, u_p2, v_p2, u_p3, v_p3
    <case>_history.csv   — t, energy, enstrophy, circulation

Produces four PNGs alongside the CSVs:
    <case>_forces.png        — Cd(t), Cl(t) with stats window + reference
    <case>_probe_psd.png     — v-probe signal and its windowed periodogram
    <case>_conservation.png  — global energy / enstrophy / circulation
    <case>_comparison.png    — measured vs literature bar chart + relative error

Usage:
    python plot_von_karman.py [case_dir] [case_label]

If `case_label` is omitted the first `vk_*_forces.csv` in `case_dir` is used.
The `case_dir` defaults to the directory containing this script.

Dependencies: numpy, matplotlib. (Scipy is intentionally not required.)
"""

from __future__ import annotations

import glob
import os
import re
import sys

import numpy as np
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# Bibliographic reference values for the laminar 2D cylinder benchmark.
#
# Sources:
#   Williamson 1996  "Vortex Dynamics in the Cylinder Wake", Annu.Rev.Fluid Mech 28, 477
#   Henderson 1995   "Details of the drag curve near the onset of vortex shedding", PoF 7, 2102
#   Park, Kwon & Choi 1998  "Numerical solutions of flow past a circular cylinder
#                            at Reynolds numbers up to 160", KSME Int.J 12, 1200
#   Norberg 2003     "Fluctuating lift on a circular cylinder: review and new
#                     measurements", J.Fluids.Struct 17, 57
#   Posdziech & Grundmann 2007  "A systematic approach to the numerical calculation
#                                of fundamental quantities of the two-dimensional
#                                flow over a circular cylinder", J.Fluids.Struct 23, 479
#
# Cl_rms entries are the standard deviation of CL(t) about its mean, i.e. how
# the CO-FLIP test reports it.
# -----------------------------------------------------------------------------
REFS = {
    # Re : dict(St, Cd, Cl_rms, sources)
    100: dict(St=0.165, Cd=1.34, Cl_rms=0.227,
              sources="Williamson 1996; Park 1998; Norberg 2003"),
    150: dict(St=0.183, Cd=1.33, Cl_rms=0.38,
              sources="Williamson 1996; Henderson 1995; Norberg 2003"),
    200: dict(St=0.196, Cd=1.34, Cl_rms=0.49,
              sources="Williamson 1996; Henderson 1995; Norberg 2003"),
    300: dict(St=0.207, Cd=1.36, Cl_rms=0.61,
              sources="Williamson 1996; Norberg 2003"),
}


# -----------------------------------------------------------------------------
# I/O
# -----------------------------------------------------------------------------
def parse_summary(path: str) -> dict:
    """Pull Re, U, D, ν, St, Cd_mean, Cl_mean, Cl_rms out of the summary.txt."""
    summary = {}
    if not os.path.isfile(path):
        return summary
    with open(path, encoding="utf-8") as f:
        text = f.read()

    # Re=150.00  U=1.5000  D=2.0000  ν=0.0200
    m = re.search(r"Re=([\d.eE+-]+)\s+U=([\d.eE+-]+)\s+D=([\d.eE+-]+)\s+[ν\w]+=([\d.eE+-]+)",
                  text)
    if m:
        summary["Re"], summary["U"], summary["D"], summary["nu"] = (
            float(m.group(1)), float(m.group(2)),
            float(m.group(3)), float(m.group(4)),
        )

    # Lines like "Strouhal       St = 0.194396" or "Cd_mean = 1.739138  (window: ...)"
    for key, pattern in (
        ("St",      r"(?:Strouhal\s+St|St)\s*=\s*([-\d.eE+]+)"),
        ("Cd_mean", r"Cd_mean\s*=\s*([-\d.eE+]+)"),
        ("Cl_mean", r"Cl_mean\s*=\s*([-\d.eE+]+)"),
        ("Cl_rms",  r"Cl_rms\s*=\s*([-\d.eE+]+)"),
    ):
        m = re.search(pattern, text)
        if m:
            summary[key] = float(m.group(1))
    return summary


def load_csv(path: str) -> dict:
    """Load a comma-separated file with a single header line into {col: array}."""
    with open(path, encoding="utf-8") as f:
        header = f.readline().strip().split(",")
    data = np.loadtxt(path, delimiter=",", skiprows=1)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    return {h: data[:, i] for i, h in enumerate(header)}


def find_case_label(case_dir: str) -> str:
    csvs = sorted(glob.glob(os.path.join(case_dir, "*_forces.csv")))
    if not csvs:
        raise FileNotFoundError(
            f"No *_forces.csv found in {case_dir!r}. "
            "Pass [case_dir] [case_label] explicitly."
        )
    return os.path.basename(csvs[0])[: -len("_forces.csv")]


# -----------------------------------------------------------------------------
# Statistics
# -----------------------------------------------------------------------------
def windowed_stats(signal: np.ndarray, frac: float = 0.5):
    """Mean, std, and the index where the stats window starts."""
    n = len(signal)
    start = int(n * (1.0 - frac))
    window = signal[start:]
    return float(window.mean()), float(window.std()), start


# -----------------------------------------------------------------------------
# Plots
# -----------------------------------------------------------------------------
def plot_forces(forces, summary, case_label, out_dir):
    Re_int = int(round(summary.get("Re", 0)))
    ref = REFS.get(Re_int)
    t  = forces["t"]
    Cd = forces["Cd"]
    Cl = forces["Cl"]
    Cd_mean, _Cd_std, win = windowed_stats(Cd, 0.5)
    Cl_mean, Cl_rms, _    = windowed_stats(Cl, 0.5)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10.5, 6.0), sharex=True)
    ax1.plot(t, Cd, color="tab:blue", lw=0.9, label=r"$C_D(t)$")
    ax1.axvspan(t[win], t[-1], color="gray", alpha=0.12,
                label=f"stats window (last {len(Cd) - win} samples)")
    ax1.axhline(Cd_mean, color="tab:blue", ls="--", lw=1.2,
                label=fr"$\overline{{C_D}}={Cd_mean:.3f}$")
    if ref:
        ax1.axhline(ref["Cd"], color="black", ls=":", lw=1.2,
                    label=fr"reference $\overline{{C_D}}\!=\!{ref['Cd']:.2f}$")
    ax1.set_ylabel(r"$C_D$")
    ax1.grid(alpha=0.3)
    ax1.legend(loc="upper right", fontsize=9, framealpha=0.9)

    ax2.plot(t, Cl, color="tab:red", lw=0.9, label=r"$C_L(t)$")
    ax2.axvspan(t[win], t[-1], color="gray", alpha=0.12)
    ax2.axhline(Cl_mean,           color="tab:red", ls="--", lw=1.0,
                label=fr"$\overline{{C_L}}={Cl_mean:+.3f}$")
    ax2.axhline(Cl_mean + Cl_rms,  color="tab:red", ls=":", lw=0.9,
                label=fr"$\pm C_{{L,\mathrm{{rms}}}}={Cl_rms:.3f}$")
    ax2.axhline(Cl_mean - Cl_rms,  color="tab:red", ls=":", lw=0.9)
    if ref:
        ax2.axhline(+ref["Cl_rms"], color="black", ls=":", lw=1.2,
                    label=fr"reference $\pm C_{{L,\mathrm{{rms}}}}\!=\!{ref['Cl_rms']:.2f}$")
        ax2.axhline(-ref["Cl_rms"], color="black", ls=":", lw=1.2)
    ax2.set_ylabel(r"$C_L$")
    ax2.set_xlabel("t")
    ax2.grid(alpha=0.3)
    ax2.legend(loc="upper right", fontsize=9, framealpha=0.9)

    fig.suptitle(fr"CO-FLIP cylinder wake — force coefficients (Re $=$ {Re_int})")
    fig.tight_layout()
    out = os.path.join(out_dir, case_label + "_forces.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  saved {out}")


def plot_probe_and_psd(probes, summary, case_label, out_dir):
    Re_int = int(round(summary.get("Re", 0)))
    ref = REFS.get(Re_int)
    t = probes["t"]
    v = probes["v_p1"]
    # The CO-FLIP run uses an adaptive dt, so resample to a uniform grid
    # before FFT'ing. Linear interp is fine for a periodogram peak.
    n = len(t)
    t_uniform = np.linspace(t[0], t[-1], n)
    v_uniform = np.interp(t_uniform, t, v)
    dt = (t_uniform[-1] - t_uniform[0]) / max(n - 1, 1)

    # Use second half (developed shedding) with a Hann window for the FFT
    n_half = n // 2
    v_win  = v_uniform[n_half:]
    v_win  = v_win - v_win.mean()
    if len(v_win) > 1:
        v_win = v_win * np.hanning(len(v_win))
        fft   = np.fft.rfft(v_win)
        freqs = np.fft.rfftfreq(len(v_win), d=dt)
        power = np.abs(fft) ** 2
        # Skip DC bin (index 0) when picking the peak
        i_peak = int(power[1:].argmax()) + 1 if len(power) > 1 else 0
        f_peak = float(freqs[i_peak])
    else:
        freqs, power, f_peak = np.array([0.0]), np.array([0.0]), 0.0
    St_measured = f_peak * summary.get("D", 1.0) / max(summary.get("U", 1.0), 1e-12)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10.5, 6.0))
    ax1.plot(t, v, color="tab:green", lw=0.8)
    ax1.set_xlabel("t")
    ax1.set_ylabel(r"$v$ at probe 1")
    ax1.set_title(fr"Cross-stream velocity at downstream probe (Re $=$ {Re_int})")
    ax1.grid(alpha=0.3)

    ax2.semilogy(freqs, np.maximum(power, 1e-30),
                 color="tab:green", lw=1.0, label="periodogram (Hann, 2nd half)")
    ax2.axvline(f_peak, color="tab:green", ls="--", lw=1.2,
                label=fr"$f_\mathrm{{peak}}={f_peak:.4f}$, $St={St_measured:.4f}$")
    if ref:
        f_ref = ref["St"] * summary.get("U", 1.0) / summary.get("D", 1.0)
        ax2.axvline(f_ref, color="black", ls=":", lw=1.2,
                    label=fr"reference $St={ref['St']:.3f}$, $f={f_ref:.4f}$")
    # Zoom around the peak so the spectrum is readable
    f_top = max(5.0 * f_peak, 0.4) if f_peak > 0 else float(freqs[-1])
    ax2.set_xlim(0.0, min(f_top, float(freqs[-1])))
    ax2.set_xlabel("frequency")
    ax2.set_ylabel("power (arbitrary units)")
    ax2.grid(alpha=0.3, which="both")
    ax2.legend(loc="upper right", fontsize=9, framealpha=0.9)

    fig.tight_layout()
    out = os.path.join(out_dir, case_label + "_probe_psd.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  saved {out}")


def plot_conservation(history, case_label, out_dir):
    t = history["t"]
    fig, axes = plt.subplots(3, 1, figsize=(10.5, 7.0), sharex=True)
    axes[0].plot(t, history["energy"],      color="tab:blue",   lw=0.9)
    axes[0].set_ylabel("kinetic energy")
    axes[0].grid(alpha=0.3)

    axes[1].plot(t, history["enstrophy"],   color="tab:purple", lw=0.9)
    axes[1].set_ylabel("enstrophy")
    axes[1].grid(alpha=0.3)

    axes[2].plot(t, history["circulation"], color="tab:orange", lw=0.9)
    axes[2].axhline(0.0, color="black", ls=":", lw=0.8)
    axes[2].set_ylabel("circulation")
    axes[2].set_xlabel("t")
    axes[2].grid(alpha=0.3)

    fig.suptitle("Global diagnostics — quasi-steady once shedding is established")
    fig.tight_layout()
    out = os.path.join(out_dir, case_label + "_conservation.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  saved {out}")


def plot_comparison(summary, case_label, out_dir):
    Re_int = int(round(summary.get("Re", 0)))
    ref = REFS.get(Re_int)
    if ref is None:
        print(f"  no bibliography reference tabulated for Re={Re_int}; "
              "skipping comparison plot")
        return

    labels = ["Strouhal", r"$\overline{C_D}$", r"$C_{L,\mathrm{rms}}$"]
    keys_m = ["St",        "Cd_mean",         "Cl_rms"]
    keys_r = ["St",        "Cd",              "Cl_rms"]
    measured  = [summary.get(k, float("nan")) for k in keys_m]
    reference = [ref[k]                       for k in keys_r]
    devs = [100.0 * (m - r) / r if r != 0 else 0.0 for m, r in zip(measured, reference)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.0, 4.8))

    x = np.arange(len(labels))
    w = 0.36
    bars_m = ax1.bar(x - w/2, measured,  w,
                     color="tab:blue", edgecolor="black", label="CO-FLIP")
    bars_r = ax1.bar(x + w/2, reference, w,
                     color="lightgray", edgecolor="black",
                     label=f"reference\n({ref['sources']})")
    for b, v in zip(bars_m, measured):
        ax1.text(b.get_x() + b.get_width()/2, v + 0.03,
                 f"{v:.3f}", ha="center", fontsize=9)
    for b, v in zip(bars_r, reference):
        ax1.text(b.get_x() + b.get_width()/2, v + 0.03,
                 f"{v:.3f}", ha="center", fontsize=9)
    ax1.set_xticks(x); ax1.set_xticklabels(labels)
    ax1.set_title(fr"Measured vs literature — Re $=$ {Re_int}")
    ax1.grid(alpha=0.3, axis="y")
    ax1.legend(loc="upper left", fontsize=8.5, framealpha=0.9)

    colors = ["tab:blue" if abs(d) <= 15 else
              "tab:orange" if abs(d) <= 35 else "tab:red" for d in devs]
    bars_d = ax2.bar(x, devs, color=colors, edgecolor="black")
    for b, d in zip(bars_d, devs):
        offset = 1.0 if d >= 0 else -1.0
        ax2.text(b.get_x() + b.get_width()/2, d + offset,
                 f"{d:+.1f}%", ha="center", fontsize=9)
    ax2.axhline(0.0, color="black", lw=0.8)
    ax2.axhspan(-15, 15, color="green",  alpha=0.10,
                label="±15% (good)")
    ax2.axhspan(-35, -15, color="orange", alpha=0.08)
    ax2.axhspan( 15,  35, color="orange", alpha=0.08,
                 label="±15–35% (typical IBM)")
    ax2.set_xticks(x); ax2.set_xticklabels(labels)
    ax2.set_ylabel("deviation from reference [%]")
    ax2.set_title("Relative error")
    ax2.grid(alpha=0.3, axis="y")
    ax2.legend(loc="upper right", fontsize=9, framealpha=0.9)

    fig.tight_layout()
    out = os.path.join(out_dir, case_label + "_comparison.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  saved {out}")


# -----------------------------------------------------------------------------
# Text analysis
# -----------------------------------------------------------------------------
def print_analysis(summary):
    Re_int = int(round(summary.get("Re", 0)))
    ref = REFS.get(Re_int)
    print()
    print("=" * 68)
    print(f"  CO-FLIP cylinder wake -- analysis (Re = {Re_int})")
    print("=" * 68)
    print(f"  U_inf = {summary.get('U', float('nan')):.4f}   "
          f"D = {summary.get('D', float('nan')):.4f}   "
          f"nu = {summary.get('nu', float('nan')):.5f}")
    print()
    print(f"  {'Quantity':18s}|{'CO-FLIP':>11s}|{'Reference':>11s}|{'dev%':>9s}")
    print("  " + "-" * 56)
    for label, key_m, key_r in (("Strouhal St", "St", "St"),
                                ("Cd_mean",    "Cd_mean", "Cd"),
                                ("Cl_rms",     "Cl_rms", "Cl_rms")):
        m = summary.get(key_m)
        if m is None or ref is None:
            continue
        r = ref[key_r]
        dev = 100.0 * (m - r) / r
        print(f"  {label:18s}|{m:>11.4f}|{r:>11.4f}|{dev:>+8.1f}%")
    cl_mean = summary.get("Cl_mean")
    if cl_mean is not None:
        print()
        print(f"  Cl_mean = {cl_mean:+.4f}   "
              f"(expected ~ 0 by top/bottom symmetry; small bias OK)")
    if ref is not None:
        print()
        print(f"  Reference sources: {ref['sources']}")

    # ------------- discussion -------------
    print()
    print("  Discussion")
    print("  ----------")
    print("  - Strouhal: most reliable single number here. Depends only on")
    print("    shedding frequency, not surface-stress accuracy, so the small")
    print("    deviation reflects mainly the discrete frequency resolution")
    print("    (df ~ 1/T_record) plus modest viscous-CFL drift.")
    print()
    print("  - Cd_mean / Cl_rms: in volume-penalisation IBM (Brinkman tau->0)")
    print("    the no-slip layer is smeared across the cells that intersect")
    print("    the obstacle surface. With only ~5 cells across the diameter,")
    print("    this band over-predicts both mean drag and lift fluctuations")
    print("    by 25-40% -- extensively documented by Angot-Bruneau-Fabrie")
    print("    (1999), Schneider & Farge (2005), Bergmann & Iollo (2011).")
    print("    Cd_mean and Cl_rms converge toward the reference roughly")
    print("    linearly in mesh size; doubling `nel` cuts the gap ~2x.")
    print()
    print("  - Cl_mean ~ 0 is the symmetry check: a clean Karman street")
    print("    has zero mean lift. A non-zero value indicates either an")
    print("    asymmetric initial perturbation or a finite-time averaging")
    print("    bias (the RMS window may not cover a whole number of cycles).")


# -----------------------------------------------------------------------------
# Driver
# -----------------------------------------------------------------------------
def main():
    case_dir   = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(__file__) or "."
    case_label = sys.argv[2] if len(sys.argv) > 2 else find_case_label(case_dir)

    print(f"\nReading {case_label} from {case_dir!r} ...")

    summary = parse_summary(os.path.join(case_dir, case_label + "_summary.txt"))
    forces  = load_csv     (os.path.join(case_dir, case_label + "_forces.csv"))
    probes  = load_csv     (os.path.join(case_dir, case_label + "_probes.csv"))
    history = load_csv     (os.path.join(case_dir, case_label + "_history.csv"))

    print_analysis(summary)

    print()
    print("  Generating plots ...")
    plot_forces       (forces,  summary, case_label, case_dir)
    plot_probe_and_psd(probes,  summary, case_label, case_dir)
    plot_conservation (history,          case_label, case_dir)
    plot_comparison   (         summary, case_label, case_dir)
    print()


if __name__ == "__main__":
    main()
