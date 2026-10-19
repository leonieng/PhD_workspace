"""
All plotting routines for the LSM mode calculation. Each plot_* function takes the
results dict from lsm_mode_functions.run_lsm_analysis and saves one PNG; plotter_wrapper
runs the ones requested in lsm_mode_input.py.
"""
import os
from fractions import Fraction

import matplotlib
matplotlib.use("Agg")   # figures are only saved to file, never shown
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers the 3d projection)
from scipy.constants import c

from lsm_mode_functions import mirror_in_y, physical_field

FIELD_NAMES = ["E_x", "E_y", "E_z", "H_x", "H_y", "H_z"]


def _save(fig, path, dpi, **kw):
    fig.savefig(path, dpi=dpi, **kw)
    plt.close(fig)
    return path


def plot_dispersion(results, path, dpi=150, n_light_line=100):
    """beta(f) of every requested mode with the light line."""
    g, f_min, f_max = results["guide"], results["f_min"], results["f_max"]
    fig = plt.figure(figsize=(7, 5))
    for (family, m, n), (f, bt) in results["curves"].items():
        plt.plot(f / 1e9, bt, marker="o", ms=2.5, lw=1, label=f"{family}$_{{{m}{n}}}$")

    f = np.linspace(f_min, f_max, n_light_line)
    plt.plot(f / 1e9, 2 * np.pi * f / c, "k", lw=1, label="light line ($k_0$)")

    plt.xlabel("frequency (GHz)")
    plt.ylabel("propagation constant β (rad/m)")
    plt.title(f"w={g.w*1e3:.2f} mm, b={g.b*1e3:.2f} mm, a={g.a*1e3:.2f} mm, εr={g.eps_r}")
    plt.ylim(bottom=0)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    return _save(fig, path, dpi)


def plot_field_components(results, path, dpi=150):
    """
    3x2 plot of the six field components over the transverse plane at z = 0
    (wave phase omega*t - beta*z = 0, the accelerating crest). Real parts only: at the crest
    E_x, E_y, H_x are exactly zero (90 degrees out of phase with E_z), H_y is identically
    zero for LSM.
    """
    g, m, n = results["guide"], results["m"], results["n"]
    X, Y, fields = results["X"], results["Y"], results["fields"]

    fig, axes = plt.subplots(3, 2, figsize=(10, 12), sharex=True, sharey=True)
    for ax, name, field in zip(axes.flat, FIELD_NAMES, fields):
        pcm = ax.pcolormesh(X * 1e3, Y * 1e3, np.real(field), shading="auto", cmap="RdBu_r")
        ax.axhline(g.a * 1e3, color="k", lw=0.8, ls="--")  # dielectric interface
        ax.set_title(name)
        ax.set_xlabel("x (mm)")
        ax.set_ylabel("y (mm)")
        fig.colorbar(pcm, ax=ax)

    fig.suptitle(f"LSM$_{{{m}{n}}}$ field components at the accelerating crest ($\\omega t-\\beta z=0$), "
                 f"f = {results['frequency']/1e9:.2f} GHz, beta = {results['beta_mn']:.1f} rad/m\n"
                 "(blank panels are zero at this phase, not absent)")
    fig.tight_layout()
    return _save(fig, path, dpi)


def plot_field_snapshots(results, path, dpi=150, thetas=(0, np.pi / 2, np.pi, 3 * np.pi / 2)):
    """
    Physical (time-domain) fields at several wave phases theta = omega*t - beta*z:
    Re[F(x, y) exp(+j theta)]. theta = 0 is E_z at its peak (E_x, E_y, H_x zero);
    theta = pi/2 is a quarter period later, with the transverse fields at their peak.
    """
    g, m, n = results["guide"], results["m"], results["n"]
    X, Y, fields = results["X"], results["Y"], results["fields"]

    def _label(th):
        fr = Fraction(th / np.pi).limit_denominator(12)
        if fr == 0:
            return "0"
        num = "" if fr.numerator == 1 else str(fr.numerator)
        den = "" if fr.denominator == 1 else f"/{fr.denominator}"
        return f"${num}\\pi{den}$"
    theta_labels = [_label(th) for th in thetas]
    show = [("E_x", 0), ("E_y", 1), ("E_z", 2), ("H_x", 3), ("H_z", 5)]   # H_y is identically zero

    fig, axes = plt.subplots(len(show), len(thetas), figsize=(3.5 * len(thetas), 2.6 * len(show)),
                             sharex=True, sharey=True, squeeze=False)
    for row, (name, idx) in enumerate(show):
        snaps = [physical_field(fields[idx], th) for th in thetas]
        vmax = max(np.max(np.abs(s)) for s in snaps)          # one symmetric scale per row
        for col, snap in enumerate(snaps):
            ax = axes[row, col]
            pcm = ax.pcolormesh(X * 1e3, Y * 1e3, snap, shading="auto", cmap="RdBu_r",
                                vmin=-vmax, vmax=vmax)
            ax.axhline(g.a * 1e3, color="k", lw=0.6, ls="--")
            if row == 0:
                ax.set_title(f"$\\omega t - \\beta z$ = {theta_labels[col]}")
            if col == 0:
                ax.set_ylabel(f"{name}\ny (mm)")
            if row == len(show) - 1:
                ax.set_xlabel("x (mm)")
        fig.colorbar(pcm, ax=axes[row, :], pad=0.01, aspect=12)

    fig.suptitle(f"LSM$_{{{m}{n}}}$ physical fields vs wave phase, f = {results['frequency']/1e9:.2f} GHz")
    return _save(fig, path, dpi, bbox_inches="tight")


def plot_accel_vs_transverse(results, path, dpi=150, quiver_stride=12, pw_stride=8):
    """
    (a) accelerating force E_z, (b) transverse force map, (c) F_y along x = 0 split into its
    electric and magnetic parts and compared with the Panofsky-Wenzel prediction.
    """
    g, m, n = results["guide"], results["m"], results["n"]
    X, Y, y_grid, fr = results["X"], results["Y"], results["y_grid"], results["forces"]
    y_mm = y_grid * 1e3
    s = quiver_stride

    fig, axes = plt.subplots(1, 3, figsize=(18, 4.8))

    # (a) accelerating force
    ax = axes[0]
    vmax = np.max(np.abs(fr["Ez_acc"])) * 1e-3
    pcm = ax.pcolormesh(X * 1e3, Y * 1e3, fr["Ez_acc"] * 1e-3, shading="auto", cmap="RdBu_r",
                        vmin=-vmax, vmax=vmax)
    fig.colorbar(pcm, ax=ax, label="$F_z/q = E_z$  (kV/m)")
    ax.set_title("(a) accelerating force, $\\omega t-\\beta z = 0$")

    # (b) transverse force, arrows = direction and relative size
    ax = axes[1]
    pcm = ax.pcolormesh(X * 1e3, Y * 1e3, np.hypot(fr["Fx"], fr["Fy"]) * 1e-3, shading="auto", cmap="viridis")
    fig.colorbar(pcm, ax=ax, label="$|F_\\perp|/q$  (kV/m)")
    ax.quiver(X[::s, ::s] * 1e3, Y[::s, ::s] * 1e3, fr["Fx"][::s, ::s], fr["Fy"][::s, ::s],
              color="w", pivot="mid")
    ax.set_title("(b) transverse force $(F_x, F_y)/q$, $\\omega t-\\beta z = \\pi/2$")

    for ax in axes[:2]:
        ax.axhline(g.a * 1e3, color="k", lw=0.8, ls="--")
        ax.set_xlabel("x (mm)")
        ax.set_ylabel("y (mm)")

    # (c) F_y along x = 0: electric and magnetic parts, their sum, and the Panofsky-Wenzel prediction
    ax = axes[2]
    E_part, B_part, pw = fr["E_part"], fr["B_part"], fr["pw"]
    ax.plot(y_mm, E_part * 1e-3, label="$E_y$")
    ax.plot(y_mm, B_part * 1e-3, label="$c\\mu_0 H_x$")
    ax.plot(y_mm, (E_part + B_part) * 1e-3, "k", lw=2, label="$F_y/q = E_y + c\\mu_0 H_x$")
    ax.plot(y_mm[::pw_stride], pw[::pw_stride] * 1e-3, "o", ms=5, mfc="none", color="C3",
            label="$-\\frac{1}{k}\\,\\partial_y E_z$  (Panofsky-Wenzel)")
    ax.axvline(g.a * 1e3, color="k", lw=0.8, ls="--")
    ax.axhline(0, color="gray", lw=0.5)
    ax.set_xlabel("y (mm)")
    ax.set_ylabel("kV/m")
    ax.set_title("(c) $F_y$ along x = 0: magnetic term cancels most of $E_y$")
    ax.legend(fontsize=8)

    fig.suptitle(f"LSM$_{{{m}{n}}}$ at {results['frequency']/1e9:.2f} GHz: "
                 "accelerating vs transverse force on a v = c particle")
    fig.tight_layout()
    return _save(fig, path, dpi)


def plot_Ez_3d(results, path, dpi=150):
    """
    Normalized longitudinal electric field |E_z|(x, y), 3D surface (cf. Fig. 4 of the paper).
    y = 0 is a symmetry plane (E_z even in y), so the full cross-section -b < y < b is
    obtained by mirroring.
    """
    m, n = results["m"], results["n"]
    x_grid, y_grid, E_z = results["x_grid"], results["y_grid"], results["fields"][2]
    E_z_norm = np.abs(E_z) / np.max(np.abs(E_z))

    y_full, E_z_full = mirror_in_y(y_grid, E_z_norm)
    X_full, Y_full = np.meshgrid(x_grid, y_full)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(projection="3d")
    ax.plot_surface(X_full * 1e3, Y_full * 1e3, E_z_full, cmap="viridis", linewidth=0, antialiased=True)
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_zlabel("normalized |E_z|")
    ax.set_title(f"LSM$_{{{m}{n}}}$ normalized longitudinal E-field, f = {results['frequency']/1e9:.2f} GHz")
    fig.tight_layout()
    return _save(fig, path, dpi)


def plot_Ez_vs_z(results, path, dpi=150):
    """
    Effect of the z-dependence exp(-j*beta*z): a pure travelling-wave phase factor, so |E_z|
    is unchanged and only the physical (real) part oscillates along z.
    """
    z, E_z_z = results["z"], results["E_z_z"]
    x0, y0 = results["z_point"]

    fig = plt.figure(figsize=(7, 4))
    plt.plot(z * 1e3, np.real(E_z_z), label="Re(E_z)")
    plt.plot(z * 1e3, np.abs(E_z_z), "k--", label="|E_z|")
    plt.xlabel("z (mm)")
    plt.ylabel("E_z (a.u.)")
    plt.title(f"Travelling wave along z at (x,y) = ({x0*1e3:.1f}, {y0*1e3:.1f}) mm")
    plt.legend()
    plt.tight_layout()
    return _save(fig, path, dpi)


# name used in lsm_mode_input.py -> (plot function, output file name)
PLOTS = {
    "dispersion": (plot_dispersion, "dispersion.png"),
    "field_components": (plot_field_components, "field_components.png"),
    "field_snapshots": (plot_field_snapshots, "field_snapshots.png"),
    "accel_vs_transverse": (plot_accel_vs_transverse, "accel_vs_transverse.png"),
    "Ez_3d": (plot_Ez_3d, "Ez_3d.png"),
    "Ez_vs_z": (plot_Ez_vs_z, "Ez_vs_z.png"),
}


def plotter_wrapper(results, output_dir, dpi=150, plots=None, plot_options=None):
    """
    Make every requested plot and save it in output_dir.

    results      : dict from lsm_mode_functions.run_lsm_analysis
    output_dir   : folder for the PNGs (created if missing)
    dpi          : resolution of the saved figures
    plots        : names from PLOTS to make (default: all)
    plot_options : optional {plot_name: {kwarg: value}} passed to that plot function
    Returns the list of saved file paths.
    """
    os.makedirs(output_dir, exist_ok=True)
    plots = list(PLOTS) if plots is None else plots
    plot_options = plot_options or {}
    unknown = [p for p in plots if p not in PLOTS]
    if unknown:
        raise ValueError(f"unknown plot(s) {unknown}; choose from {list(PLOTS)}")

    saved = []
    for name in plots:
        func, filename = PLOTS[name]
        saved.append(func(results, os.path.join(output_dir, filename), dpi=dpi, **plot_options.get(name, {})))
    return saved
