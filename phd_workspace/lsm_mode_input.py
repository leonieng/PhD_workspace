"""
User input for the LSM mode calculation. Edit the variables below, then run:

    python lsm_mode_input.py

Plots and the verification report are written to OUTPUT_DIR.
All lengths in metres, frequencies in Hz.
"""
import numpy as np

from lsm_mode_functions import (Guide, format_field_statistics, run_lsm_analysis,
                                run_verification, summary_lines, write_text_report)
from lsm_mode_plotter import plotter_wrapper

# =============================================================================
# 1. Waveguide geometry   (-w/2 < x < w/2, 0 < y < b; dielectric fills a < y < b)
# =============================================================================
w = 23e-3        # guide width in x                                   [m]
b = 5e-3       # guide height in y (metal wall at y = b)            [m]
a = 3e-3       # height of the air region (dielectric from y = a)   [m]
eps_r = 10       # relative permittivity of the slab                  [-]

# =============================================================================
# 2. Dispersion curves
# =============================================================================
# Modes to plot, as (family, m, n). family is "LSM" or "LSE"; m = x index, n = branch.
DISPERSION_MODES = [("LSM", 1, 1), ("LSM", 2, 1)]

f_max = 15e9          # upper end of the frequency range                              [Hz]
n_f = 100             # points per dispersion curve (more = smoother, slower)         [-]
f_min_factor = 0.95   # lower end = f_min_factor * (lowest cutoff of the modes above)  [-]
f_min = 1e9          # or set an explicit lower end [Hz]; None -> use f_min_factor

# =============================================================================
# 3. Mode used for the field plots (LSM_mn) and its operating frequency
# =============================================================================
m, n = 1, 1                        # mode indices of the LSM mode to plot the fields of

use_light_line_frequency = True    # True: operate where the mode's curve crosses the light
                                   #       line (beta = k, synchronous with a v = c beam)
                                   # False: use user_frequency below
user_frequency = None              # [Hz], only used if use_light_line_frequency is False

A_mn = 1.0                         # field amplitude / normalisation constant (arbitrary)

# =============================================================================
# 4. Field-plot settings
# =============================================================================
n_x, n_y = 200, 200                # cross-section grid points in x and y

# wave phases omega*t - beta*z for the field-snapshot plot
snapshot_thetas = [0, np.pi / 2, np.pi, 3 * np.pi / 2]

quiver_stride = 12                 # arrow spacing (in grid points) in the transverse-force map
pw_stride = 8                      # marker spacing for the Panofsky-Wenzel points

# E_z(z) trace
z_point = None                     # (x0, y0) [m] where E_z is sampled; None -> (0, a/2)
n_z = 400                          # points along z
z_periods = 2                      # number of guide wavelengths shown

# Fourier multipole expansion of the longitudinal field
fourier_modes_n = 4                # highest multipole order n (components n = 1..fourier_modes_n)
theta_fourier = 0.284              # fixed polar angle [rad] at which the expansion is swept over r
z_length = 1.0                     # structure length in the field amplitude alpha  [m]

# =============================================================================
# 5. Output
# =============================================================================
OUTPUT_DIR = "lsm_mode_outputs"    # all plots and the report go here
dpi = 150                          # resolution of the saved figures
report_name = "verification.txt"   # verification report (written to OUTPUT_DIR)

# Which plots to make. Available:
#   "dispersion", "field_components", "field_snapshots",
#   "accel_vs_transverse", "Ez_3d", "Ez_vs_z", "fourier_multipoles"
PLOTS_TO_MAKE = ["dispersion", "field_components", "field_snapshots",
                 "accel_vs_transverse", "Ez_3d", "Ez_vs_z", "fourier_multipoles"]

# =============================================================================
# 6. Verification
# =============================================================================
run_checks = True                  # set False to skip the verification
tol_boundary = 1e-6                # boundary / interface mismatch, relative to max|field|
tol_maxwell = 1e-3                 # finite-difference Maxwell residual at the finest grid
tol_limit = 1e-3                   # relative error vs closed-form limiting cases
tol_beta_interp = 1e-2             # interpolated beta vs exact root
boundary_grid = (201, 401)         # (points in x, points in y) for the boundary checks
maxwell_grids = (100, 200)         # (coarse, fine) grids for the Maxwell residuals
limit_n_modes = (1, 2, 3)          # n branches checked in the limiting cases
limit_f_factors = (1.1, 1.5, 2.5)  # frequencies as multiples of the analytic cutoff


def main():
    import os

    guide = Guide(w=w, b=b, a=a, eps_r=eps_r)
    results = run_lsm_analysis(
        guide, DISPERSION_MODES, f_max, n_f, (m, n), use_light_line_frequency, user_frequency,
        A_mn, n_x, n_y, f_min_factor=f_min_factor, f_min=f_min,
        z_point=z_point, n_z=n_z, z_periods=z_periods,
        fourier_modes_n=fourier_modes_n, theta_fourier=theta_fourier, z_length=z_length)

    saved = plotter_wrapper(
        results, OUTPUT_DIR, dpi=dpi, plots=PLOTS_TO_MAKE,
        plot_options={
            "field_snapshots": {"thetas": snapshot_thetas},
            "accel_vs_transverse": {"quiver_stride": quiver_stride, "pw_stride": pw_stride},
        })

    # report: operating point, field statistics, verification
    lines = summary_lines(results) + [""] + format_field_statistics(results["stats"]) + [""]
    if run_checks:
        _, check_lines = run_verification(
            guide, m, n, results["frequency"], results["beta_mn"], use_light_line_frequency,
            tol_boundary, tol_maxwell, tol_limit, tol_beta_interp,
            boundary_grid, maxwell_grids, limit_n_modes, limit_f_factors)
        lines += ["Verification", "============", ""] + check_lines
    report_path = os.path.join(OUTPUT_DIR, report_name)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    write_text_report(report_path, lines)

    print(f"saved {len(saved)} plots and {report_path}")


if __name__ == "__main__":
    main()
