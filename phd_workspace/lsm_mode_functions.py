"""
Calculation routines for LSM/LSE modes of a rectangular waveguide, width w, loaded
with a dielectric slab (Xiao, Gai & Sun 2001, Eqs. (3)-(7)).

Geometry: -w/2 < x < w/2, 0 < y < b. Vacuum for 0 < y < a, dielectric (eps_r) for
a < y < b. y = 0 is a symmetry plane, y = b a metal wall.

This module contains only functions (no plotting, no module-level state). Every
function takes the geometry explicitly as a `Guide`. Plotting lives in
lsm_mode_plotter.py; user settings live in lsm_mode_input.py.
"""
from dataclasses import dataclass, replace

import numpy as np
from scipy.constants import c, epsilon_0, mu_0
from scipy.optimize import brentq
import scipy.integrate as integrate


# =============================================================================
# Geometry
# =============================================================================
@dataclass(frozen=True)
class Guide:
    """Waveguide geometry: width w, height b, air height a, slab permittivity eps_r (all SI)."""
    w: float
    b: float
    a: float
    eps_r: float

    @property
    def d(self):
        """Dielectric thickness b - a."""
        return self.b - self.a


def _sqrt(q):
    return np.sqrt(complex(q))


# =============================================================================
# Dispersion relations, cutoff and beta(f)
# =============================================================================
def lsm(guide, s, k):
    """
    Eq. (4a) divided by ky0 so it stays real when ky0 is imaginary:
    ky1 sin(ky1 d) * sin(ky0 a)/ky0 - eps_r cos(ky0 a) cos(ky1 d) = 0
    """
    ky0 = _sqrt(s)
    ky1 = _sqrt(s + (guide.eps_r - 1) * k**2)
    sinc0 = np.sin(ky0 * guide.a) / ky0 if abs(ky0) > 1e-12 else guide.a
    val = (ky1 * np.sin(ky1 * guide.d) * sinc0
           - guide.eps_r * np.cos(ky0 * guide.a) * np.cos(ky1 * guide.d))
    return val.real


def lse(guide, s, k):
    """
    LSE relation  ky0 cot(ky0 a) = -ky1 cot(ky1 d), written pole-free and
    divided by ky0*ky1 so it stays real:
    cos(ky1 d) sin(ky0 a)/ky0 + cos(ky0 a) sin(ky1 d)/ky1 = 0
    (check against the paper's Eq. 4b)
    """
    ky0 = _sqrt(s)
    ky1 = _sqrt(s + (guide.eps_r - 1) * k**2)
    sinc0 = np.sin(ky0 * guide.a) / ky0 if abs(ky0) > 1e-12 else guide.a
    sinc1 = np.sin(ky1 * guide.d) / ky1 if abs(ky1) > 1e-12 else guide.d
    val = np.cos(ky1 * guide.d) * sinc0 + np.cos(ky0 * guide.a) * sinc1
    return val.real


DISPERSION_RELATIONS = {"LSM": lsm, "LSE": lse}


def roots_in_s(guide, func, k, n_grid=4000):
    """All roots s = ky0^2, sorted ascending (= descending beta = n = 1, 2, ...)."""
    s_lo = -(guide.eps_r - 1) * k**2 * (1 - 1e-9)   # beta at its max (field all in dielectric)
    s_hi = k**2                                      # beta = 0 for m = 0
    grid = np.linspace(s_lo, s_hi, n_grid)
    vals = np.array([func(guide, s, k) for s in grid])
    out = []
    for i in range(n_grid - 1):
        if vals[i] == 0:
            out.append(grid[i])
        elif vals[i] * vals[i + 1] < 0:
            out.append(brentq(lambda s: func(guide, s, k), grid[i], grid[i + 1]))
    return out


def cutoff(guide, func, m, n, n_grid=4000):
    """
    Cutoff frequency of mode (m, n).

    At cutoff beta = 0, so from Eq. (3):  s = ky0^2 = k^2 - (m pi / w)^2.
    Putting that s into the dispersion relation leaves an equation in f only:
        G(f) = F(k^2 - (m pi/w)^2, k) = 0
    The n-th sign change of G as f increases is the cutoff of mode (m, n).
    """
    kx2 = (m * np.pi / guide.w) ** 2
    G = lambda f: func(guide, (2 * np.pi * f / c) ** 2 - kx2, 2 * np.pi * f / c)

    # Lowest possible cutoff: ky1 must be real, i.e. eps_r k^2 >= kx^2.
    # (For m = 0 this is 0, so start slightly above it.)
    f_lo = c * np.sqrt(kx2) / (2 * np.pi * np.sqrt(guide.eps_r)) * (1 + 1e-9) + 1e6
    # upper search limit: above where n half-waves fit in the air region.
    f_hi = c / 2 * np.sqrt((m / guide.w) ** 2 + (n / min(guide.a, guide.d)) ** 2) * 3 + 1e9

    grid = np.linspace(f_lo, f_hi, n_grid)
    vals = np.array([G(f) for f in grid])
    found = 0
    for i in range(n_grid - 1):
        if vals[i] * vals[i + 1] < 0:
            found += 1
            if found == n:
                return brentq(G, grid[i], grid[i + 1])
    raise ValueError(f"no cutoff found for mode ({m},{n})")


def beta_curve(guide, func, m, n, f_c, f_max, n_f):
    """
    beta(f) for mode (m, n), sampled from its own cutoff f_c up to f_max.

    Starting exactly at f_c puts a point at beta=0. Points are packed
    more densely near cutoff, where beta=sqrt(f-f_c) rises steeply.
    """
    t = np.linspace(0, 1, n_f)
    freqs = f_c + (f_max - f_c) * t**2      # quadratic spacing
    betas = np.full(n_f, np.nan)
    betas[0] = 0.0
    kx2 = (m * np.pi / guide.w) ** 2
    for i, f in enumerate(freqs[1:], start=1):
        k = 2 * np.pi * f / c
        s_roots = roots_in_s(guide, func, k)
        if len(s_roots) >= n:
            beta2 = k**2 - kx2 - s_roots[n - 1]
            if beta2 > 0:
                betas[i] = np.sqrt(beta2)
    return freqs, betas


def beta_exact(guide, f, m, n, func=lsm):
    """
    beta of mode (m, n) at frequency f from the roots of the dispersion relation.
    """
    k = 2 * np.pi * f / c
    roots = roots_in_s(guide, func, k)
    if len(roots) < n:
        return np.nan
    beta2 = k**2 - (m * np.pi / guide.w) ** 2 - roots[n - 1]
    return np.sqrt(beta2) if beta2 > 0 else np.nan


def find_light_line_crossing(guide, func, m, f_lo, f_hi, n_grid=20000):
    """
    Frequency at which mode m's dispersion curve crosses the light line
    (beta = k). At that point ky0^2 = k^2 - kx^2 - beta^2 = -kx^2 for any f,
    so this fixes s = -kx^2 and searches for the f at which the dispersion
    relation func(s, k) = 0 is satisfied.
    """
    kx2 = (m * np.pi / guide.w) ** 2
    s_sync = -kx2
    k_grid = np.linspace(2 * np.pi * f_lo / c, 2 * np.pi * f_hi / c, n_grid)
    vals = np.array([func(guide, s_sync, k) for k in k_grid])
    for i in range(n_grid - 1):
        if vals[i] * vals[i + 1] < 0:
            k_root = brentq(lambda k: func(guide, s_sync, k), k_grid[i], k_grid[i + 1])
            return k_root * c / (2 * np.pi)
    raise ValueError(f"no light-line crossing found for mode m={m} in "
                     f"[{f_lo/1e9:.2f}, {f_hi/1e9:.2f}] GHz")


def compute_dispersion(guide, modes, f_max, n_f, f_min_factor=0.95, f_min=None):
    """
    Cutoffs and beta(f) curves for a list of (family, m, n) modes, e.g. ("LSM", 1, 1).

    Returns dict with:
        cutoffs : {(family, m, n): f_c}
        curves  : {(family, m, n): (freqs, betas)}
        f_min   : lower edge of the frequency range (f_min_factor * lowest cutoff,
                  unless f_min is given explicitly)
    """
    cutoffs, curves = {}, {}
    for family, m, n in modes:
        cutoffs[(family, m, n)] = cutoff(guide, DISPERSION_RELATIONS[family], m, n)
    if f_min is None:
        f_min = f_min_factor * min(cutoffs.values())
    for family, m, n in modes:
        key = (family, m, n)
        curves[key] = beta_curve(guide, DISPERSION_RELATIONS[family], m, n,
                                 cutoffs[key], f_max, n_f)
    return {"cutoffs": cutoffs, "curves": curves, "f_min": f_min}


def select_operating_point(guide, m, n, f_min, f_max, n_f,
                           use_light_line_frequency=True, user_frequency=None):
    """
    Choose the operating frequency of LSM_mn and return (frequency, beta_mn, f_c).

    Either the frequency where the mode's dispersion curve crosses the light line
    (beta = k, synchronous acceleration of an ultrarelativistic beam) or a
    user-given frequency. beta is interpolated from the dispersion curve.
    """
    if use_light_line_frequency:
        frequency = find_light_line_crossing(guide, lsm, m, f_min, f_max)
    else:
        if user_frequency is None:
            raise ValueError("set user_frequency (Hz) when use_light_line_frequency is False")
        frequency = user_frequency

    # The dispersion solver (roots_in_s / beta_curve) and the mode-continuity assumption it
    # relies on (n-th smallest s root = n-th mode branch) were only validated over
    # f_min-f_max; far outside that band the wrong branch can silently be picked up.
    if not (f_min <= frequency <= 3 * f_max):
        raise ValueError(
            f"f = {frequency/1e9:.3f} GHz is far outside the range the dispersion solver "
            f"was set up for ({f_min/1e9:.2f}-{f_max/1e9:.2f} GHz). Double-check the value "
            f"(e.g. 105e9 = 105 GHz, not 10.5 GHz) - if it's really intended, results "
            f"are not reliable that far from cutoff.")

    f_c = cutoff(guide, lsm, m, n)
    if frequency < f_c:
        raise ValueError(f"f = {frequency/1e9:.3f} GHz is below the LSM{m}{n} cutoff "
                         f"({f_c/1e9:.3f} GHz) - mode does not propagate there")

    freqs, betas = beta_curve(guide, lsm, m, n, f_c, max(f_max, frequency), n_f)
    valid = ~np.isnan(betas)
    beta_mn = np.interp(frequency, freqs[valid], betas[valid])
    return frequency, beta_mn, f_c


# =============================================================================
# Field components (Eqs. 3, 6, 7)
# =============================================================================
def calculate_k_mn(guide, m, beta_mn, f):
    """ky0, ky1 and kc for the given m and beta (Eq. 3)."""
    k = 2 * np.pi * f / c
    ky0 = _sqrt(k**2 - (m * np.pi / guide.w) ** 2 - beta_mn**2)
    ky1 = _sqrt(guide.eps_r * k**2 - (m * np.pi / guide.w) ** 2 - beta_mn**2)
    kcmn = np.sqrt((m * np.pi / guide.w) ** 2 + beta_mn**2)
    return ky0, ky1, kcmn


def calculate_B_mn(guide, m, beta_mn, f, A_mn=1.0):
    """
    B_mn/A_mn from the field-continuity conditions Eq. (7a)/(7b). The two
    are equivalent (their ratio is the dispersion relation, Eq. 4a), but
    whichever denominator is closer to zero is numerically unstable, so use
    whichever of the two has the larger denominator.
    """
    ky0, ky1, _ = calculate_k_mn(guide, m, beta_mn, f)
    denom_7a = ky1 * np.sin(ky1 * guide.d)                  # Eq. (7a)
    denom_7b = guide.eps_r * np.cos(ky1 * guide.d)          # Eq. (7b)
    if abs(denom_7a) >= abs(denom_7b):
        return A_mn * (ky0 * np.cos(ky0 * guide.a)) / denom_7a
    return A_mn * np.sin(ky0 * guide.a) / denom_7b


def get_field_components(guide, m, beta_mn, f, x, y, A_mn=1.0):
    """
    Electric and magnetic field components of the LSM_mn mode (Eq. 6) at
    transverse position (x, y); x, y may be arrays (e.g. from np.meshgrid).
    The z-dependence exp(-j*beta*z) is added separately by add_z_dependence.

    Returns (E_x, E_y, E_z, H_x, H_y, H_z).
    """
    a, b, w, eps_r = guide.a, guide.b, guide.w, guide.eps_r
    ky0, ky1, kcmn = calculate_k_mn(guide, m, beta_mn, f)
    B_mn = calculate_B_mn(guide, m, beta_mn, f, A_mn)
    omega = 2 * np.pi * f
    kx = m * np.pi / w

    region0 = y < a
    phase_s = np.sin(kx * (x + w / 2))
    phase_c = np.cos(kx * (x + w / 2))

    E_x = np.where(region0,
                   A_mn * kx * ky0 * phase_c * np.cos(ky0 * y),
                   B_mn * kx * ky1 * phase_c * np.sin(ky1 * (b - y)))
    E_y = np.where(region0,
                   A_mn * kcmn**2 * phase_s * np.sin(ky0 * y),
                   B_mn * kcmn**2 * phase_s * np.cos(ky1 * (b - y)))
    E_z = np.where(region0,
                   A_mn * (-1j * beta_mn) * ky0 * phase_s * np.cos(ky0 * y),
                   B_mn * (-1j * beta_mn) * ky1 * phase_s * np.sin(ky1 * (b - y)))

    H_x = np.where(region0,
                   -A_mn * omega * epsilon_0 * beta_mn * phase_s * np.sin(ky0 * y),
                   -B_mn * omega * epsilon_0 * eps_r * beta_mn * phase_s * np.cos(ky1 * (b - y)))
    H_y = np.zeros_like(E_x)
    H_z = np.where(region0,
                   A_mn * (1j * omega * epsilon_0) * kx * phase_c * np.sin(ky0 * y),
                   B_mn * (1j * omega * epsilon_0 * eps_r) * kx * phase_c * np.cos(ky1 * (b - y)))

    return E_x, E_y, E_z, H_x, H_y, H_z


def add_z_dependence(fields, beta_mn, z):
    """Multiply each field component by the exp(-j*beta*z) travelling-wave factor."""
    z_dependence = np.exp(-1j * beta_mn * z)
    return [field * z_dependence for field in fields]


def physical_field(F, theta):
    """Time-domain field Re[F exp(+j theta)] at wave phase theta = omega*t - beta*z."""
    return np.real(F * np.exp(1j * theta))


def make_transverse_grid(guide, n_x, n_y):
    """Cross-section grid, -w/2 < x < w/2, 0 < y < b. Returns (x_grid, y_grid, X, Y)."""
    x_grid = np.linspace(-guide.w / 2, guide.w / 2, n_x)
    y_grid = np.linspace(0, guide.b, n_y)
    X, Y = np.meshgrid(x_grid, y_grid)
    return x_grid, y_grid, X, Y


def mirror_in_y(y_grid, F):
    """
    Extend a field on 0 < y < b to -b < y < b using that E_z is even in y
    (y = 0 is a symmetry plane); y = 0 is not duplicated. Returns (y_full, F_full).
    """
    y_full = np.concatenate([-y_grid[:0:-1], y_grid])
    F_full = np.vstack([F[:0:-1], F])
    return y_full, F_full


# =============================================================================
# Derived quantities: forces, travelling wave, energy
# =============================================================================
def compute_force_analysis(guide, fields, x_grid, y_grid, k, theta_acc=0.0, theta_def=np.pi / 2):
    """
    Force per charge on a v = c particle riding the wave at fixed phase theta.
        F_z/q = E_z
        F_x/q = E_x - c*mu0*H_y = E_x
        F_y/q = E_y + c*mu0*H_x
    Panofsky-Wenzel for the synchronous case beta = k:
        F_perp/q = (j/k) grad_perp(E_z)  ->  at theta = pi/2:  F_perp/q = -(1/k) grad_perp(Re E_z)

    Returns dict with Ez_acc (accelerating force at theta_acc), Fx, Fy (transverse force at
    theta_def), the x = 0 line cuts E_part, B_part, pw, and the column index ix0.
    """
    Ex, Ey, Ez, Hx, Hy, Hz = fields
    Fx_ph = Ex - mu_0 * c * Hy
    Fy_ph = Ey + mu_0 * c * Hx

    dy = y_grid[1] - y_grid[0]
    ix0 = int(np.argmin(np.abs(x_grid)))                    # column closest to x = 0
    return {
        "theta_acc": theta_acc,
        "theta_def": theta_def,
        "Ez_acc": physical_field(Ez, theta_acc),
        "Fx": physical_field(Fx_ph, theta_def),
        "Fy": physical_field(Fy_ph, theta_def),
        "ix0": ix0,
        "E_part": physical_field(Ey, theta_def)[:, ix0],
        "B_part": physical_field(mu_0 * c * Hx, theta_def)[:, ix0],
        "pw": -np.gradient(np.real(Ez)[:, ix0], dy) / k,
    }


def compute_Ez_vs_z(guide, m, beta_mn, f, x0, y0, n_z=400, n_beta_lengths=4 * np.pi, A_mn=1.0):
    """
    E_z at the point (x0, y0) along z (travelling wave). The z-dependence is a pure phase
    factor, so |E_z| is constant and only Re(E_z) oscillates.

    z runs from 0 to n_beta_lengths / beta (default 4 pi / beta = two guide wavelengths).
    Returns (z, E_z(z)).
    """
    z = np.linspace(0, n_beta_lengths / beta_mn, n_z)
    E_z_point = get_field_components(guide, m, beta_mn, f, np.array(x0), np.array(y0), A_mn)[2]
    E_z_z = add_z_dependence([np.full_like(z, E_z_point, dtype=complex)], beta_mn, z)[0]
    return z, E_z_z


def compute_field_statistics(guide, m, beta_mn, fields, forces, x_grid, y_grid, Y):
    """
    Summary numbers: E_y fraction left after the magnetic force, vacuum divergence of the
    transverse force, and the share of energy / power in the dielectric layer.
    Returns a dict.
    """
    Ex, Ey, Ez, Hx, Hy, Hz = fields
    dx, dy = x_grid[1] - x_grid[0], y_grid[1] - y_grid[0]
    kx = m * np.pi / guide.w

    # in the vacuum the transverse force has zero divergence: if it focuses in x it defocuses in y
    dFx = np.gradient(forces["Fx"], dx, axis=1)
    div = dFx + np.gradient(forces["Fy"], dy, axis=0)
    vac = (Y < guide.a - 3 * dy)[1:-1, 1:-1]
    rel_div = np.max(np.abs(div[1:-1, 1:-1][vac])) / np.max(np.abs(dFx[1:-1, 1:-1][vac]))

    # where does the energy and power live?
    region0 = Y < guide.a
    we = 0.25 * epsilon_0 * np.where(region0, 1.0, guide.eps_r) * (np.abs(Ex)**2 + np.abs(Ey)**2 + np.abs(Ez)**2)
    wm = 0.25 * mu_0 * (np.abs(Hx)**2 + np.abs(Hy)**2 + np.abs(Hz)**2)
    Sz = 0.5 * np.real(Ex * np.conj(Hy) - Ey * np.conj(Hx))
    share = {label: 100 * np.sum(q[~region0]) / np.sum(q)
             for label, q in (("electric energy", we), ("magnetic energy", wm), ("power flow S_z", Sz))}

    return {
        "Ey_fraction_left": kx**2 / (kx**2 + beta_mn**2),
        "vacuum_div_ratio": rel_div,
        "dielectric_share_percent": share,
    }

# Fourier series expansion-------------------------

def calc_field_amplitude(A_mn, beta_mn, z_length, ky0):
    """
    Calculates field amplitude, alpha in Apsimon 2021, for given user inputs
    """
    field_amplitude = -1j *  A_mn * beta_mn * z_length * ky0
    return field_amplitude

def fourier_a_0_integrand(theta, w, r, ky0):
    """
    Calculates the integrand for the a_0 Fourier coefficient
    """
    first_cosine_arg = ((np.pi / w) * r) * np.cos(theta)
    second_cosine_arg = ky0 * r * np.sin(theta)

    first_cosine = np.cos(first_cosine_arg)
    second_cosine = np.cos(second_cosine_arg)

    integrand = first_cosine * second_cosine
    return np.real(integrand)

def fourier_a_n_integrand(theta, w, r, ky0, n):
    """
    Calculates the integrand for a_n Fourier coefficient
    """
    first_cosine_arg = ((np.pi / w) * r) * np.cos(theta)
    second_cosine_arg = ky0 * r * np.sin(theta)
    third_cosine_arg = n * theta
    
    first_cosine = np.cos(first_cosine_arg)
    second_cosine = np.cos(second_cosine_arg)
    third_cosine = np.cos(third_cosine_arg)
    
    integrand = first_cosine * second_cosine * third_cosine
    return np.real(integrand)

def fourier_b_n_integrand(theta, w, r, ky0, n):
    """
    Calculates the integrand for b_n Fourier coefficient
    """
    first_cosine_arg = ((np.pi / w) * r) * np.cos(theta)
    second_cosine_arg = ky0 * r * np.sin(theta)
    third_arg = n * theta
        
    first_cosine = np.cos(first_cosine_arg)
    second_cosine = np.cos(second_cosine_arg)
    third_sin = np.sin(third_arg)
        
    integrand = first_cosine * second_cosine * third_sin
    return np.real(integrand)

def fourier_a_0(field_amplitude, w, r, ky0):
    """
    Calculates the value of the Fourier coefficient a_0 for the longitudinal field

    ----PUT FIELD_AMPLITUDE IN THE MAIN FUNCTION!!-----
    """
    integral, _ =  integrate.quad(fourier_a_0_integrand, -np.pi, np.pi, args=(w, r, ky0))
    return integral * (field_amplitude / np.pi) 


def fourier_a_n(field_amplitude, w, r, ky0, n):
    """
    Calculates the value of the Fourier coefficient a_n for the longitudinal field
    """
    integral, _  = integrate.quad(fourier_a_n_integrand, -np.pi, np.pi, args=(w, r, ky0, n))
    return integral * (field_amplitude / np.pi) 

def fourier_b_n(field_amplitude, w, r, ky0, n):
    """
    Calculates the value of the Fourier coefficient b_n for the longitudinal field
    """
    integral, _ = integrate.quad(fourier_b_n_integrand, -np.pi, np.pi, args=(w, r, ky0, n))
    return integral * (field_amplitude/np.pi)

def longitudinal_field_expansion_polar(A_mn, beta_mn, z_length, ky0, n, w, r, theta):
    """
    Calculates the multipole potential of the longitudinal field in polar coordinates r, theta, for a single value of
    n that is inputted by the user.

    n is the selected mode
    """
    field_amplitude = calc_field_amplitude(A_mn, beta_mn, z_length, ky0)

    a_n = fourier_a_n(field_amplitude, w, r, ky0, n)
    b_n = fourier_b_n(field_amplitude, w, r, ky0, n)

    v_z = (a_n * np.cos(n*theta)) + (b_n * np.sin(n*theta))

    return v_z


def calc_fourier_poles(A_mn, beta_mn, z_length, ky0, n, w, a, theta):
    """
    Multipole expansion up to order n at a fixed angle theta, swept over r.

    Returns
    -------
    v_n_array : (n, 100) complex array, one row per mode n = 1..n
    a_0_row   : (100,) complex array, the a_0 coefficient at each r
    v_z_total : (100,) complex array, a_0/2 + sum over modes
    r_linspace: (100,) radii, so you can plot against r / w
    """
    r_max = np.sqrt((w / 2) ** 2 + a ** 2)
    r_min = -r_max  # symmetry of DLW
    r_linspace = np.linspace(r_max, r_min, 100)

    field_amplitude = calc_field_amplitude(A_mn, beta_mn, z_length, ky0)

    # a_0 does not depend on n, so compute it once per radius
    a_0_row = np.array([fourier_a_0(field_amplitude, w, radius, ky0)
                        for radius in r_linspace])

    # works for n == 1 as well, so no if/else needed
    n_values = np.arange(1, n + 1)

    rows = []
    for n_i in n_values:
        row = [longitudinal_field_expansion_polar(A_mn, beta_mn, z_length, ky0,
                                                  n_i, w, radius, theta)
               for radius in r_linspace]
        rows.append(row)

    v_n_array = np.array(rows)                        # shape (n, 100)
    v_z_total = a_0_row / 2 + v_n_array.sum(axis=0)   # shape (100,)

    return v_n_array, a_0_row, v_z_total, r_linspace

#-------------------------------------


def format_field_statistics(stats):
    """
    Plain-text lines for the dict returned by compute_field_statistics.
    """
    lines = [
        f"Fraction of E_y left after the magnetic force: kx^2/(kx^2+beta^2) = {stats['Ey_fraction_left']:.2f}",
        f"vacuum: max|div_perp F_perp| / max|dF_x/dx| = {stats['vacuum_div_ratio']:.1e}   "
        f"(~0: cannot focus in both planes)",
        "share in the dielectric layer (a < y < b):",
    ]
    for label, pct in stats["dielectric_share_percent"].items():
        lines.append(f"   {label:16s} {pct:5.1f} %")
    return lines



# =============================================================================
# Verification of the LSM solution. Every check_* returns (ok, list_of_text_lines).
#   1. Boundary / interface conditions (uses the *exact* root of Eq. 4a)
#   2. Maxwell's equations by finite differences, inside each region
#   3. Limiting cases with closed-form answers (eps_r=1, a->0, d->0)
# =============================================================================
def _status(ok):
    return "PASS" if ok else "FAIL"


def analytic_beta(guide, f, m, n, eps):
    """Guide with uniform permittivity eps, symmetry at y=0 and E_x=0 at y=b."""
    k = 2 * np.pi * f / c
    beta2 = eps * k**2 - (m * np.pi / guide.w) ** 2 - ((n - 0.5) * np.pi / guide.b) ** 2
    return np.sqrt(beta2) if beta2 > 0 else np.nan


def analytic_cutoff(guide, m, n, eps):
    return c / (2 * np.pi * np.sqrt(eps)) * np.sqrt(
        (m * np.pi / guide.w) ** 2 + ((n - 0.5) * np.pi / guide.b) ** 2)


def check_boundary_conditions(guide, f, beta, m, tol=1e-6, n_x=201, n_y=401, gap=1e-12):
    """Boundary and interface conditions for the LSM_mn mode at frequency f."""
    a, b, w, eps_r = guide.a, guide.b, guide.w, guide.eps_r
    x = np.linspace(-w / 2, w / 2, n_x)
    y = np.linspace(0, b, n_y)

    # just below / above the dielectric interface
    lo = get_field_components(guide, m, beta, f, x, np.full_like(x, a - gap))   # vacuum side
    hi = get_field_components(guide, m, beta, f, x, np.full_like(x, a + gap))   # dielectric side
    E_scale = max(np.max(np.abs(F)) for F in (lo[:3] + hi[:3]))
    H_scale = max(np.max(np.abs(F)) for F in (lo[3:] + hi[3:]))

    err = lambda u, v, s: np.max(np.abs(u - v)) / s     # relative mismatch, scaled by s
    res = {
        "y=a: E_x continuous": err(lo[0], hi[0], E_scale),
        "y=a: E_z continuous": err(lo[2], hi[2], E_scale),
        "y=a: eps_r*E_y(diel) = E_y(vac)": err(lo[1], eps_r * hi[1], E_scale),
        "y=a: H_x continuous": err(lo[3], hi[3], H_scale),
        "y=a: H_z continuous": err(lo[5], hi[5], H_scale),
    }

    # metal wall at y = b
    top = get_field_components(guide, m, beta, f, x, np.full_like(x, b))
    res["y=b: E_x = 0"] = np.max(np.abs(top[0])) / E_scale
    res["y=b: E_z = 0"] = np.max(np.abs(top[2])) / E_scale

    # side walls at x = -w/2 and x = +w/2
    for xw, label in ((-w / 2, "x=0"), (w / 2, "x=w")):
        side = get_field_components(guide, m, beta, f, np.full_like(y, xw), y)
        res[f"{label}: E_y = 0"] = np.max(np.abs(side[1])) / E_scale
        res[f"{label}: E_z = 0"] = np.max(np.abs(side[2])) / E_scale
        res[f"{label}: H_x = 0"] = np.max(np.abs(side[3])) / H_scale

    lines = [f"1. Boundary / interface conditions (tolerance {tol:.0e})"]
    for key, val in res.items():
        lines.append(f"   {_status(val < tol)}  {key:34s} {val:.2e}")

    # informational: what the y=0 plane is
    bot = get_field_components(guide, m, beta, f, x, np.zeros_like(x))
    lines.append("   info  y=0: max|E_x|, max|E_z|, max|H_z| / scale = "
                 f"{np.max(np.abs(bot[0]))/E_scale:.2e}, {np.max(np.abs(bot[2]))/E_scale:.2e}, "
                 f"{np.max(np.abs(bot[5]))/H_scale:.2e}")
    lines.append("         (E_x, E_z != 0 and H_z = 0 means y=0 is a symmetry plane (PMC), not a metal wall)")
    return all(v < tol for v in res.values()), lines


def maxwell_residuals(guide, f, beta, m, n_pts):
    """
    Residuals of Faraday's law, Ampere's law and div(D) by finite differences on an
    n_pts x n_pts grid in each region (vacuum 0 < y < a, dielectric a < y < b).

    Returns {region: (faraday, ampere, divD)}, each relative to the size of its terms.
    """
    a, b, w, eps_r = guide.a, guide.b, guide.w, guide.eps_r
    omega = 2 * np.pi * f
    jb = 1j * beta
    out = {}
    regions = (("vacuum", 0.0, a * (1 - 1e-9), 1.0),
               ("dielectric", a * (1 + 1e-9), b, eps_r))

    for name, y0, y1, eps in regions:
        x = np.linspace(-w / 2, w / 2, n_pts)
        y = np.linspace(y0, y1, n_pts)
        X, Y = np.meshgrid(x, y)
        Ex, Ey, Ez, Hx, Hy, Hz = get_field_components(guide, m, beta, f, X, Y)
        ddx = lambda F: np.gradient(F, x[1] - x[0], axis=1)
        ddy = lambda F: np.gradient(F, y[1] - y[0], axis=0)

        curlE = (ddy(Ez) + jb * Ey, -jb * Ex - ddx(Ez), ddx(Ey) - ddy(Ex))
        curlH = (ddy(Hz) + jb * Hy, -jb * Hx - ddx(Hz), ddx(Hy) - ddy(Hx))
        E, H = (Ex, Ey, Ez), (Hx, Hy, Hz)
        inner = (slice(1, -1), slice(1, -1))      # drop one-sided edge differences

        faraday = max(np.max(np.abs((curlE[i] + 1j * omega * mu_0 * H[i])[inner])) for i in range(3))
        faraday /= max(np.max(np.abs(omega * mu_0 * H[i])) for i in range(3))
        ampere = max(np.max(np.abs((curlH[i] - 1j * omega * epsilon_0 * eps * E[i])[inner])) for i in range(3))
        ampere /= max(np.max(np.abs(omega * epsilon_0 * eps * E[i])) for i in range(3))
        divD = np.max(np.abs((ddx(Ex) + ddy(Ey) - jb * Ez)[inner]))
        divD /= max(np.max(np.abs(ddx(Ex))), np.max(np.abs(ddy(Ey))), np.max(np.abs(beta * Ez)))
        out[name] = (faraday, ampere, divD)
    return out


def check_maxwell(guide, f, beta, m, tol=1e-3, n_coarse=100, n_fine=200):
    """
    Maxwell's equations by finite differences in each region. The coarse/fine residual
    ratio should be ~4 for second-order finite differences.
    """
    lines = [f"2. Maxwell residuals, relative (tolerance {tol:.0e} at the finest grid)"]
    coarse = maxwell_residuals(guide, f, beta, m, n_coarse)
    fine = maxwell_residuals(guide, f, beta, m, n_fine)
    ok = True
    for region in fine:
        for i, eq in enumerate(("curl E + jw mu0 H", "curl H - jw eps E", "div(eps E)")):
            r1, r2 = coarse[region][i], fine[region][i]
            good = r2 < tol
            ok &= good
            ratio = r1 / r2 if r2 > 0 else np.inf
            lines.append(f"   {_status(good)}  {region:10s} {eq:18s} N={n_coarse}: {r1:.2e}  "
                         f"N={n_fine}: {r2:.2e}  (ratio {ratio:.1f}, ~4 expected)")
    return bool(ok), lines


def check_limits(guide, m, n_modes=(1, 2, 3), f_factors=(1.1, 1.5, 2.5), tol=1e-3):
    """
    Limiting cases of the LSM solution against closed-form results:
        eps_r = 1 (empty guide), a -> 0 (fully filled), d -> 0 (empty again)
    """
    lines = [f"3. Limiting cases vs closed-form results (tolerance {tol:.0e})"]
    ok = True
    cases = (
        ("eps_r = 1 (empty guide)", replace(guide, eps_r=1.0), 1.0),
        ("a -> 0 (fully filled)",   replace(guide, a=1e-8),    guide.eps_r),
        ("d -> 0 (empty again)",    replace(guide, a=guide.b - 1e-8), 1.0),
    )
    for label, g, eps_eff in cases:
        f_ref = analytic_cutoff(g, m, 1, eps_eff)
        worst, nchk = 0.0, 0
        for fac in f_factors:
            f = fac * f_ref
            for n_ in n_modes:
                ref = analytic_beta(g, f, m, n_, eps_eff)
                if np.isnan(ref):
                    continue
                got = beta_exact(g, f, m, n_)
                rel = abs(got - ref) / ref if np.isfinite(got) else np.inf
                worst, nchk = max(worst, rel), nchk + 1
        good = nchk > 0 and worst < tol
        ok &= good
        lines.append(f"   {_status(good)}  beta, {label:26s} worst rel. error {worst:.2e} over {nchk} (f, n) points")

    # cutoff() for the empty guide (its search window assumes a and d are not tiny)
    g = replace(guide, eps_r=1.0)
    worst = max(abs(cutoff(g, lsm, m, n_) - analytic_cutoff(g, m, n_, 1.0)) / analytic_cutoff(g, m, n_, 1.0)
                for n_ in n_modes)
    good = worst < tol
    ok &= good
    lines.append(f"   {_status(good)}  cutoff(), eps_r = 1{'':17s} worst rel. error {worst:.2e}")
    return bool(ok), lines


def check_bounds(guide, m, n, f, beta_interp, use_light_line_frequency=True, tol=1e-2):
    """Consistency of the interpolated beta with bounds for the actual structure."""
    lines = ["3b. Consistency for the actual structure"]
    k = 2 * np.pi * f / c
    ok = True

    beta_ex = beta_exact(guide, f, m, n)
    rel = abs(beta_interp - beta_ex) / beta_ex
    good = rel < tol
    ok &= good
    lines.append(f"   {_status(good)}  interpolated beta (beta_curve) vs exact root: {rel:.2e}")

    lo = analytic_beta(guide, f, m, n, 1.0)
    lo = 0.0 if np.isnan(lo) else lo
    hi = analytic_beta(guide, f, m, n, guide.eps_r)
    good = lo <= beta_ex <= hi
    ok &= good
    lines.append(f"   {_status(good)}  empty-guide beta {lo:.2f} <= beta {beta_ex:.2f} <= filled-guide beta {hi:.2f} rad/m")

    good = beta_ex < np.sqrt(guide.eps_r) * k
    ok &= good
    lines.append(f"   {_status(good)}  beta < sqrt(eps_r)*k  ({beta_ex:.2f} < {np.sqrt(guide.eps_r)*k:.2f})")

    if use_light_line_frequency:
        rel = abs(beta_ex - k) / k
        good = rel < 1e-6
        ok &= good
        lines.append(f"   {_status(good)}  at the light-line frequency beta = k: |beta-k|/k = {rel:.2e}")
    return bool(ok), lines


def run_verification(guide, m, n, frequency, beta_interp, use_light_line_frequency=True,
                     tol_bc=1e-6, tol_maxwell=1e-3, tol_limit=1e-3, tol_beta_interp=1e-2,
                     bc_grid=(201, 401), maxwell_grids=(100, 200),
                     limit_n_modes=(1, 2, 3), limit_f_factors=(1.1, 1.5, 2.5)):
    """
    Run all checks for the selected mode / frequency.

    Returns (all_ok, report_lines): the full plain-text report including a summary.
    """
    beta = beta_exact(guide, frequency, m, n)
    lines = [f"LSM{m}{n} at f = {frequency/1e9:.4f} GHz, exact beta = {beta:.4f} rad/m", ""]

    checks = {}
    for name, (ok, out) in (
        ("boundary / interface conditions",
         check_boundary_conditions(guide, frequency, beta, m, tol_bc, *bc_grid)),
        ("Maxwell residuals",
         check_maxwell(guide, frequency, beta, m, tol_maxwell, *maxwell_grids)),
        ("limiting cases",
         check_limits(guide, m, limit_n_modes, limit_f_factors, tol_limit)),
        ("consistency / bounds",
         check_bounds(guide, m, n, frequency, beta_interp, use_light_line_frequency, tol_beta_interp)),
    ):
        checks[name] = ok
        lines += out + [""]

    lines.append("Summary:")
    for name, ok in checks.items():
        lines.append(f"   {_status(ok)}  {name}")
    return all(checks.values()), lines





# =============================================================================
# run whole analysis
# =============================================================================
def run_lsm_analysis(guide, dispersion_modes, f_max, n_f, mode, use_light_line_frequency,
                     user_frequency, A_mn, n_x, n_y, f_min_factor=0.95, f_min=None,
                     z_point=None, n_z=400, z_periods=2,
                     fourier_modes_n=4, theta_fourier=0.284, z_length=1.0):
    """
    Run the whole LSM calculation and return a results dict for lsm_mode_plotter.

    guide            : Guide geometry
    dispersion_modes : list of (family, m, n) for the dispersion plot
    mode             : (m, n) of the LSM mode used for the field plots
    z_point          : (x0, y0) for the E_z(z) trace; default (0, a/2)
    z_periods        : number of guide wavelengths shown in the E_z(z) trace
    fourier_modes_n  : highest multipole order n in the Fourier expansion (n = 1..fourier_modes_n)
    theta_fourier    : fixed polar angle [rad] at which the expansion is swept over r
    z_length         : structure length used in the field amplitude alpha
    """
    m, n = mode
    disp = compute_dispersion(guide, dispersion_modes, f_max, n_f, f_min_factor, f_min)
    f_min = disp["f_min"]

    frequency, beta_mn, f_c = select_operating_point(
        guide, m, n, f_min, f_max, n_f, use_light_line_frequency, user_frequency)
    k = 2 * np.pi * frequency / c

    x_grid, y_grid, X, Y = make_transverse_grid(guide, n_x, n_y)
    fields = get_field_components(guide, m, beta_mn, frequency, X, Y, A_mn)
    forces = compute_force_analysis(guide, fields, x_grid, y_grid, k)
    stats = compute_field_statistics(guide, m, beta_mn, fields, forces, x_grid, y_grid, Y)

    ky0, _, _ = calculate_k_mn(guide, m, beta_mn, frequency)
    v_n, a_0, v_total, r = calc_fourier_poles(
        A_mn, beta_mn, z_length, ky0, fourier_modes_n, guide.w, guide.a, theta_fourier)

    x0, y0 = z_point if z_point is not None else (0.0, guide.a / 2)
    z, E_z_z = compute_Ez_vs_z(guide, m, beta_mn, frequency, x0, y0, n_z, 2 * np.pi * z_periods, A_mn)

    return {
        "guide": guide, "m": m, "n": n, "A_mn": A_mn,
        "f_max": f_max, "f_min": f_min, "k": k,
        "cutoffs": disp["cutoffs"], "curves": disp["curves"],
        "frequency": frequency, "beta_mn": beta_mn, "f_c": f_c,
        "use_light_line_frequency": use_light_line_frequency,
        "x_grid": x_grid, "y_grid": y_grid, "X": X, "Y": Y,
        "fields": fields, "forces": forces, "stats": stats,
        "z": z, "E_z_z": E_z_z, "z_point": (x0, y0),
        "multipole_components_2d_array": v_n, "total_multipole_components": v_total,
        "polar_r_multipole": r, "multipole_a0_half": a_0 / 2,
        "theta_fourier": theta_fourier,
    }


def summary_lines(results):
    """Plain-text header describing the operating point of a results dict."""
    g, m, n = results["guide"], results["m"], results["n"]
    lines = [f"w={g.w*1e3:.2f} mm, b={g.b*1e3:.2f} mm, a={g.a*1e3:.2f} mm, eps_r={g.eps_r}"]
    for (family, mm, nn), fc in results["cutoffs"].items():
        lines.append(f"{family}{mm}{nn} cutoff = {fc/1e9:.3f} GHz")
    lines.append(f"f range: {results['f_min']/1e9:.2f} - {results['f_max']/1e9:.2f} GHz")
    if results["use_light_line_frequency"]:
        lines.append(f"LSM{m}{n} crosses the light line at f = {results['frequency']/1e9:.4f} GHz")
    lines.append(f"LSM{m}{n}: beta = {results['beta_mn']:.2f} rad/m at f = {results['frequency']/1e9:.2f} GHz")
    return lines


def write_text_report(path, lines):
    """Write a list of text lines to `path`."""
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
