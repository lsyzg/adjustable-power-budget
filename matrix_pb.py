"""Matrix-based (finite-difference) photonic mode solver.

Independent architecture from power_budget.py: instead of the 1D x 1D
effective-index-method (EIM) approximation, this discretizes the waveguide
cross-section onto a grid and solves the resulting sparse eigenvalue
problem directly -- the same class of approach Lumerical's FDE uses.
"""

import math
import os
import re
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

C_CONST = 299792458.0

DEFAULT_SOURCE_SPECTRUM_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "lumerical", "source_numpy_mats.txt"
)


# ---------------------------------------------------------------------------
# 2D scalar (semi-vectorial) finite-difference eigenmode solver
# ---------------------------------------------------------------------------

def _laplacian_1d(n, d):
    diagonals = [np.full(n, -2.0), np.ones(n - 1), np.ones(n - 1)]
    return sp.diags(diagonals, [0, -1, 1], format="csr") / d ** 2


def _build_index_grid(n_core, n_clad, width_um, height_um,
                       domain_width_um, domain_height_um, dx_um, dy_um):
    nx = max(int(round(domain_width_um / dx_um)), 3)
    ny = max(int(round(domain_height_um / dy_um)), 3)
    xs = (np.arange(nx) - (nx - 1) / 2) * dx_um
    ys = (np.arange(ny) - (ny - 1) / 2) * dy_um
    x_grid, y_grid = np.meshgrid(xs, ys, indexing="ij")
    n_grid = np.where(
        (np.abs(x_grid) <= width_um / 2) & (np.abs(y_grid) <= height_um / 2),
        n_core, n_clad,
    )
    return xs, ys, n_grid


def _build_helmholtz_matrix(n_grid, dx_um, dy_um, k0):
    nx, ny = n_grid.shape
    Dxx = _laplacian_1d(nx, dx_um)
    Dyy = _laplacian_1d(ny, dy_um)
    Ix = sp.identity(nx, format="csr")
    Iy = sp.identity(ny, format="csr")
    laplacian = sp.kron(Dxx, Iy, format="csr") + sp.kron(Ix, Dyy, format="csr")
    index_term = sp.diags((k0 * n_grid.reshape(-1)) ** 2)
    return (laplacian + index_term).tocsr()


def _confinement_from_field(psi, n_grid, n_core):
    intensity = psi ** 2
    core_mask = n_grid == n_core
    total = intensity.sum()
    return intensity[core_mask].sum() / total


def solve_mode_matrix(n_core, n_clad, width_um, height_um, wavelength_um,
                       domain_scale=10.0, resolution_um=0.02):
    """Solve for the fundamental guided mode via a 2D finite-difference
    eigenvalue matrix, instead of the effective-index-method approximation.

    Returns a dict with n_eff, confinement, the 2D field, and the grid axes.
    """
    domain_width_um = width_um * domain_scale
    domain_height_um = height_um * domain_scale

    xs, ys, n_grid = _build_index_grid(
        n_core, n_clad, width_um, height_um,
        domain_width_um, domain_height_um, resolution_um, resolution_um,
    )

    k0 = 2 * math.pi / wavelength_um
    matrix_a = _build_helmholtz_matrix(n_grid, resolution_um, resolution_um, k0)

    eigval, eigvec = spla.eigsh(matrix_a, k=1, which="LA")
    beta2 = eigval[0]
    if beta2 <= 0:
        raise ValueError("No guided mode found (eigenvalue <= 0); check geometry/indices.")

    n_eff = math.sqrt(beta2) / k0
    if not (n_clad < n_eff < n_core):
        raise ValueError(
            f"Solved n_eff={n_eff:.4f} outside guided range ({n_clad}, {n_core}); "
            "domain likely too small or grid too coarse for this geometry."
        )

    psi = eigvec[:, 0].reshape(n_grid.shape)
    confinement = _confinement_from_field(psi, n_grid, n_core)

    return {
        "n_eff": n_eff,
        "confinement": confinement,
        "field": psi,
        "x_um": xs,
        "y_um": ys,
        "n_grid": n_grid,
    }


# ---------------------------------------------------------------------------
# Spectrally-weighted n_eff over the Gaussian source spectrum
# ---------------------------------------------------------------------------

_FLOAT_RE = re.compile(r"[-+]?\d*\.\d+e[-+]\d+")


def load_source_spectrum(path=None):
    """Parses the frequency / power-spectrum arrays out of the Lumerical
    source dump (numpy's multi-line bracketed print format)."""
    if path is None:
        path = DEFAULT_SOURCE_SPECTRUM_PATH
    text = Path(path).read_text()
    freq_section, power_section = text.split("power spectrum")
    freq_section = freq_section.split("frequency")[1]
    freq_hz = np.array([float(x) for x in _FLOAT_RE.findall(freq_section)])
    power_spectrum = np.array([float(x) for x in _FLOAT_RE.findall(power_section)])
    return freq_hz, power_spectrum


def select_spectral_samples(freq_hz, power_spectrum, n_samples=18, threshold_db=20.0):
    """Downselects to the samples within threshold_db of peak power, then
    evenly subsamples down to n_samples -- full 512-point resolution isn't
    worth the 512x solver cost when the tails contribute negligibly to a
    trapz-weighted average anyway."""
    peak = power_spectrum.max()
    mask = power_spectrum >= peak * 10 ** (-threshold_db / 10)
    f_sel = freq_hz[mask]
    p_sel = power_spectrum[mask]

    if len(f_sel) > n_samples:
        idx = np.unique(np.linspace(0, len(f_sel) - 1, n_samples).round().astype(int))
        f_sel = f_sel[idx]
        p_sel = p_sel[idx]

    order = np.argsort(f_sel)
    return f_sel[order], p_sel[order]


def solve_mode_matrix_spectral(n_core, n_clad, width_um, height_um,
                                freq_hz=None, power_spectrum=None,
                                n_samples=18, threshold_db=20.0,
                                domain_scale=10.0, resolution_um=0.02):
    """Runs solve_mode_matrix at samples drawn from the Gaussian source
    spectrum and returns the power-spectrum-weighted n_eff/confinement,
    matching how a broadband Lumerical EME run implicitly averages over
    its pulse."""
    if freq_hz is None or power_spectrum is None:
        freq_hz, power_spectrum = load_source_spectrum()

    f_sel, p_sel = select_spectral_samples(freq_hz, power_spectrum, n_samples, threshold_db)

    n_effs = np.empty(len(f_sel))
    confinements = np.empty(len(f_sel))
    for i, f in enumerate(f_sel):
        wavelength_um = (C_CONST / f) * 1e6
        result = solve_mode_matrix(
            n_core, n_clad, width_um, height_um, wavelength_um,
            domain_scale=domain_scale, resolution_um=resolution_um,
        )
        n_effs[i] = result["n_eff"]
        confinements[i] = result["confinement"]

    weight_norm = np.trapezoid(p_sel, f_sel)
    n_eff_weighted = np.trapezoid(n_effs * p_sel, f_sel) / weight_norm
    confinement_weighted = np.trapezoid(confinements * p_sel, f_sel) / weight_norm

    return {
        "n_eff_weighted": n_eff_weighted,
        "confinement_weighted": confinement_weighted,
        "freq_hz": f_sel,
        "n_eff": n_effs,
        "confinement": confinements,
        "power_spectrum": p_sel,
    }


# ---------------------------------------------------------------------------
# Full-vectorial (Hx, Hy) finite-difference eigenmode solver
#
# The scalar solver above can't represent the vectorial corner/polarization
# coupling that matters for small, high-index-contrast, near-square
# waveguides -- validated by comparing both against digitized Lumerical FDE
# data (see matrix_pb.py's development history / plan notes). This section
# fixes that by solving the coupled transverse-H-field system instead of a
# single scalar field.
#
# The coefficient formulas below are ported (adapted for the isotropic,
# uniform-grid, hard-Dirichlet-boundary case used here) from the reference
# algorithm and implementation:
#
#   A. B. Fallahkhair, K. S. Li, T. E. Murphy, "Vector Finite Difference
#   Modesolver for Anisotropic Dielectric Waveguides", J. Lightwave
#   Technol. 26(11), 1423-1431 (2008). https://doi.org/10.1109/JLT.2008.923643
#
#   Reference implementation: github.com/thomas-e-murphy/modesolver
#   (modesolver/wgmodes.py), (c) 2025 Thomas E. Murphy, MIT License.
#
# Ported verbatim rather than hand-simplified for the isotropic case, to
# avoid introducing new transcription bugs -- isotropy is achieved simply by
# passing epsxy = epsyx = 0 at the call site.
# ---------------------------------------------------------------------------

def _build_index_grid_cells(n_core, n_clad, width_um, height_um,
                             domain_width_um, domain_height_um, dx_um, dy_um):
    """Like _build_index_grid, but cell-centered (shape (ny_cells, nx_cells)),
    matching the vectorial solver's convention: permittivity is defined per
    cell, and Hx/Hy live at the (ny_cells+1) x (nx_cells+1) vertices."""
    nx_cells = max(int(round(domain_width_um / dx_um)), 3)
    ny_cells = max(int(round(domain_height_um / dy_um)), 3)
    xs = (np.arange(nx_cells) - (nx_cells - 1) / 2) * dx_um
    ys = (np.arange(ny_cells) - (ny_cells - 1) / 2) * dy_um
    x_grid, y_grid = np.meshgrid(xs, ys)  # shape (ny_cells, nx_cells)
    n_grid = np.where(
        (np.abs(x_grid) <= width_um / 2) & (np.abs(y_grid) <= height_um / 2),
        n_core, n_clad,
    )
    return xs, ys, n_grid


def _assemble_vector_matrices(epsxx, epsyy, epszz, dx_um, dy_um, k0):
    """Assembles the sparse A (2N x 2N) and B (N x 2N) matrices for the
    coupled Hx/Hy vectorial eigenvalue problem, on a uniform grid, isotropic
    media (epsxy = epsyx = 0), hard-Dirichlet boundary (field = 0 outside
    domain). See module-level comment above for the algorithm reference.
    """
    epsxy = np.zeros_like(epsxx)
    epsyx = np.zeros_like(epsxx)

    ny_cells, nx_cells = epsxx.shape
    ny, nx = ny_cells + 1, nx_cells + 1
    N = nx * ny

    epsxx = np.pad(epsxx, pad_width=1, mode="edge")
    epsyy = np.pad(epsyy, pad_width=1, mode="edge")
    epszz = np.pad(epszz, pad_width=1, mode="edge")
    epsxy = np.pad(epsxy, pad_width=1, mode="edge")
    epsyx = np.pad(epsyx, pad_width=1, mode="edge")

    dx = np.full(nx + 1, dx_um)
    dy = np.full(ny + 1, dy_um)

    n = np.repeat(dy[1:ny + 1], nx)
    s = np.repeat(dy[0:ny], nx)
    e = np.tile(dx[1:nx + 1], ny)
    w = np.tile(dx[0:nx], ny)

    exx1 = np.asarray(epsxx[1:ny + 1, 0:nx]).ravel(order="C")
    exx2 = np.asarray(epsxx[0:ny, 0:nx]).ravel(order="C")
    exx3 = np.asarray(epsxx[0:ny, 1:nx + 1]).ravel(order="C")
    exx4 = np.asarray(epsxx[1:ny + 1, 1:nx + 1]).ravel(order="C")

    eyy1 = np.asarray(epsyy[1:ny + 1, 0:nx]).ravel(order="C")
    eyy2 = np.asarray(epsyy[0:ny, 0:nx]).ravel(order="C")
    eyy3 = np.asarray(epsyy[0:ny, 1:nx + 1]).ravel(order="C")
    eyy4 = np.asarray(epsyy[1:ny + 1, 1:nx + 1]).ravel(order="C")

    ezz1 = np.asarray(epszz[1:ny + 1, 0:nx]).ravel(order="C")
    ezz2 = np.asarray(epszz[0:ny, 0:nx]).ravel(order="C")
    ezz3 = np.asarray(epszz[0:ny, 1:nx + 1]).ravel(order="C")
    ezz4 = np.asarray(epszz[1:ny + 1, 1:nx + 1]).ravel(order="C")

    exy1 = np.asarray(epsxy[1:ny + 1, 0:nx]).ravel(order="C")
    exy2 = np.asarray(epsxy[0:ny, 0:nx]).ravel(order="C")
    exy3 = np.asarray(epsxy[0:ny, 1:nx + 1]).ravel(order="C")
    exy4 = np.asarray(epsxy[1:ny + 1, 1:nx + 1]).ravel(order="C")

    eyx1 = np.asarray(epsyx[1:ny + 1, 0:nx]).ravel(order="C")
    eyx2 = np.asarray(epsyx[0:ny, 0:nx]).ravel(order="C")
    eyx3 = np.asarray(epsyx[0:ny, 1:nx + 1]).ravel(order="C")
    eyx4 = np.asarray(epsyx[1:ny + 1, 1:nx + 1]).ravel(order="C")

    ns21 = n * eyy2 + s * eyy1
    ns34 = n * eyy3 + s * eyy4
    ew14 = e * exx1 + w * exx4
    ew23 = e * exx2 + w * exx3

    axxn = (
        ((2 * eyy4 * e - eyx4 * n) * (eyy3 / ezz4) / ns34)
        + ((2 * eyy1 * w + eyx1 * n) * (eyy2 / ezz1) / ns21)
    ) / (n * (e + w))

    axxs = (
        ((2 * eyy3 * e + eyx3 * s) * (eyy4 / ezz3) / ns34)
        + ((2 * eyy2 * w - eyx2 * s) * (eyy1 / ezz2) / ns21)
    ) / (s * (e + w))

    ayye = (
        (2 * n * exx4 - e * exy4) * exx1 / ezz4 / e / ew14 / (n + s)
        + (2 * s * exx3 + e * exy3) * exx2 / ezz3 / e / ew23 / (n + s)
    )

    ayyw = (
        (2 * exx1 * n + exy1 * w) * exx4 / ezz1 / w / ew14 / (n + s)
        + (2 * exx2 * s - exy2 * w) * exx3 / ezz2 / w / ew23 / (n + s)
    )

    axxe = 2 / (e * (e + w)) + (eyy4 * eyx3 / ezz3 - eyy3 * eyx4 / ezz4) / (e + w) / ns34
    axxw = 2 / (w * (e + w)) + (eyy2 * eyx1 / ezz1 - eyy1 * eyx2 / ezz2) / (e + w) / ns21
    ayyn = 2 / (n * (n + s)) + (exx4 * exy1 / ezz1 - exx1 * exy4 / ezz4) / (n + s) / ew14
    ayys = 2 / (s * (n + s)) + (exx2 * exy3 / ezz3 - exx3 * exy2 / ezz2) / (n + s) / ew23

    axxne = +eyx4 * eyy3 / ezz4 / (e + w) / ns34
    axxse = -eyx3 * eyy4 / ezz3 / (e + w) / ns34
    axxnw = -eyx1 * eyy2 / ezz1 / (e + w) / ns21
    axxsw = +eyx2 * eyy1 / ezz2 / (e + w) / ns21

    ayyne = +exy4 * exx1 / ezz4 / (n + s) / ew14
    ayyse = -exy3 * exx2 / ezz3 / (n + s) / ew23
    ayynw = -exy1 * exx4 / ezz1 / (n + s) / ew14
    ayysw = +exy2 * exx3 / ezz2 / (n + s) / ew23

    axxp = (
        -axxn - axxs - axxe - axxw - axxne - axxse - axxnw - axxsw
        + k0 ** 2 * (n + s) * (eyy4 * eyy3 * e / ns34 + eyy1 * eyy2 * w / ns21) / (e + w)
    )

    ayyp = (
        -ayyn - ayys - ayye - ayyw - ayyne - ayyse - ayynw - ayysw
        + k0 ** 2 * (e + w) * (exx1 * exx4 * n / ew14 + exx2 * exx3 * s / ew23) / (n + s)
    )

    axyn = (
        (eyy3 * eyy4 / ezz4 / ns34 - eyy2 * eyy1 / ezz1 / ns21
         + s * (eyy2 * eyy4 - eyy1 * eyy3) / ns21 / ns34)
        / (e + w)
    )

    axys = (
        (eyy1 * eyy2 / ezz2 / ns21 - eyy4 * eyy3 / ezz3 / ns34
         + n * (eyy2 * eyy4 - eyy1 * eyy3) / ns21 / ns34)
        / (e + w)
    )

    ayxe = (
        (exx1 * exx4 / ezz4 / ew14 - exx2 * exx3 / ezz3 / ew23
         + w * (exx2 * exx4 - exx1 * exx3) / ew23 / ew14)
        / (n + s)
    )

    ayxw = (
        (exx3 * exx2 / ezz2 / ew23 - exx4 * exx1 / ezz1 / ew14
         + e * (exx4 * exx2 - exx1 * exx3) / ew23 / ew14)
        / (n + s)
    )

    axye = (
        (eyy4 * (1 - eyy3 / ezz3) - eyy3 * (1 - eyy4 / ezz4)) / ns34 / (e + w)
        - 2 * (
            eyx1 * eyy2 / ezz1 * n * w / ns21
            + eyx2 * eyy1 / ezz2 * s * w / ns21
            + eyx4 * eyy3 / ezz4 * n * e / ns34
            + eyx3 * eyy4 / ezz3 * s * e / ns34
            + eyy1 * eyy2 * (1 / ezz1 - 1 / ezz2) * w ** 2 / ns21
            + eyy3 * eyy4 * (1 / ezz4 - 1 / ezz3) * e * w / ns34
        ) / e / (e + w) ** 2
    )

    axyw = (
        (eyy2 * (1 - eyy1 / ezz1) - eyy1 * (1 - eyy2 / ezz2)) / ns21 / (e + w)
        - 2 * (
            eyx4 * eyy3 / ezz4 * n * e / ns34
            + eyx3 * eyy4 / ezz3 * s * e / ns34
            + eyx1 * eyy2 / ezz1 * n * w / ns21
            + eyx2 * eyy1 / ezz2 * s * w / ns21
            + eyy4 * eyy3 * (1 / ezz3 - 1 / ezz4) * e ** 2 / ns34
            + eyy2 * eyy1 * (1 / ezz2 - 1 / ezz1) * w * e / ns21
        ) / w / (e + w) ** 2
    )

    ayxn = (
        (exx4 * (1 - exx1 / ezz1) - exx1 * (1 - exx4 / ezz4)) / ew14 / (n + s)
        - 2 * (
            exy3 * exx2 / ezz3 * e * s / ew23
            + exy2 * exx3 / ezz2 * w * s / ew23
            + exy4 * exx1 / ezz4 * e * n / ew14
            + exy1 * exx4 / ezz1 * w * n / ew14
            + exx3 * exx2 * (1 / ezz3 - 1 / ezz2) * s ** 2 / ew23
            + exx1 * exx4 * (1 / ezz4 - 1 / ezz1) * n * s / ew14
        ) / n / (n + s) ** 2
    )

    ayxs = (
        (exx2 * (1 - exx3 / ezz3) - exx3 * (1 - exx2 / ezz2)) / ew23 / (n + s)
        - 2 * (
            exy4 * exx1 / ezz4 * e * n / ew14
            + exy1 * exx4 / ezz1 * w * n / ew14
            + exy3 * exx2 / ezz3 * e * s / ew23
            + exy2 * exx3 / ezz2 * w * s / ew23
            + exx4 * exx1 * (1 / ezz1 - 1 / ezz4) * n ** 2 / ew14
            + exx2 * exx3 * (1 / ezz2 - 1 / ezz3) * s * n / ew23
        ) / s / (n + s) ** 2
    )

    axyne = +eyy3 * (1 - eyy4 / ezz4) / (e + w) / ns34
    axyse = -eyy4 * (1 - eyy3 / ezz3) / (e + w) / ns34
    axynw = -eyy2 * (1 - eyy1 / ezz1) / (e + w) / ns21
    axysw = +eyy1 * (1 - eyy2 / ezz2) / (e + w) / ns21

    ayxne = +exx1 * (1 - exx4 / ezz4) / (n + s) / ew14
    ayxse = -exx2 * (1 - exx3 / ezz3) / (n + s) / ew23
    ayxnw = -exx4 * (1 - exx1 / ezz1) / (n + s) / ew14
    ayxsw = +exx3 * (1 - exx2 / ezz2) / (n + s) / ew23

    axyp = (
        -(axyn + axys + axye + axyw + axyne + axyse + axynw + axysw)
        - k0 ** 2 * (
            w * (n * eyx1 * eyy2 + s * eyx2 * eyy1) / ns21
            + e * (s * eyx3 * eyy4 + n * eyx4 * eyy3) / ns34
        ) / (e + w)
    )

    ayxp = (
        -(ayxn + ayxs + ayxe + ayxw + ayxne + ayxse + ayxnw + ayxsw)
        - k0 ** 2 * (
            n * (w * exy1 * exx4 + e * exy4 * exx1) / ew14
            + s * (w * exy2 * exx3 + e * exy3 * exx2) / ew23
        ) / (n + s)
    )

    bh12 = e * w / ((2 * e + 2 * w) * (ezz1 * ezz2 * s / (2 * eyy2) + ezz1 * ezz2 * n / (2 * eyy1)))
    bh34 = -e * w / ((2 * e + 2 * w) * (-ezz3 * ezz4 * n / (2 * eyy4) - ezz3 * ezz4 * s / (2 * eyy3)))
    bv14 = -n * s / ((2 * n + 2 * s) * (-e * ezz1 * ezz4 / (2 * exx4) - ezz1 * ezz4 * w / (2 * exx1)))
    bv23 = n * s / ((2 * n + 2 * s) * (-e * ezz2 * ezz3 / (2 * exx3) - ezz2 * ezz3 * w / (2 * exx2)))

    bxn = (
        bh12 * (-eyx1 * ezz2 / (2 * eyy1 * w) - ezz2 / n)
        + bh34 * (ezz3 / n - eyx4 * ezz3 / (2 * e * eyy4))
        + bv14 * (
            e * exy4 * ezz1 * s / (exx4 * n ** 3 + exx4 * n ** 2 * s)
            - e * exy4 * ezz1 / (exx4 * n ** 2)
            + exy1 * ezz4 * s * w / (exx1 * n ** 3 + exx1 * n ** 2 * s)
            - ezz1 * s / (n ** 2 + n * s)
            + ezz1 / (2 * n)
            + ezz4 * s / (n ** 2 + n * s)
            - ezz4 / (2 * n)
            - ezz1 * ezz4 / (2 * exx4 * n)
            - exy1 * ezz4 * w / (exx1 * n ** 2)
            + ezz1 * ezz4 / (2 * exx1 * n)
        )
        + bv23 * (
            -e * exy3 * ezz2 / (exx3 * n ** 2 + exx3 * n * s)
            - exy2 * ezz3 * w / (exx2 * n ** 2 + exx2 * n * s)
            - ezz2 * s / (n ** 2 + n * s)
            + ezz3 * s / (n ** 2 + n * s)
        )
    )
    bxs = (
        bh12 * (eyx2 * ezz1 / (2 * eyy2 * w) - ezz1 / s)
        + bh34 * (ezz4 / s + eyx3 * ezz4 / (2 * e * eyy3))
        + bv14 * (
            -e * exy4 * ezz1 / (exx4 * n * s + exx4 * s ** 2)
            - exy1 * ezz4 * w / (exx1 * n * s + exx1 * s ** 2)
            + ezz1 * n / (n * s + s ** 2)
            - ezz4 * n / (n * s + s ** 2)
        )
        + bv23 * (
            e * exy3 * ezz2 * n / (exx3 * n * s ** 2 + exx3 * s ** 3)
            - e * exy3 * ezz2 / (exx3 * s ** 2)
            + exy2 * ezz3 * n * w / (exx2 * n * s ** 2 + exx2 * s ** 3)
            + ezz2 * n / (n * s + s ** 2)
            - ezz2 / (2 * s)
            - ezz3 * n / (n * s + s ** 2)
            + ezz3 / (2 * s)
            + ezz2 * ezz3 / (2 * exx3 * s)
            - exy2 * ezz3 * w / (exx2 * s ** 2)
            - ezz2 * ezz3 / (2 * exx2 * s)
        )
    )
    bxe = (
        bh34 * (
            eyx3 * ezz4 / (2 * e * eyy3)
            - eyx4 * ezz3 / (2 * e * eyy4)
            + ezz3 * ezz4 * n / (e ** 2 * eyy4)
            + ezz3 * ezz4 * s / (e ** 2 * eyy3)
        )
        + bv14 * (ezz1 / (2 * n) - ezz1 * ezz4 / (2 * exx4 * n))
        + bv23 * (-ezz2 / (2 * s) + ezz2 * ezz3 / (2 * exx3 * s))
    )
    bxw = (
        bh12 * (
            -eyx1 * ezz2 / (2 * eyy1 * w)
            + eyx2 * ezz1 / (2 * eyy2 * w)
            - ezz1 * ezz2 * s / (eyy2 * w ** 2)
            - ezz1 * ezz2 * n / (eyy1 * w ** 2)
        )
        + bv14 * (-ezz4 / (2 * n) + ezz1 * ezz4 / (2 * exx1 * n))
        + bv23 * (ezz3 / (2 * s) - ezz2 * ezz3 / (2 * exx2 * s))
    )

    bxne = bh34 * eyx4 * ezz3 / (2 * e * eyy4) + bv14 * (-ezz1 / (2 * n) + ezz1 * ezz4 / (2 * exx4 * n))
    bxse = -bh34 * eyx3 * ezz4 / (2 * e * eyy3) + bv23 * (ezz2 / (2 * s) - ezz2 * ezz3 / (2 * exx3 * s))
    bxnw = bh12 * eyx1 * ezz2 / (2 * eyy1 * w) + bv14 * (ezz4 / (2 * n) - ezz1 * ezz4 / (2 * exx1 * n))
    bxsw = -bh12 * eyx2 * ezz1 / (2 * eyy2 * w) + bv23 * (-ezz3 / (2 * s) + ezz2 * ezz3 / (2 * exx2 * s))

    bxp = (
        bh12 * (
            eyx1 * ezz2 / (2 * eyy1 * w)
            - eyx2 * ezz1 / (2 * eyy2 * w)
            + ezz1 / s
            + ezz2 / n
            + ezz1 * ezz2 * s / (eyy2 * w ** 2)
            + ezz1 * ezz2 * n / (eyy1 * w ** 2)
        )
        + bh34 * (
            -ezz3 / n
            - ezz4 / s
            - eyx3 * ezz4 / (2 * e * eyy3)
            + eyx4 * ezz3 / (2 * e * eyy4)
            - ezz3 * ezz4 * n / (e ** 2 * eyy4)
            - ezz3 * ezz4 * s / (e ** 2 * eyy3)
        )
        + bv14 * (
            e * exy4 * ezz1 / (exx4 * n * s)
            - ezz1 / s
            + ezz1 / (2 * n)
            + ezz4 / s
            - ezz4 / (2 * n)
            + ezz1 * ezz4 / (2 * exx4 * n)
            + exy1 * ezz4 * w / (exx1 * n * s)
            - ezz1 * ezz4 / (2 * exx1 * n)
        )
        + bv23 * (
            e * exy3 * ezz2 / (exx3 * n * s)
            - ezz2 / (2 * s)
            + ezz2 / n
            + ezz3 / (2 * s)
            - ezz3 / n
            - ezz2 * ezz3 / (2 * exx3 * s)
            + exy2 * ezz3 * w / (exx2 * n * s)
            + ezz2 * ezz3 / (2 * exx2 * s)
        )
    )
    bxp += k0 ** 2 * (
        bh12 * (-ezz1 * ezz2 * n / 2 - ezz1 * ezz2 * s / 2)
        + bh34 * (ezz3 * ezz4 * n / 2 + ezz3 * ezz4 * s / 2)
        + bv14 * (-e * exy4 * ezz1 * ezz4 / (2 * exx4) - exy1 * ezz1 * ezz4 * w / (2 * exx1))
        + bv23 * (-e * exy3 * ezz2 * ezz3 / (2 * exx3) - exy2 * ezz2 * ezz3 * w / (2 * exx2))
    )

    byn = (
        bh12 * (ezz2 / (2 * w) - ezz1 * ezz2 / (2 * eyy1 * w))
        + bh34 * (ezz3 / (2 * e) - ezz3 * ezz4 / (2 * e * eyy4))
        + bv14 * (
            e * ezz1 * ezz4 / (exx4 * n ** 2)
            - exy4 * ezz1 / (2 * exx4 * n)
            + exy1 * ezz4 / (2 * exx1 * n)
            + ezz1 * ezz4 * w / (exx1 * n ** 2)
        )
    )
    bys = (
        bh12 * (-ezz1 / (2 * w) + ezz1 * ezz2 / (2 * eyy2 * w))
        + bh34 * (-ezz4 / (2 * e) + ezz3 * ezz4 / (2 * e * eyy3))
        + bv23 * (
            e * ezz2 * ezz3 / (exx3 * s ** 2)
            + exy3 * ezz2 / (2 * exx3 * s)
            - exy2 * ezz3 / (2 * exx2 * s)
            + ezz2 * ezz3 * w / (exx2 * s ** 2)
        )
    )
    bye = (
        bh12 * (
            eyx1 * ezz2 * n / (e ** 2 * eyy1 + e * eyy1 * w)
            + eyx2 * ezz1 * s / (e ** 2 * eyy2 + e * eyy2 * w)
            - ezz1 * w / (e ** 2 + e * w)
            + ezz2 * w / (e ** 2 + e * w)
        )
        + bh34 * (
            eyx3 * ezz4 * s * w / (e ** 3 * eyy3 + e ** 2 * eyy3 * w)
            + eyx4 * ezz3 * n * w / (e ** 3 * eyy4 + e ** 2 * eyy4 * w)
            - ezz3 * w / (e ** 2 + e * w)
            + ezz4 * w / (e ** 2 + e * w)
            + ezz3 / (2 * e)
            - ezz4 / (2 * e)
            - ezz3 * ezz4 / (2 * e * eyy4)
            + ezz3 * ezz4 / (2 * e * eyy3)
            - eyx3 * ezz4 * s / (e ** 2 * eyy3)
            - eyx4 * ezz3 * n / (e ** 2 * eyy4)
        )
        + bv14 * (-exy4 * ezz1 / (2 * exx4 * n) + ezz1 / e)
        + bv23 * (exy3 * ezz2 / (2 * exx3 * s) + ezz2 / e)
    )
    byw = (
        bh12 * (
            -e * eyx1 * ezz2 * n / (e * eyy1 * w ** 2 + eyy1 * w ** 3)
            - e * eyx2 * ezz1 * s / (e * eyy2 * w ** 2 + eyy2 * w ** 3)
            + e * ezz1 / (e * w + w ** 2)
            - e * ezz2 / (e * w + w ** 2)
            + eyx1 * ezz2 * n / (eyy1 * w ** 2)
            + eyx2 * ezz1 * s / (eyy2 * w ** 2)
            - ezz1 / (2 * w)
            + ezz2 / (2 * w)
            + ezz1 * ezz2 / (2 * eyy2 * w)
            - ezz1 * ezz2 / (2 * eyy1 * w)
        )
        + bh34 * (
            e * ezz3 / (e * w + w ** 2)
            - e * ezz4 / (e * w + w ** 2)
            - eyx3 * ezz4 * s / (e * eyy3 * w + eyy3 * w ** 2)
            - eyx4 * ezz3 * n / (e * eyy4 * w + eyy4 * w ** 2)
        )
        + bv14 * (ezz4 / w + exy1 * ezz4 / (2 * exx1 * n))
        + bv23 * (ezz3 / w - exy2 * ezz3 / (2 * exx2 * s))
    )

    byne = bh34 * (-ezz3 / (2 * e) + ezz3 * ezz4 / (2 * e * eyy4)) + bv14 * exy4 * ezz1 / (2 * exx4 * n)
    byse = bh34 * (ezz4 / (2 * e) - ezz3 * ezz4 / (2 * e * eyy3)) - bv23 * exy3 * ezz2 / (2 * exx3 * s)
    bynw = bh12 * (-ezz2 / (2 * w) + ezz1 * ezz2 / (2 * eyy1 * w)) - bv14 * exy1 * ezz4 / (2 * exx1 * n)
    bysw = bh12 * (ezz1 / (2 * w) - ezz1 * ezz2 / (2 * eyy2 * w)) + bv23 * exy2 * ezz3 / (2 * exx2 * s)

    byp = (
        bh12 * (
            -ezz1 / (2 * w)
            + ezz2 / (2 * w)
            - ezz1 * ezz2 / (2 * eyy2 * w)
            + ezz1 * ezz2 / (2 * eyy1 * w)
            - eyx1 * ezz2 * n / (e * eyy1 * w)
            - eyx2 * ezz1 * s / (e * eyy2 * w)
            + ezz1 / e
            - ezz2 / e
        )
        + bh34 * (
            -ezz3 / w
            + ezz4 / w
            + eyx3 * ezz4 * s / (e * eyy3 * w)
            + eyx4 * ezz3 * n / (e * eyy4 * w)
            + ezz3 / (2 * e)
            - ezz4 / (2 * e)
            + ezz3 * ezz4 / (2 * e * eyy4)
            - ezz3 * ezz4 / (2 * e * eyy3)
        )
        + bv14 * (
            -e * ezz1 * ezz4 / (exx4 * n ** 2)
            - ezz4 / w
            + exy4 * ezz1 / (2 * exx4 * n)
            - exy1 * ezz4 / (2 * exx1 * n)
            - ezz1 * ezz4 * w / (exx1 * n ** 2)
            - ezz1 / e
        )
        + bv23 * (
            -e * ezz2 * ezz3 / (exx3 * s ** 2)
            - ezz3 / w
            - exy3 * ezz2 / (2 * exx3 * s)
            + exy2 * ezz3 / (2 * exx2 * s)
            - ezz2 * ezz3 * w / (exx2 * s ** 2)
            - ezz2 / e
        )
    )
    byp += k0 ** 2 * (
        bh12 * (eyx1 * ezz1 * ezz2 * n / (2 * eyy1) + eyx2 * ezz1 * ezz2 * s / (2 * eyy2))
        + bh34 * (-eyx3 * ezz3 * ezz4 * s / (2 * eyy3) - eyx4 * ezz3 * ezz4 * n / (2 * eyy4))
        + bv14 * (e * ezz1 * ezz4 / 2 + ezz1 * ezz4 * w / 2)
        + bv23 * (e * ezz2 * ezz3 / 2 + ezz2 * ezz3 * w / 2)
    )

    jj = np.arange(nx * ny).reshape((ny, nx), order="C")

    # Hard Dirichlet boundary (boundary='0000'): Hx = Hy = 0 immediately
    # outside the domain -- no boundary-condition adjustment terms needed
    # (the reference algorithm's A/S/E/M boundary options are skipped).

    js = jj[:-1, :].ravel(order="C")
    jn = jj[1:, :].ravel(order="C")
    jw = jj[:, :-1].ravel(order="C")
    je = jj[:, 1:].ravel(order="C")

    jsw = jj[:-1, :-1].ravel(order="C")
    jse = jj[:-1, 1:].ravel(order="C")
    jnw = jj[1:, :-1].ravel(order="C")
    jne = jj[1:, 1:].ravel(order="C")

    jall = jj.ravel(order="C")

    rows = np.concatenate([jall, jw, je, js, jn, jne, jse, jsw, jnw])
    cols = np.concatenate([jall, je, jw, jn, js, jsw, jnw, jne, jse])

    axx = np.concatenate([axxp[jall], axxe[jw], axxw[je], axxn[js], axxs[jn], axxsw[jne], axxnw[jse], axxne[jsw], axxse[jnw]])
    axy = np.concatenate([axyp[jall], axye[jw], axyw[je], axyn[js], axys[jn], axysw[jne], axynw[jse], axyne[jsw], axyse[jnw]])
    ayx = np.concatenate([ayxp[jall], ayxe[jw], ayxw[je], ayxn[js], ayxs[jn], ayxsw[jne], ayxnw[jse], ayxne[jsw], ayxse[jnw]])
    ayy = np.concatenate([ayyp[jall], ayye[jw], ayyw[je], ayyn[js], ayys[jn], ayysw[jne], ayynw[jse], ayyne[jsw], ayyse[jnw]])

    Axx = sp.coo_matrix((axx, (rows, cols)), shape=(N, N))
    Axy = sp.coo_matrix((axy, (rows, cols)), shape=(N, N))
    Ayx = sp.coo_matrix((ayx, (rows, cols)), shape=(N, N))
    Ayy = sp.coo_matrix((ayy, (rows, cols)), shape=(N, N))

    bx = np.concatenate([bxp[jall], bxe[jw], bxw[je], bxn[js], bxs[jn], bxsw[jne], bxnw[jse], bxne[jsw], bxse[jnw]])
    by = np.concatenate([byp[jall], bye[jw], byw[je], byn[js], bys[jn], bysw[jne], bynw[jse], byne[jsw], byse[jnw]])

    Bx = sp.coo_matrix((bx, (rows, cols)), shape=(N, N))
    By = sp.coo_matrix((by, (rows, cols)), shape=(N, N))

    A = sp.bmat([[Axx, Axy], [Ayx, Ayy]], format="csr")
    A.eliminate_zeros()

    B = sp.bmat([[Bx, By]], format="csr")
    B.eliminate_zeros()

    return A, B, nx, ny, N


def _te_confinement(ex, ey, n_grid, n_core):
    intensity = np.abs(ex) ** 2 + np.abs(ey) ** 2
    core_mask = n_grid == n_core
    total = intensity.sum()
    return intensity[core_mask].sum() / total


def _poynting_confinement(ex, ey, hx, hy, n_grid, n_core):
    """Confinement from the z-component of the time-averaged Poynting
    vector, Sz = Re(Ex*conj(Hy) - Ey*conj(Hx)), rather than raw |E|^2 --
    Lumerical's own confinement convention is often flux-based, and the two
    can diverge meaningfully in high-index-contrast waveguides."""
    sz = np.real(ex * np.conj(hy) - ey * np.conj(hx))
    core_mask = n_grid == n_core
    total = sz.sum()
    return sz[core_mask].sum() / total


def solve_mode_vectorial(n_core, n_clad, width_um, height_um, wavelength_um,
                          domain_scale=10.0, resolution_um=0.02, n_eff_guess=None):
    """Solve for the fundamental quasi-TE guided mode via the full-vectorial
    (coupled Hx/Hy) finite-difference eigenvalue matrix -- see the module
    comment above this section for the algorithm reference. Unlike
    solve_mode_matrix, this captures the vectorial corner/polarization
    coupling that a scalar solver can't, at a substantially higher compute
    cost (2x the unknowns, non-symmetric shift-invert solve).

    Returns a dict with n_eff, confinement (from |Ex|^2 + |Ey|^2), the
    cell-centered Ex/Ey fields, and the grid axes.
    """
    if n_eff_guess is None:
        # Shift-invert converges to whichever eigenvalue is numerically
        # closest to this guess, not necessarily the fundamental mode -- a
        # midpoint guess was found (via the wide-slab-limit validation
        # test) to converge to a spurious higher-order mode instead. The
        # fundamental is always the single highest-n_eff bound mode, so
        # biasing the guess close to n_core reliably lands on it.
        n_eff_guess = n_clad + 0.9 * (n_core - n_clad)

    domain_width_um = width_um * domain_scale
    domain_height_um = height_um * domain_scale

    xs, ys, n_grid = _build_index_grid_cells(
        n_core, n_clad, width_um, height_um,
        domain_width_um, domain_height_um, resolution_um, resolution_um,
    )

    k0 = 2 * math.pi / wavelength_um
    eps_grid = n_grid ** 2

    A, B, nx, ny, N = _assemble_vector_matrices(
        eps_grid, eps_grid, eps_grid, resolution_um, resolution_um, k0,
    )

    shift = (k0 * n_eff_guess) ** 2
    nmodes = 2
    vals, vecs = spla.eigs(A, k=nmodes, sigma=shift, which="LM")

    if not np.allclose(vals.imag, 0, atol=1e-6 * np.abs(vals).max()):
        raise ValueError(f"Unexpected complex eigenvalues: {vals}")
    vals = vals.real
    neff = np.sqrt(vals.astype(complex)) / k0
    order = np.argsort(-neff.real)
    neff = neff[order].real
    vecs = vecs[:, order]

    candidates = []
    for m in range(nmodes):
        beta = k0 * neff[m]
        v = vecs[:, m]
        hzj = (B @ v) / beta
        hx_2d = v[:N].reshape((ny, nx), order="C")
        hy_2d = v[N:].reshape((ny, nx), order="C")
        hzj_2d = hzj.reshape((ny, nx), order="C")

        hx1, hx2, hx3, hx4 = hx_2d[1:, :-1], hx_2d[:-1, :-1], hx_2d[:-1, 1:], hx_2d[1:, 1:]
        hy1, hy2, hy3, hy4 = hy_2d[1:, :-1], hy_2d[:-1, :-1], hy_2d[:-1, 1:], hy_2d[1:, 1:]
        hzj1, hzj2, hzj3, hzj4 = hzj_2d[1:, :-1], hzj_2d[:-1, :-1], hzj_2d[:-1, 1:], hzj_2d[1:, 1:]

        dHx_dy = (hx1 + hx4 - hx2 - hx3) / (2 * resolution_um)
        dHy_dx = (hy3 + hy4 - hy1 - hy2) / (2 * resolution_um)
        dHzj_dx = (hzj3 + hzj4 - hzj1 - hzj2) / (2 * resolution_um)
        dHzj_dy = (hzj1 + hzj4 - hzj2 - hzj3) / (2 * resolution_um)
        Hy_cell = (hy1 + hy2 + hy3 + hy4) / 4
        Hx_cell = (hx1 + hx2 + hx3 + hx4) / 4

        Dx = -dHzj_dy / k0 + (beta / k0) * Hy_cell
        Dy = +dHzj_dx / k0 - (beta / k0) * Hx_cell

        # isotropic: E = D / eps
        ex = Dx / eps_grid
        ey = Dy / eps_grid

        te_fraction = np.abs(ex) ** 2
        candidates.append((te_fraction.sum(), neff[m].real, ex, ey, Hx_cell, Hy_cell))

    # pick the quasi-TE mode: larger Ex energy (dominant Ex <-> dominant Hy
    # by the Dx curl relation above), mirroring the "TE fraction > 0.5"
    # mode-selection convention used in lumerical_validation.py
    candidates.sort(key=lambda c: c[0], reverse=True)
    _, n_eff, ex, ey, hx_cell, hy_cell = candidates[0]

    if not (n_clad < n_eff < n_core):
        raise ValueError(
            f"Solved n_eff={n_eff:.4f} outside guided range ({n_clad}, {n_core}); "
            "domain likely too small or grid too coarse for this geometry."
        )

    confinement_e2 = _te_confinement(ex, ey, n_grid, n_core)
    confinement_poynting = _poynting_confinement(ex, ey, hx_cell, hy_cell, n_grid, n_core)

    return {
        "n_eff": n_eff,
        "confinement": confinement_e2,
        "confinement_e2": confinement_e2,
        "confinement_poynting": confinement_poynting,
        "ex": ex,
        "ey": ey,
        "hx": hx_cell,
        "hy": hy_cell,
        "x_um": xs,
        "y_um": ys,
        "n_grid": n_grid,
    }


def solve_mode_vectorial_spectral(n_core, n_clad, width_um, height_um,
                                   freq_hz=None, power_spectrum=None,
                                   n_samples=18, threshold_db=20.0,
                                   domain_scale=10.0, resolution_um=0.02,
                                   n_eff_guess=None):
    """Full-vectorial counterpart to solve_mode_matrix_spectral -- runs
    solve_mode_vectorial at samples drawn from the Gaussian source spectrum
    and returns the power-spectrum-weighted n_eff/confinement."""
    if freq_hz is None or power_spectrum is None:
        freq_hz, power_spectrum = load_source_spectrum()

    f_sel, p_sel = select_spectral_samples(freq_hz, power_spectrum, n_samples, threshold_db)

    n_effs = np.empty(len(f_sel))
    confinements = np.empty(len(f_sel))
    for i, f in enumerate(f_sel):
        wavelength_um = (C_CONST / f) * 1e6
        result = solve_mode_vectorial(
            n_core, n_clad, width_um, height_um, wavelength_um,
            domain_scale=domain_scale, resolution_um=resolution_um,
            n_eff_guess=n_eff_guess,
        )
        n_effs[i] = result["n_eff"]
        confinements[i] = result["confinement"]

    weight_norm = np.trapezoid(p_sel, f_sel)
    n_eff_weighted = np.trapezoid(n_effs * p_sel, f_sel) / weight_norm
    confinement_weighted = np.trapezoid(confinements * p_sel, f_sel) / weight_norm

    return {
        "n_eff_weighted": n_eff_weighted,
        "confinement_weighted": confinement_weighted,
        "freq_hz": f_sel,
        "n_eff": n_effs,
        "confinement": confinements,
        "power_spectrum": p_sel,
    }
