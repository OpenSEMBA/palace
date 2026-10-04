# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Mie series for the radar cross section (RCS) of a perfectly electrically conducting (PEC)
or dielectric sphere illuminated by a linearly polarized plane wave.

Conventions (Bohren & Huffman, "Absorption and Scattering of Light by Small Particles",
1983, chapter 4):

  - The sphere has radius a, the free-space wavenumber is k, and x = k a is the size
    parameter.
  - The scattering angle ψ is measured from the forward direction (ψ = 0 is forward
    scattering, ψ = π is backscattering).
  - The E-plane is the plane containing the incident propagation direction and the incident
    electric field, the H-plane is the one containing the propagation direction and the
    incident magnetic field.

The bistatic RCS is σ(ψ) = 4π |S(ψ)|² / k², with S = S₂ in the E-plane and S = S₁ in the
H-plane, where

  S₁(ψ) = Σₙ (2n+1)/(n(n+1)) [aₙ πₙ(cos ψ) + bₙ τₙ(cos ψ)],
  S₂(ψ) = Σₙ (2n+1)/(n(n+1)) [aₙ τₙ(cos ψ) + bₙ πₙ(cos ψ)],

and, for a PEC sphere, aₙ = [x jₙ(x)]' / [x hₙ(x)]' and bₙ = jₙ(x) / hₙ(x) with hₙ = hₙ⁽¹⁾.
For a nonmagnetic dielectric sphere with relative permittivity εᵣ (refractive index
m = √εᵣ), with the Riccati-Bessel functions ψₙ(z) = z jₙ(z) and ξₙ(z) = z hₙ(z),

  aₙ = [m ψₙ(mx) ψₙ'(x) - ψₙ(x) ψₙ'(mx)] / [m ψₙ(mx) ξₙ'(x) - ξₙ(x) ψₙ'(mx)],
  bₙ = [ψₙ(mx) ψₙ'(x) - m ψₙ(x) ψₙ'(mx)] / [ψₙ(mx) ξₙ'(x) - m ξₙ(x) ψₙ'(mx)],

which reduce to the PEC coefficients as m → ∞. The RCS is independent of the time-harmonic
convention.
"""

import numpy as np
from scipy.special import spherical_jn, spherical_yn

C0 = 299792458.0  # Speed of light in vacuum, m/s


def num_terms(x):
    """Number of terms in the Mie series (Wiscombe's criterion, with a safety margin)."""
    return int(np.ceil(x + 4.05 * x ** (1.0 / 3.0) + 2.0)) + 5


def mie_coefficients(x, eps_r=None):
    """
    Return n = 1..N and the scattering coefficients aₙ, bₙ for size parameter x, for a PEC
    sphere (eps_r = None) or a dielectric sphere of relative permittivity eps_r.
    """
    m = None if eps_r is None else np.sqrt(complex(eps_r))
    n = np.arange(1, num_terms(x) + 1)
    jn = spherical_jn(n, x)
    yn = spherical_yn(n, x)
    djn = spherical_jn(n, x, derivative=True)
    dyn = spherical_yn(n, x, derivative=True)
    # Riccati-Bessel functions and derivatives: [x zₙ(x)]' = zₙ(x) + x zₙ'(x).
    psi, dpsi = x * jn, jn + x * djn
    xi, dxi = x * (jn + 1j * yn), (jn + 1j * yn) + x * (djn + 1j * dyn)
    if m is None:
        return n, dpsi / dxi, psi / xi
    mx = m * x
    jm = spherical_jn(n, mx)
    djm = spherical_jn(n, mx, derivative=True)
    psim, dpsim = mx * jm, jm + mx * djm
    a = (m * psim * dpsi - psi * dpsim) / (m * psim * dxi - xi * dpsim)
    b = (psim * dpsi - m * psi * dpsim) / (psim * dxi - m * xi * dpsim)
    return n, a, b


def angular_functions(nmax, mu):
    """Angular functions πₙ(μ) and τₙ(μ), n = 1..nmax, evaluated at μ = cos ψ."""
    mu = np.atleast_1d(mu)
    pi = np.zeros((nmax + 1, mu.size))
    tau = np.zeros((nmax + 1, mu.size))
    pi[1] = 1.0
    tau[1] = mu
    for n in range(2, nmax + 1):
        pi[n] = ((2 * n - 1) * mu * pi[n - 1] - n * pi[n - 2]) / (n - 1)
        tau[n] = n * mu * pi[n] - (n + 1) * pi[n - 1]
    return pi[1:], tau[1:]


def scattering_amplitudes(x, psi, eps_r=None):
    """Amplitudes S₁(ψ), S₂(ψ) for size parameter x and scattering angles ψ (radians)."""
    n, a, b = mie_coefficients(x, eps_r)
    pi, tau = angular_functions(n.size, np.cos(psi))
    c = ((2 * n + 1) / (n * (n + 1)))[:, None]
    S1 = np.sum(c * (a[:, None] * pi + b[:, None] * tau), axis=0)
    S2 = np.sum(c * (a[:, None] * tau + b[:, None] * pi), axis=0)
    return S1, S2


def bistatic_rcs(radius, freq_hz, psi, eps_r=None):
    """
    Bistatic RCS (m²) of a sphere in the E-plane and H-plane.

    Arguments:
      radius: Sphere radius (m).
      freq_hz: Frequency (Hz).
      psi: Scattering angles from the forward direction (radians).
      eps_r: Relative permittivity of a dielectric sphere, or None for a PEC sphere.

    Returns (sigma_E, sigma_H).
    """
    k = 2.0 * np.pi * freq_hz / C0
    S1, S2 = scattering_amplitudes(k * radius, psi, eps_r)
    return 4.0 * np.pi * np.abs(S2) ** 2 / k**2, 4.0 * np.pi * np.abs(S1) ** 2 / k**2


def monostatic_rcs(radius, freq_hz, eps_r=None):
    """
    Backscattering (monostatic) RCS (m²) of a sphere, for an array of frequencies. A PEC
    sphere is assumed if eps_r is None.
    """
    freq_hz = np.atleast_1d(freq_hz)
    sigma = np.empty(freq_hz.size)
    for i, f in enumerate(freq_hz):
        k = 2.0 * np.pi * f / C0
        n, a, b = mie_coefficients(k * radius, eps_r)
        s = np.sum((-1.0) ** n * (2 * n + 1) * (a - b))
        sigma[i] = np.pi / k**2 * np.abs(s) ** 2
    return sigma


if __name__ == "__main__":
    # Sanity checks: Rayleigh limit σ ≈ 9πa²(ka)⁴ and optical limit σ → πa².
    a = 0.5
    for ka in [0.05, 1.0, np.pi, 10.0, 100.0]:
        f = ka * C0 / (2 * np.pi * a)
        print(
            f"ka = {ka:8.3f}: sigma / (pi a^2) = "
            f"{monostatic_rcs(a, f)[0] / (np.pi * a**2):.6f}"
            + (f" (Rayleigh: {9 * ka**4:.6f})" if ka < 0.1 else "")
        )
