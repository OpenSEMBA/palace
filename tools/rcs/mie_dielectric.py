#!/usr/bin/env python3
"""Mie series for a homogeneous dielectric (or magnetic) sphere.

Follows Bohren & Huffman, *Absorption and Scattering of Light by Small
Particles*, using the same conventions as ``mie_pec.py``: exp(j*omega*t) time
dependence, outgoing Green's function exp(-j*k*r), incident plane wave
propagating along -z (backscatter at theta=0), electric field along x. Returns
``(es_theta, es_phi)`` whose squared magnitudes are the RCS in m^2.

The relative refractive index is ``m = sqrt(epsilon_r * mu_r)`` (principal
branch): for a passive medium with ``eps = eps' - j*eps''`` this gives
``Im(m) < 0`` and an outgoing wave that decays with distance.
"""
import numpy as np
from scipy.special import spherical_jn, spherical_yn

C0 = 299792458.0


def mie_coefficients_dielectric(x, m, nmax=None):
    """Riccati-Bessel Mie coefficients A_n, B_n for a dielectric sphere.

    Same phase/normalization convention as ``mie_pec.mie_coefficients`` so the
    angular sums below are shared: A_n = -(-j)^n a_n (2n+1)/(n(n+1)),
    B_n = (-j)^(n+1) b_n (2n+1)/(n(n+1)), with a_n, b_n the Bohren-Huffman
    coefficients (Eq. 4.52). In the PEC limit |m| -> infinity they reduce to the
    PEC coefficients.
    """
    if nmax is None:
        nmax = int(np.ceil(x + 4.0 * x ** (1.0 / 3.0) + 10.0))
    n = np.arange(1, nmax + 1)
    mx = m * x
    psi1 = x * spherical_jn(n, x)
    xi1 = x * (spherical_jn(n, x) - 1j * spherical_yn(n, x))
    psi1m = mx * spherical_jn(n, mx)
    dpsi1 = x * spherical_jn(n - 1, x) - n * psi1 / x
    dxi1 = x * (spherical_jn(n - 1, x) - 1j * spherical_yn(n - 1, x)) - n * xi1 / x
    dpsi1m = mx * spherical_jn(n - 1, mx) - n * psi1m / mx
    a = (m * psi1m * dpsi1 - psi1 * dpsi1m) / (m * psi1m * dxi1 - xi1 * dpsi1m)
    b = (psi1m * dpsi1 - m * psi1 * dpsi1m) / (psi1m * dxi1 - m * xi1 * dpsi1m)
    # The angular-sum convention of mie_pec.py pairs the PEC coefficient psi/xi with the
    # pi_n term (A) and psi'/xi' with the tau_n term (B). In the |m| -> infinity limit the
    # dielectric coefficients become a_n -> psi'/xi' and b_n -> psi/xi, so a_n takes the
    # place of B and b_n the place of A (verified against mie_pec at theta = 0, 90, 180).
    A = -((-1j) ** n) * b * (2 * n + 1) / (n * (n + 1))
    B = ((-1j) ** (n + 1)) * a * (2 * n + 1) / (n * (n + 1))
    return A, B


def _angular_sum(A, B, theta, phi, k, nmax):
    """Scattering amplitudes S1, S2 summed over the Mie coefficients, matching
    the angular convention of mie_pec.mie_pec."""
    n = np.arange(1, nmax + 1)
    theta = np.atleast_1d(np.asarray(theta, dtype=float))
    phi = np.atleast_1d(np.asarray(phi, dtype=float))
    es_theta = np.zeros(theta.shape, dtype=complex)
    es_phi = np.zeros(theta.shape, dtype=complex)
    for i, th in enumerate(theta):
        ct, st = np.cos(th), np.sin(th)
        plm = np.zeros(nmax + 1)
        plm[0] = -st
        if nmax >= 1:
            plm[1] = -3.0 * st * ct
        S1 = 0.0 + 0.0j
        S2 = 0.0 + 0.0j
        p = plm[0]
        for m in range(1, nmax + 1):
            if abs(ct) < 0.999999:
                if m == 1:
                    dp = ct * plm[0] / np.sqrt(1.0 - ct * ct)
                else:
                    dp = (m * ct * plm[m - 1] - (m + 1) * plm[m - 2]) / np.sqrt(
                        1.0 - ct * ct)
            if abs(st) > 1.0e-6:
                term1 = A[m - 1] * p / st
                term2 = B[m - 1] * p / st
            if ct > 0.999999:
                val = ((-1j) ** (m - 1)) * (m * (m + 1) / 2.0) * (A[m - 1] +
                                                                   1j * B[m - 1])
                S1 += val
                S2 += val
            elif ct < -0.999999:
                val = ((1j) ** (m - 1)) * (m * (m + 1) / 2.0) * (A[m - 1] -
                                                                  1j * B[m - 1])
                S1 += val
                S2 -= val
            else:
                S1 += ((-1j) ** (m + 1)) * (term1 + 1j * B[m - 1] * dp)
                S2 += ((-1j) ** (m + 1)) * (A[m - 1] * dp + 1j * term2)
            if m > 1:
                plm[m] = (2.0 * m + 1) * ct * plm[m - 1] / m - (m + 1) * plm[m - 2] / m
            p = plm[m]
        es_theta[i] = S1 * np.cos(phi[0])
        es_phi[i] = -S2 * np.sin(phi[0])
    es_theta *= np.sqrt(4.0 * np.pi) / k
    es_phi *= np.sqrt(4.0 * np.pi) / k
    return es_theta, es_phi


def mie_dielectric(radius, frequency, theta, phi, epsilon_r=1.0, mu_r=1.0, nmax=None):
    """Scattered far-field amplitudes (es_theta, es_phi) of a dielectric sphere.

    ``epsilon_r`` may be complex (passive media use eps = eps' - j*eps''), and
    ``mu_r`` likewise. |es|^2 is the RCS in m^2 for the incident field amplitude
    of 1 V/m.
    """
    k = 2.0 * np.pi * frequency / C0
    x = k * radius
    m = np.sqrt(epsilon_r * mu_r)
    if nmax is None:
        nmax = int(np.ceil(x + 4.0 * x ** (1.0 / 3.0) + 10.0))
    A, B = mie_coefficients_dielectric(x, m, nmax)
    return _angular_sum(A, B, theta, phi, k, nmax)


def monostatic_normalized_dielectric(ka, epsilon_r, mu_r=1.0):
    """Monostatic RCS normalized by the projected area, sigma/(pi a^2).

    Vectorized over ``ka``; theta = 0 (backscatter) special case, incidence along
    -z in this convention.
    """
    ka = np.atleast_1d(np.asarray(ka, dtype=float))
    out = np.empty_like(ka)
    m = np.sqrt(epsilon_r * mu_r)
    for i, x in enumerate(ka):
        if x <= 0.0:
            out[i] = 0.0
            continue
        A, B = mie_coefficients_dielectric(x, m)
        n = np.arange(1, A.size + 1)
        val = ((-1j) ** (n - 1)) * (n * (n + 1) / 2.0) * (A + 1j * B)
        S1 = np.sum(val)
        out[i] = 4.0 * abs(S1) ** 2 / x ** 2
    return out


if __name__ == "__main__":
    a = 0.159154943
    f = 0.3e9
    # PEC limit check against mie_pec.
    from mie_pec import mie_pec
    for th in [0.0, np.pi / 2, np.pi]:
        et_d, ep_d = mie_dielectric(a, f, [th], [0.0], epsilon_r=1e12 + 0j)
        et_p, ep_p = mie_pec(a, f, [th], [0.0])
        print(f"theta={np.degrees(th):6.1f}  dielectric(m->inf) {abs(et_d[0])**2:.6f}"
              f"  pec {abs(et_p[0])**2:.6f} m^2")
    # Rayleigh limit: sigma_mono/(pi a^2) -> 4 (ka)^4 |(eps-1)/(eps+2)|^2.
    eps = 4.0 + 0j
    ka = 0.01
    val = monostatic_normalized_dielectric(ka, eps)[0]
    rayleigh = 4.0 * ka**4 * abs((eps - 1) / (eps + 2)) ** 2
    print(f"Rayleigh check ka={ka}: mie {val:.6e}  analytic {rayleigh:.6e}")
