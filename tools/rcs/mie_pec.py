#!/usr/bin/env python3
"""Mie series for a PEC sphere, following Ruck et al., Radar Cross Section Handbook.

Convention: exp(j*omega*t) time dependence, outgoing Green's function exp(-j*k*r),
incident plane wave propagating along -z (so backscatter is theta=0 and forward
scattering is theta=pi), with the electric field along x.
Returns (es_theta, es_phi) whose squared magnitudes are the RCS in m^2.
"""
import numpy as np
from scipy.special import spherical_jn, spherical_yn

C0 = 299792458.0


def mie_coefficients(x, nmax=None):
    """Return the Mie coefficients A_n and B_n for a PEC sphere at size parameter x=ka."""
    if nmax is None:
        nmax = int(np.ceil(x + 4.0 * x ** (1.0 / 3.0) + 10.0))
    n = np.arange(1, nmax + 1)
    psi1 = x * spherical_jn(n, x)
    xi1 = x * (spherical_jn(n, x) - 1j * spherical_yn(n, x))
    psi2 = x * spherical_jn(n - 1, x)
    xi2 = x * (spherical_jn(n - 1, x) - 1j * spherical_yn(n - 1, x))
    A = -((-1j) ** n) * psi1 / xi1 * (2 * n + 1) / (n * (n + 1))
    B = ((-1j) ** (n + 1)) * (x * psi2 - n * psi1) / (x * xi2 - n * xi1) \
        * (2 * n + 1) / (n * (n + 1))
    return A, B


def monostatic_normalized(ka):
    """Monostatic RCS of a PEC sphere normalized by the projected area, sigma/(pi a^2).

    `ka` is the size parameter = circumference / wavelength. Vectorized over `ka`.
    Uses the theta = 0 (backscatter) special case of the Mie series; the incident
    wave propagates along -z in this convention.
    """
    ka = np.atleast_1d(np.asarray(ka, dtype=float))
    out = np.empty_like(ka)
    for i, x in enumerate(ka):
        if x <= 0.0:
            out[i] = 0.0
            continue
        A, B = mie_coefficients(x)
        n = np.arange(1, A.size + 1)
        val = ((-1j) ** (n - 1)) * (n * (n + 1) / 2.0) * (A + 1j * B)
        S1 = np.sum(val)
        # |es|^2 = 4*pi/k^2 |S1|^2 and k = ka/a, so sigma/(pi a^2) = 4 |S1|^2/(ka)^2.
        out[i] = 4.0 * abs(S1) ** 2 / x ** 2
    return out


def monostatic_normalized_db(ka):
    return 10.0 * np.log10(np.maximum(monostatic_normalized(ka), 1.0e-300))


def mie_pec(radius, frequency, theta, phi, nmax=None):
    c = C0
    k = 2.0 * np.pi * frequency / c
    x = k * radius
    if nmax is None:
        nmax = int(np.ceil(x + 4.0 * x ** (1.0 / 3.0) + 10.0))

    n = np.arange(1, nmax + 1)
    psi1 = x * spherical_jn(n, x)
    xi1 = x * (spherical_jn(n, x) - 1j * spherical_yn(n, x))
    psi2 = x * spherical_jn(n - 1, x)
    xi2 = x * (spherical_jn(n - 1, x) - 1j * spherical_yn(n - 1, x))

    A = -((-1j) ** n) * psi1 / xi1 * (2 * n + 1) / (n * (n + 1))
    B = ((-1j) ** (n + 1)) * (x * psi2 - n * psi1) / (x * xi2 - n * xi1) \
        * (2 * n + 1) / (n * (n + 1))

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


if __name__ == "__main__":
    a = 0.159154943
    f = 0.3e9
    # Monostatic check (theta=pi) and forward (theta=0).
    for th in [0.0, np.pi / 2, np.pi]:
        et, ep = mie_pec(a, f, [th], [0.0])
        print(f"theta={np.degrees(th):6.1f}  sigma_theta={abs(et[0])**2:.6f} m^2"
              f"  sigma_phi={abs(ep[0])**2:.6f} m^2"
              f"  sigma/pi a^2={abs(et[0])**2/(np.pi*a*a):.4f}")
