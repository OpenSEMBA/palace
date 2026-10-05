#!/usr/bin/env python3
"""Mie series for a PEC circular cylinder (2D), TE and TM polarizations.

Conventions: exp(+i*omega*t) time dependence, outgoing Hankel functions
H_n^(2)(k rho) (spatial factor exp(-i k rho)), incident wave propagating along
-x (phasor exp(+i k x) = exp(+i k rho cos(phi))), so the backscatter is at
phi = 0. TE means H along the cylinder axis z (Palace's 2D in-plane E
formulation); TM means E along z.

The 2D RCS is defined as sigma(phi) = lim 2*pi*rho |H_z^scat|^2/|H_z^inc|^2
(length units, m for SI radii).
"""
import numpy as np
from scipy.special import jv, jvp, hankel2, h2vp

C0 = 299792458.0


def _nmax(ka):
    return int(np.ceil(ka + 4.0 * ka ** (1.0 / 3.0) + 10.0))


def te_coefficients(ka, nmax=None):
    """Scattering coefficients c_n for TE (H_z) incidence on a PEC cylinder."""
    if nmax is None:
        nmax = _nmax(ka)
    n = np.arange(-nmax, nmax + 1)
    c = -jvp(n, ka, 1) / h2vp(n, ka, 1)
    return n, c


def tm_coefficients(ka, nmax=None):
    """Scattering coefficients c_n for TM (E_z) incidence on a PEC cylinder."""
    if nmax is None:
        nmax = _nmax(ka)
    n = np.arange(-nmax, nmax + 1)
    c = -jv(n, ka) / hankel2(n, ka)
    return n, c


def te_near(radius, frequency, rho, phi, nmax=None):
    """TE near field (H_z, E_rho, E_phi) at polar points (rho, phi).

    The incident H_z = exp(i k x) = exp(i k rho cos(phi)); the returned fields
    are the total fields (incident + scattered).
    """
    k = 2.0 * np.pi * frequency / C0
    ka = k * radius
    if nmax is None:
        nmax = _nmax(ka)
    n, c = te_coefficients(ka, nmax)
    rho = np.atleast_1d(np.asarray(rho, dtype=float))
    phi = np.atleast_1d(np.asarray(phi, dtype=float))
    RHO, PHI = np.meshgrid(rho, phi, indexing="ij")
    z = k * RHO
    phase = np.exp(1j * n[:, None, None] * PHI[None, :, :])
    J = jv(n[:, None, None], z[None, :, :])
    H = hankel2(n[:, None, None], z[None, :, :])
    Jp = jvp(n[:, None, None], z[None, :, :], 1)
    Hp = h2vp(n[:, None, None], z[None, :, :], 1)
    in_c = (1j ** n)[:, None, None]
    Hz = np.sum(in_c * (J + c[:, None, None] * H) * phase, axis=0)
    dHz_dphi = np.sum(in_c * (J + c[:, None, None] * H) * phase
                      * (1j * n)[:, None, None], axis=0)
    dHz_drho = k * np.sum(in_c * (Jp + c[:, None, None] * Hp) * phase, axis=0)
    # Internal units: c = 1, eps = mu = 1, so omega = k.
    Erho = dHz_dphi / (1j * k * RHO)
    Ephi = -dHz_drho / (1j * k)
    return Hz, Erho, Ephi


def te_far(radius, frequency, phi, nmax=None):
    """TE far-field RCS sigma(phi) (m) for incidence along -x (backscatter at
    phi = 0)."""
    k = 2.0 * np.pi * frequency / C0
    ka = k * radius
    if nmax is None:
        nmax = _nmax(ka)
    n, c = te_coefficients(ka, nmax)
    phi = np.atleast_1d(np.asarray(phi, dtype=float))
    f = np.sum(((-1.0) ** n)[:, None] * c[:, None] * np.exp(1j * n[:, None] * phi[None, :]),
               axis=0)
    return (4.0 / k) * np.abs(f) ** 2


def tm_far(radius, frequency, phi, nmax=None):
    """TM far-field RCS sigma(phi) (m) for incidence along -x."""
    k = 2.0 * np.pi * frequency / C0
    ka = k * radius
    if nmax is None:
        nmax = _nmax(ka)
    n, c = tm_coefficients(ka, nmax)
    phi = np.atleast_1d(np.asarray(phi, dtype=float))
    f = np.sum(((-1.0) ** n)[:, None] * c[:, None] * np.exp(1j * n[:, None] * phi[None, :]),
               axis=0)
    return (4.0 / k) * np.abs(f) ** 2


if __name__ == "__main__":
    a = 0.5
    for ka in [0.5, 1.0, 2.0, 5.0]:
        f = ka * C0 / (2.0 * np.pi * a)
        s0 = te_far(a, f, [0.0])[0]
        s180 = te_far(a, f, [np.pi])[0]
        # PEC boundary check: E_phi(a) = 0.
        Hz, Erho, Ephi = te_near(a, f, [a], [0.3, 1.1])
        print(f"ka={ka:4.1f}: TE mono(phi=0) {10*np.log10(s0):+7.2f} dB, "
              f"forward {10*np.log10(s180):+7.2f} dB, |E_phi(a)| max "
              f"{np.max(np.abs(Ephi)):.2e}")
