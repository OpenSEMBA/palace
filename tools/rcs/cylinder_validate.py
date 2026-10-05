#!/usr/bin/env python3
"""2D PEC cylinder RCS validation: Palace scattered-field near fields on three
concentric probe circles, transformed to the far field with the scalar 2D
Kirchhoff integral, compared with the TE cylinder Mie series.

Conventions: incident wave along -x (SF Direction = [-1, 0]), TE (in-plane E,
H_z out of plane), 2D RCS sigma(phi) = 2*pi*rho |H_z^scat|^2/|H_z^inc|^2 in m.
Internal units (c = 1, Z = 1) are used throughout, matching Palace's
nondimensional fields with L0 = 1 m.
"""
import csv
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/home/luis/palace/tools/rcs")
from mie_cylinder import te_far  # noqa: E402

C0 = 299792458.0
PALACE = "/home/luis/palace/build/bin/palace"
TOOLS = Path("/home/luis/palace/tools/rcs")
WORK = Path(os.environ.get("CYL_WORKDIR", "/tmp/opencode/pwtest/cyl"))
NPROC = int(os.environ.get("RCS_NPROC", "16"))
RUN = os.environ.get("RUN_PALACE", "1") == "1"
KA = [float(x) for x in os.environ.get(
    "CYL_KA", "0.25,0.5,0.75,1,1.5,2,2.5,3,4,5").split(",")]
A = 0.5
NPROBE = int(os.environ.get("CYL_NPROBE", "360"))
DRHO = 0.01  # circle spacing in wavelengths


def params(ka):
    f = ka * C0 / (2.0 * np.pi * A)
    lam = C0 / f
    rho0 = A + 0.5 * lam
    return {
        "f": f, "lam": lam, "R": A + 1.5 * lam,
        "hmin": min(A / 6.0, lam / 16.0), "hmax": lam / 8.0,
        "dmax": min(0.75 * lam, 2.5),
        "rhos": [rho0 - DRHO * lam, rho0, rho0 + DRHO * lam],
    }


def make_mesh(ka):
    d = WORK / f"ka_{ka:g}"
    d.mkdir(parents=True, exist_ok=True)
    msh = d / "cylinder.msh"
    if msh.exists():
        return msh
    prm = params(ka)
    env = dict(os.environ, GMSH_HIGH_ORDER_OPTIMIZE=os.environ.get(
        "GMSH_HIGH_ORDER_OPTIMIZE", "1"))
    subprocess.run([sys.executable, str(TOOLS / "cylinder_mesh.py"), "cylinder.msh",
                    str(A), str(prm["R"]), str(prm["hmin"]), str(prm["hmax"]),
                    str(prm["dmax"]), "2"], cwd=d, env=env, check=True)
    return msh


def make_config(ka):
    d = WORK / f"ka_{ka:g}"
    prm = params(ka)
    probes = []
    idx = 1
    for rho in prm["rhos"]:
        for j in range(NPROBE):
            ph = 2.0 * np.pi * j / NPROBE
            probes.append({"Index": idx, "Center": [rho * math.cos(ph),
                                                    rho * math.sin(ph), 0.0]})
            idx += 1
    cfg = {
        "Problem": {"Type": "Driven", "Verbose": 1, "Output": f"postpro/ka_{ka:g}"},
        "Model": {"Mesh": "cylinder.msh", "L0": 1.0},
        "Domains": {"Materials": [{"Attributes": [1]}],
                    "Postprocessing": {"Probe": probes}},
        "Boundaries": {
            "PEC": {"Attributes": [2]},
            "Absorbing": {"Attributes": [1], "Order": 1, "ScatteredField": True},
            "PlaneWave": [{"Index": 1, "Attributes": [1], "Excitation": 1,
                           "Direction": [-1.0, 0.0, 0.0],
                           "Polarization": [0.0, 1.0, 0.0], "Amplitude": 1.0}],
        },
        "Solver": {"Order": 2,
                   "Linear": {"Type": "Default", "KSPType": "GMRES", "Tol": 1.0e-6,
                              "MaxIts": 1000, "InitialGuess": True},
                   "Driven": {"Samples": [{"Type": "Point", "Freq": [prm["f"] / 1e9]}]}},
    }
    (d / f"ka_{ka:g}.json").write_text(json.dumps(cfg, indent=1))


def read_probes(path, nprobe):
    """Return a (3, nprobe, 2) array of complex (E_x, E_y) on the three circles."""
    with open(path) as f:
        rows = list(csv.reader(f))
    vals = np.array([float(x) for x in rows[1][1:]])
    comp = (vals[0::2] + 1j * vals[1::2]).reshape(3 * nprobe, 2)
    return comp.reshape(3, nprobe, 2)


def _polar(E, phi):
    Er = E[:, 0] * np.cos(phi) + E[:, 1] * np.sin(phi)
    Ep = -E[:, 0] * np.sin(phi) + E[:, 1] * np.cos(phi)
    return Er, Ep


def _d_dphi(f, phi):
    n = len(phi)
    m = np.fft.fftfreq(n, 1.0 / n)
    return np.fft.ifft(1j * m * np.fft.fft(f))


def kirchhoff_far(E, phi_c, rhos, k, phis):
    """Scalar 2D Kirchhoff far field from the in-plane E on three circles.

    E: (3, nprobe, 2) complex total E on the circles at rhos = [r-d, r, r+d].
    Returns sigma(phi) in m (2D RCS).
    """
    r1, r2, r3 = rhos
    d = r2 - r1
    E = E.copy()
    # Add the analytic incident field E_inc = yhat exp(+i k x) (the SF solution
    # is the scattered field).
    for i, rho in enumerate(rhos):
        x = rho * np.cos(phi_c)
        E[i, :, 1] += np.exp(1j * k * x)
    Er = np.zeros((3, len(phi_c)), dtype=complex)
    Ep = np.zeros((3, len(phi_c)), dtype=complex)
    for i in range(3):
        Er[i], Ep[i] = _polar(E[i], phi_c)
    # Radial derivatives (second-order three-point stencils).
    dEp_dr = np.zeros((3, len(phi_c)), dtype=complex)
    dEp_dr[1] = (Ep[2] - Ep[0]) / (2.0 * d)
    dEp_dr[0] = (-3.0 * Ep[0] + 4.0 * Ep[1] - Ep[2]) / (2.0 * d)
    dEp_dr[2] = (3.0 * Ep[2] - 4.0 * Ep[1] + Ep[0]) / (2.0 * d)
    dEr_dphi = np.zeros((3, len(phi_c)), dtype=complex)
    for i in range(3):
        dEr_dphi[i] = _d_dphi(Er[i], phi_c)
    # H_z = -(1/(i k)) (dE_phi/drho + E_phi/rho - (1/rho) dE_r/dphi).
    Hz = np.zeros((3, len(phi_c)), dtype=complex)
    for i, rho in enumerate(rhos):
        Hz[i] = -(1.0 / (1j * k)) * (dEp_dr[i] + Ep[i] / rho - dEr_dphi[i] / rho)
    dHz_dr = (Hz[2] - Hz[0]) / (2.0 * d)
    dl = 2.0 * np.pi * r2 / len(phi_c)
    rp = np.column_stack([np.cos(phi_c), np.sin(phi_c)]) * r2
    nhat = np.column_stack([np.cos(phi_c), np.sin(phi_c)])
    sig = np.zeros(len(phis))
    for i, ph in enumerate(phis):
        rhat = np.array([np.cos(ph), np.sin(ph)])
        kern = np.exp(1j * k * (rp @ rhat))
        I = np.sum((Hz[1] * (1j * k * (nhat @ rhat)) - dHz_dr) * kern) * dl
        sig[i] = np.abs(I) ** 2 / (4.0 * k)  # |H_inc| = 1 in internal units
    return sig


def main():
    rows = []
    for ka in KA:
        d = WORK / f"ka_{ka:g}"
        if RUN:
            make_mesh(ka)
            make_config(ka)
            log = d / "run.log"
            with open(log, "w") as fh:
                rc = subprocess.run([PALACE, "-np", str(NPROC), f"ka_{ka:g}.json"], cwd=d,
                                    stdout=fh, stderr=subprocess.STDOUT).returncode
            print(f"ka={ka:g}: rc={rc}", flush=True)
        probe_csv = d / "postpro" / f"ka_{ka:g}" / "probe-E.csv"
        if not probe_csv.exists():
            print(f"ka={ka:g}: no probe output", flush=True)
            continue
        prm = params(ka)
        k = 2.0 * np.pi * prm["f"] / C0
        E = read_probes(probe_csv, NPROBE)
        phi_c = 2.0 * np.pi * np.arange(NPROBE) / NPROBE
        phis = np.linspace(0.0, 2.0 * np.pi, 361)
        sig = kirchhoff_far(E, phi_c, prm["rhos"], k, phis)
        mie = te_far(A, prm["f"], phis)
        err = (10 * np.log10(np.maximum(sig, 1e-300))
               - 10 * np.log10(np.maximum(mie, 1e-300)))
        rows.append({"ka": ka, "mono_err": float(err[0]),
                     "rms": float(np.sqrt(np.mean(err**2))),
                     "max": float(np.max(np.abs(err)))})
        print(f"ka={ka:g}: mono err {rows[-1]['mono_err']:+.2f} dB  "
              f"RMS {rows[-1]['rms']:.2f} dB  max {rows[-1]['max']:.2f} dB",
              flush=True)
    if rows:
        with open(WORK / "cyl_validation.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"Wrote {WORK / 'cyl_validation.csv'}")


if __name__ == "__main__":
    main()
