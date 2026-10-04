# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Radar cross section of a sphere
#
# This notebook computes the radar cross section (RCS) of a sphere of 1 m diameter with
# *Palace*, using an incident plane wave excitation (`config["Domains"]["PlaneWave"]`), the
# exact spherical DtN absorbing boundary condition (`config["Boundaries"]["Absorbing"]` with
# `"DtN": true`), and the near-to-far-field transformation
# (`config["Boundaries"]["Postprocessing"]["FarField"]`), see
# [the documentation](../../docs/src/features/farfield.md) and
# [the example page](../../docs/src/examples/sphere_rcs.md). The results are validated
# against the Mie series.
#
# Four cases are computed:
#
#   1. Bistatic RCS of a perfectly conducting (PEC) sphere at 150 MHz and 300 MHz, in the
#      E-plane and H-plane.
#   2. Monostatic (backscattering) RCS of the PEC sphere from 20 MHz to 480 MHz, from the
#      Rayleigh region (ka ≪ 1) into the optical region.
#   3. Bistatic RCS of a dielectric sphere (εᵣ = 4) at 300 MHz.
#   4. Bistatic RCS of the PEC sphere at 300 MHz using the standard second-order absorbing
#      boundary condition instead of the DtN boundary condition, for comparison.
#
# In all cases the incident plane wave propagates along -z with the electric field along
# +x, so that backscattering is at θ = 0 and forward scattering is at θ = 180°, the
# E-plane is φ = 0 and the H-plane is φ = 90°.
#
# This file is a [Jupytext](https://jupytext.readthedocs.io) script: open it as a notebook
# with `jupytext --to notebook sphere_rcs.py`, or run it directly with `python
# sphere_rcs.py`. Requirements: `numpy`, `scipy`, `matplotlib`, and the `gmsh` Python API.
# The *Palace* launcher is found from the `PALACE` environment variable (default:
# `palace`, from the `PATH`), and the number of MPI processes from `PALACE_NP` (default 8).
# Simulations whose results already exist in `postpro/` are not run again unless
# `PALACE_RERUN=1` is set. With 16 processes, the full set of simulations takes about one
# hour, most of it for the 47 frequencies of the monostatic sweep.

# %%
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import matplotlib

if "ipykernel" not in sys.modules:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    EXAMPLE_DIR = Path(__file__).resolve().parent
except NameError:  # Running as a notebook
    EXAMPLE_DIR = Path.cwd()
sys.path.insert(0, str(EXAMPLE_DIR))
sys.path.insert(0, str(EXAMPLE_DIR / "mesh"))

import mie  # noqa: E402
from mesh import generate_sphere_mesh  # noqa: E402

PALACE = os.environ.get("PALACE", "palace")
NP = int(os.environ.get("PALACE_NP", "8"))
FIGURE_DIR = EXAMPLE_DIR / "figures"
FIGURE_DIR.mkdir(exist_ok=True)

# %% [markdown]
# ## Parameters
#
# The sphere has radius `a = 0.5 m`. The computational domain is truncated by a concentric
# sphere of radius `R = 1 m`, only 0.5 wavelengths away from the target at 300 MHz: the
# exact DtN boundary condition allows a small domain without compromising accuracy. The
# mesh uses curved (second-order) tetrahedra, and the solution uses third-order Nédélec
# elements. With an element size of 0.1 m on the outer boundary, there are more than six
# elements per wavelength up to 480 MHz.

# %%
RADIUS = 0.5  # Sphere radius (m)
RADIUS_FARFIELD = 1.0  # Radius of the absorbing boundary (m)
H_SPHERE = 0.06  # Element size on the sphere (m)
H_FARFIELD = 0.1  # Element size on the absorbing boundary (m)
H_DIELECTRIC = 0.05  # Element size inside the dielectric sphere (m)
EPS_R = 4.0  # Relative permittivity of the dielectric sphere
SOLVER_ORDER = 3  # Order of the Nédélec finite element space

FREQS_BISTATIC = [0.15, 0.3]  # GHz
FREQ_MONOSTATIC = (0.02, 0.48, 0.01)  # GHz: min, max, step
THETAS = np.arange(0.0, 180.0 + 0.5, 1.0)  # Observation angles for bistatic RCS (deg.)

# %% [markdown]
# ## Meshes
#
# The meshes are generated with Gmsh (see `mesh/mesh.py`). Attributes: domain 1 is air,
# domain 2 the interior of the dielectric sphere, boundary 1 the absorbing boundary, and
# boundary 2 the surface of the PEC sphere.

# %%
mesh_pec = EXAMPLE_DIR / "mesh" / "sphere_pec.msh"
mesh_dielectric = EXAMPLE_DIR / "mesh" / "sphere_dielectric.msh"
generate_sphere_mesh(
    filename=str(mesh_pec),
    radius=RADIUS,
    radius_farfield=RADIUS_FARFIELD,
    h_sphere=H_SPHERE,
    h_farfield=H_FARFIELD,
)
generate_sphere_mesh(
    filename=str(mesh_dielectric),
    radius=RADIUS,
    radius_farfield=RADIUS_FARFIELD,
    h_sphere=H_SPHERE,
    h_farfield=H_FARFIELD,
    h_interior=H_DIELECTRIC,
    dielectric=True,
)

# %% [markdown]
# ## Configuration
#
# The incident plane wave is specified in `config["Domains"]["PlaneWave"]`, with its
# propagation direction, polarization, and amplitude (1 V/m). The far field is computed on
# the absorbing boundary (attribute 1) for the requested directions `"ThetaPhis"`. For
# plane wave excitations, *Palace* writes the far field of the scattered field to
# `farfield-rE.csv` and the RCS to `farfield-rcs.csv`.


# %%
def thetaphis_cuts(thetas):
    """Observation directions for the E-plane (φ = 0) and H-plane (φ = 90°) cuts."""
    return [[float(t), 0.0] for t in thetas] + [[float(t), 90.0] for t in thetas]


def make_config(name, mesh, samples, thetaphis, dielectric=False, save_step=0, dtn=True):
    materials = [{"Attributes": [1], "Permeability": 1.0, "Permittivity": 1.0}]
    absorbing = {"Attributes": [1], "Order": 1, "DtN": True}
    if not dtn:
        absorbing = {"Attributes": [1], "Order": 2}
    boundaries = {
        "Absorbing": absorbing,
        "Postprocessing": {"FarField": {"Attributes": [1], "ThetaPhis": thetaphis}},
    }
    if dielectric:
        materials.append({"Attributes": [2], "Permeability": 1.0, "Permittivity": EPS_R})
    else:
        boundaries["PEC"] = {"Attributes": [2]}
    for s in samples:
        s["SaveStep"] = save_step
    return {
        "Problem": {"Type": "Driven", "Verbose": 2, "Output": f"postpro/{name}"},
        "Model": {"Mesh": os.path.relpath(mesh, EXAMPLE_DIR), "L0": 1.0},
        "Domains": {
            "Materials": materials,
            "PlaneWave": [
                {
                    "Index": 1,
                    "Direction": [0.0, 0.0, -1.0],
                    "Polarization": [1.0, 0.0, 0.0],
                    "Amplitude": 1.0,
                }
            ],
        },
        "Boundaries": boundaries,
        "Solver": {
            "Order": SOLVER_ORDER,
            "Device": "CPU",
            "Driven": {"Samples": samples},
            "Linear": {
                "Type": "Default",
                "KSPType": "GMRES",
                "Tol": 1.0e-8,
                "MaxIts": 200,
                "ComplexCoarseSolve": True,
            },
        },
    }


configs = {
    "sphere_pec_bistatic": make_config(
        "sphere_pec_bistatic",
        mesh_pec,
        [{"Type": "Point", "Freq": FREQS_BISTATIC}],
        thetaphis_cuts(THETAS),
        save_step=1,
    ),
    "sphere_pec_monostatic": make_config(
        "sphere_pec_monostatic",
        mesh_pec,
        [
            {
                "Type": "Linear",
                "MinFreq": FREQ_MONOSTATIC[0],
                "MaxFreq": FREQ_MONOSTATIC[1],
                "FreqStep": FREQ_MONOSTATIC[2],
            }
        ],
        [[0.0, 0.0]],
    ),
    "sphere_dielectric_bistatic": make_config(
        "sphere_dielectric_bistatic",
        mesh_dielectric,
        [{"Type": "Point", "Freq": [0.3]}],
        thetaphis_cuts(THETAS),
        dielectric=True,
    ),
    "sphere_pec_bistatic_abc": make_config(
        "sphere_pec_bistatic_abc",
        mesh_pec,
        [{"Type": "Point", "Freq": [0.3]}],
        thetaphis_cuts(THETAS),
        dtn=False,
    ),
}
for name, config in configs.items():
    with open(EXAMPLE_DIR / f"{name}.json", "w") as f:
        json.dump(config, f, indent=2)
        f.write("\n")

# %% [markdown]
# ## Run *Palace*


# %%
def run_palace(name):
    """Run Palace for the configuration file <name>.json in the example directory."""
    result = EXAMPLE_DIR / "postpro" / name / "farfield-rcs.csv"
    if result.exists() and os.environ.get("PALACE_RERUN", "0") != "1":
        print(f"{name}: using existing results in {result.parent}")
        return
    t0 = time.time()
    log = EXAMPLE_DIR / "postpro" / f"{name}.log"
    log.parent.mkdir(exist_ok=True)
    with open(log, "w") as f:
        subprocess.run(
            [PALACE, "-np", str(NP), f"{name}.json"],
            cwd=EXAMPLE_DIR,
            stdout=f,
            stderr=subprocess.STDOUT,
            check=True,
        )
    print(f"{name}: completed in {time.time() - t0:.1f} s (log: {log})")


for name in configs:
    if os.environ.get("SKIP_SOLVE", "0") != "1":
        run_palace(name)

# %% [markdown]
# ## Postprocessing


# %%
def load_rcs(name):
    """Load farfield-rcs.csv as a dictionary of arrays keyed by column."""
    path = EXAMPLE_DIR / "postpro" / name / "farfield-rcs.csv"
    data = np.loadtxt(path, delimiter=",", skiprows=1)
    keys = ["f", "exc", "theta", "phi", "rcs", "rcs_theta", "rcs_phi"]
    return {k: data[:, i] for i, k in enumerate(keys)}


def cut(data, phi, f=None, eps_r=None):
    """
    Return the sorted observation angles, the computed RCS, and the Mie series RCS for the
    E-plane (phi = 0) or H-plane (phi = 90) cut. Directions duplicated between cuts (the
    poles) are only reported once by Palace, so the angles are taken from the data.
    """
    sel = np.isclose(data["phi"], phi)
    if f is not None:
        sel &= np.isclose(data["f"], f)
    order = np.argsort(data["theta"][sel])
    th = data["theta"][sel][order]
    freq = data["f"][sel][0] * 1e9
    sE, sH = mie.bistatic_rcs(RADIUS, freq, np.radians(180.0 - th), eps_r=eps_r)
    return th, data["rcs"][sel][order], (sE if phi == 0.0 else sH)


def to_db(sigma):
    """RCS in dBsm."""
    return 10.0 * np.log10(sigma)


def print_errors(label, sigma, sigma_ref):
    err_db = to_db(sigma) - to_db(sigma_ref)
    rel = np.abs(sigma - sigma_ref) / np.max(sigma_ref)
    print(
        f"{label:44s} max |Δ| = {np.max(np.abs(err_db)):.3f} dB, "
        f"RMS Δ = {np.sqrt(np.mean(err_db**2)):.3f} dB, "
        f"max |Δσ| / max σ = {100 * np.max(rel):.2f} %"
    )


C0 = mie.C0

# %% [markdown]
# ### Bistatic RCS of the PEC sphere
#
# The scattering angle measured from the forward direction is ψ = 180° - θ. In the E-plane
# the scattered field is θ-polarized (co-polarized), and in the H-plane it is φ-polarized.

# %%
data = load_rcs("sphere_pec_bistatic")
fig, axes = plt.subplots(1, len(FREQS_BISTATIC), figsize=(12, 4.5), sharey=False)
for ax, f in zip(np.atleast_1d(axes), FREQS_BISTATIC):
    ka = 2 * np.pi * f * 1e9 * RADIUS / C0
    psi = np.radians(180.0 - THETAS)
    sE, sH = mie.bistatic_rcs(RADIUS, f * 1e9, psi)
    for phi, mie_curve, label, color in [
        (0.0, sE, "E-plane", "C0"),
        (90.0, sH, "H-plane", "C1"),
    ]:
        th, sigma, ref = cut(data, phi, f)
        ax.plot(THETAS, to_db(mie_curve), "-", color=color, lw=1.5, label=f"Mie, {label}")
        ax.plot(
            th[::4],
            to_db(sigma)[::4],
            "o",
            color=color,
            mfc="none",
            ms=5,
            label=f"Palace, {label}",
        )
        print_errors(f"PEC bistatic {label}, f = {f} GHz (ka = {ka:.2f})", sigma, ref)
    ax.set_title(f"PEC sphere, a = {RADIUS} m, f = {1e3 * f:.0f} MHz (ka = {ka:.2f})")
    ax.set_xlabel("θ (deg.)  [θ = 0: backscattering, θ = 180°: forward]")
    ax.set_ylabel("RCS (dBsm)")
    ax.set_xlim(0, 180)
    ax.set_xticks(np.arange(0, 181, 30))
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(FIGURE_DIR / "sphere_pec_bistatic.png", dpi=150)

# %% [markdown]
# ### Monostatic RCS of the PEC sphere
#
# The backscattering RCS normalized by the geometric cross section πa², as a function of
# the electrical size ka, shows the Rayleigh (ka ≪ 1), resonance, and optical (σ → πa²)
# regions.

# %%
data = load_rcs("sphere_pec_monostatic")
f_fine = np.linspace(FREQ_MONOSTATIC[0], FREQ_MONOSTATIC[1], 1000) * 1e9
ka_fine = 2 * np.pi * f_fine * RADIUS / C0
ref_fine = mie.monostatic_rcs(RADIUS, f_fine)
ka = 2 * np.pi * data["f"] * 1e9 * RADIUS / C0
ref = mie.monostatic_rcs(RADIUS, data["f"] * 1e9)
print_errors("PEC monostatic", data["rcs"], ref)

fig, (ax0, ax1) = plt.subplots(
    2, 1, figsize=(8, 6.5), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
)
area = np.pi * RADIUS**2
ax0.semilogy(ka_fine, ref_fine / area, "-", color="C0", lw=1.5, label="Mie series")
ax0.semilogy(ka, data["rcs"] / area, "o", color="C1", mfc="none", ms=5, label="Palace")
ax0.set_ylabel("σ / πa²")
ax0.set_title(f"Monostatic RCS of a PEC sphere, a = {RADIUS} m")
ax0.grid(alpha=0.3, which="both")
ax0.legend()
ax1.plot(ka, to_db(data["rcs"]) - to_db(ref), "o-", color="C1", ms=3)
ax1.set_xlabel("ka")
ax1.set_ylabel("Error (dB)")
ax1.grid(alpha=0.3)
secax = ax0.secondary_xaxis(
    "top",
    functions=(
        lambda x: 1e-6 * x * C0 / (2 * np.pi * RADIUS),
        lambda f: 1e6 * f * 2 * np.pi * RADIUS / C0,
    ),
)
secax.set_xlabel("Frequency (MHz)")
fig.tight_layout()
fig.savefig(FIGURE_DIR / "sphere_pec_monostatic.png", dpi=150)

# %% [markdown]
# ### Bistatic RCS of the dielectric sphere

# %%
data = load_rcs("sphere_dielectric_bistatic")
f = 0.3
ka = 2 * np.pi * f * 1e9 * RADIUS / C0
psi = np.radians(180.0 - THETAS)
sE, sH = mie.bistatic_rcs(RADIUS, f * 1e9, psi, eps_r=EPS_R)
fig, ax = plt.subplots(figsize=(7, 4.5))
for phi, mie_curve, label, color in [
    (0.0, sE, "E-plane", "C0"),
    (90.0, sH, "H-plane", "C1"),
]:
    th, sigma, ref = cut(data, phi, eps_r=EPS_R)
    ax.plot(THETAS, to_db(mie_curve), "-", color=color, lw=1.5, label=f"Mie, {label}")
    ax.plot(
        th[::4],
        to_db(sigma)[::4],
        "o",
        color=color,
        mfc="none",
        ms=5,
        label=f"Palace, {label}",
    )
    print_errors(f"Dielectric bistatic {label}", sigma, ref)
ax.set_title(
    f"Dielectric sphere (εᵣ = {EPS_R:g}), a = {RADIUS} m, f = 300 MHz (ka = {ka:.2f})"
)
ax.set_xlabel("θ (deg.)  [θ = 0: backscattering, θ = 180°: forward]")
ax.set_ylabel("RCS (dBsm)")
ax.set_xlim(0, 180)
ax.set_xticks(np.arange(0, 181, 30))
ax.grid(alpha=0.3)
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(FIGURE_DIR / "sphere_dielectric_bistatic.png", dpi=150)

# %% [markdown]
# ### Absorbing boundary condition versus DtN boundary condition
#
# With the outer boundary only half a wavelength away from the sphere, the reflections of
# the second-order absorbing boundary condition (which, as most local absorbing conditions,
# is only exact for normally incident plane waves) produce errors of the order of 1 dB in
# the RCS, much larger than the discretization error. The exact DtN boundary condition
# removes them.

# %%
data_abc = load_rcs("sphere_pec_bistatic_abc")
data_dtn = load_rcs("sphere_pec_bistatic")
f = 0.3
ka = 2 * np.pi * f * 1e9 * RADIUS / C0
sE, sH = mie.bistatic_rcs(RADIUS, f * 1e9, np.radians(180.0 - THETAS))
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
for ax, (phi, mie_curve, label) in zip(axes, [(0.0, sE, "E-plane"), (90.0, sH, "H-plane")]):
    ax.plot(THETAS, to_db(mie_curve), "-", color="k", lw=1.5, label="Mie series")
    for d, bc, color, marker in [
        (data_abc, "2nd-order ABC", "C3", "s"),
        (data_dtn, "DtN", "C0", "o"),
    ]:
        th, sigma, ref = cut(d, phi, f)
        ax.plot(
            th[::4],
            to_db(sigma)[::4],
            marker,
            color=color,
            mfc="none",
            ms=5,
            label=f"Palace, {bc}",
        )
        print_errors(f"PEC bistatic {label}, {bc}", sigma, ref)
    ax.set_title(f"PEC sphere, {label}, f = 300 MHz (ka = {ka:.2f}), R = {RADIUS_FARFIELD} m")
    ax.set_xlabel("θ (deg.)  [θ = 0: backscattering, θ = 180°: forward]")
    ax.set_ylabel("RCS (dBsm)")
    ax.set_xlim(0, 180)
    ax.set_xticks(np.arange(0, 181, 30))
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(FIGURE_DIR / "sphere_pec_abc_vs_dtn.png", dpi=150)

# %% [markdown]
# The total field (incident plus scattered) is written for visualization with ParaView in
# `postpro/sphere_pec_bistatic/paraview` (`"SaveStep": 1` for the bistatic PEC case).

# %%
if "ipykernel" in sys.modules:
    plt.show()
