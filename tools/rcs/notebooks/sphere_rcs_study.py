# %% [markdown]
# # PEC sphere RCS study
#
# Reproducible study of the radar cross section of a perfectly conducting
# sphere using Palace plane wave excitation and the analytic Mie series:
#
# 1. Monostatic normalized RCS `sigma/(pi a^2)` vs `ka` (circumference/wavelength),
#    reproducing the classic Mie theory figure, with Rayleigh and optical asymptotes.
# 2. Bistatic RCS of a 1 m radius sphere at 300 MHz (`ka = 2*pi`) in the E- and
#    H-planes, Palace finite elements vs Mie theory.
# 3. ParaView-compatible field output: scattered-field VTU and a 3D dB-scaled RCS
#    pattern VTP, with offscreen renders via `pvbatch3.12`.
#
# This file is in [jupytext](https://jupytext.readthedocs.io/) percent format. Run it
# directly (`python sphere_rcs_study.py`) or convert with
# `jupytext --to ipynb sphere_rcs_study.py`.
#
# Dependencies: numpy, scipy, matplotlib, gmsh, meshio, jupytext (see
# `tools/rcs/requirements.txt`). Set `RUN_PALACE=1` to generate the mesh and run the
# solver; otherwise the Mie figures are produced from the analytic solution alone.

# %%
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams.update({"figure.dpi": 110, "font.size": 10})

TOOLS = Path(__file__).resolve().parents[1] if "__file__" in globals() else Path(
    "/home/luis/palace/tools/rcs")
sys.path.insert(0, str(TOOLS))

from mie_pec import monostatic_normalized, mie_pec  # noqa: E402

C0 = 299792458.0
PALACE = os.environ.get("PALACE_BIN", "/home/luis/palace/build/bin/palace")
WORKDIR = Path(os.environ.get("RCS_WORKDIR", "./rcs_work")).resolve()
FIGDIR = WORKDIR / "figures"
CASE = os.environ.get("RCS_CASE", "sphere1m")
RUN_PALACE = os.environ.get("RUN_PALACE", "0") == "1"
NPROC = int(os.environ.get("RCS_NPROC", "8"))
WORKDIR.mkdir(parents=True, exist_ok=True)
FIGDIR.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## 1. Monostatic normalized RCS (Mie theory)
#
# `ka` is the circumference divided by the wavelength. The Rayleigh asymptote is
# `9 (ka)^4` and the optical limit is `1` (projected area).

# %%
ka = np.logspace(np.log10(0.01), np.log10(30.0), 1500)
sigma_norm = monostatic_normalized(ka)
rayleigh = 9.0 * ka**4

fig, ax = plt.subplots(figsize=(8.0, 6.0))
ax.loglog(ka, sigma_norm, color="#1f77b4", lw=1.8, label="Mie theory (monostatic)")
ax.loglog(ka, rayleigh, color="#d62728", lw=1.0, ls="--",
          label=r"Rayleigh: $9(ka)^4$")
ax.axhline(1.0, color="0.4", lw=1.0, ls=":", label="Optical limit: 1")
ax.set_xlim(0.01, 30.0)
ax.set_ylim(1.0e-3, 10.0)
ax.set_xlabel("relative frequency = circumference / wavelength  ($ka$)")
ax.set_ylabel(r"monostatic RCS / projected area  ($\sigma/(\pi a^2)$)")
ax.set_title("Radar cross section of a metal sphere from Mie theory")
ax.grid(True, which="both", alpha=0.3)
ax.legend(loc="upper left", fontsize=9)
ip = int(np.argmax(sigma_norm))
ax.annotate(f"peak = {sigma_norm[ip]:.2f} at $ka$ = {ka[ip]:.2f}",
            xy=(ka[ip], sigma_norm[ip]), xytext=(0.045, 3.0), fontsize=8,
            arrowprops=dict(arrowstyle="->", lw=0.8))
fig.tight_layout()
fig.savefig(FIGDIR / "sphere_monostatic_mie.png")
print("monostatic peak", sigma_norm[ip], "at ka", ka[ip])

# %% [markdown]
# ## 2. Bistatic RCS of a 1 m sphere at 300 MHz
#
# `ka = 2*pi`, so the sphere circumference is one wavelength. The plane wave
# propagates along `-z` with the electric field along `x` (matching the Mie
# convention: backscatter at `theta = 0`, forward at `theta = 180`).

# %%
RADIUS = 1.0
FREQ = 0.3e9
BOX_HALF = 1.7
KA = 2.0 * np.pi * FREQ * RADIUS / C0
print(f"ka = {KA:.4f}")

thetas = np.arange(0.0, 181.0, 1.0)
phis = [0.0, 90.0]

# Mie reference.
mie = {}
for phi in phis:
    et, ep = mie_pec(RADIUS, FREQ, np.radians(thetas), [np.radians(phi)])
    mie[phi] = {"theta": np.abs(et) ** 2, "phi": np.abs(ep) ** 2}


def make_config(path):
    import json
    tp = [[float(t), float(p)] for p in phis for t in thetas]
    cfg = {
        "Problem": {"Type": "Driven", "Verbose": 1, "Output": f"postpro/{CASE}"},
        "Model": {"Mesh": "sphere1m.msh", "L0": 1.0},
        "Domains": {"Materials": [{"Attributes": [1]}]},
        "Boundaries": {
            "PEC": {"Attributes": [2]},
            "Absorbing": {"Attributes": [1], "Order": 2},
            "PlaneWave": [{"Index": 1, "Attributes": [1], "Excitation": 1,
                           "Direction": [0.0, 0.0, -1.0],
                           "Polarization": [1.0, 0.0, 0.0], "Amplitude": 1.0}],
            "Postprocessing": {"FarField": {"Attributes": [1], "ThetaPhis": tp,
                                            "Scattered": True}},
        },
        "Solver": {"Order": 2, "Driven": {"Samples": [{"Type": "Point",
                                                       "Freq": [FREQ / 1e9],
                                                       "SaveStep": 1}]}},
    }
    Path(path).write_text(json.dumps(cfg, indent=1))


if RUN_PALACE:
    subprocess.run([sys.executable, str(TOOLS / "sphere_mesh.py"),
                    "sphere1m.msh", str(RADIUS), str(BOX_HALF), "0.045", "0.15"],
                   cwd=WORKDIR, check=True)
    make_config(WORKDIR / "sphere1m.json")
    subprocess.run([PALACE, "-np", str(NPROC), "sphere1m.json"], cwd=WORKDIR,
                   check=True)

# %% [markdown]
# Parse the scattered far field from `farfield-rE.csv` and compare with Mie.

# %%
def read_farfield(path):
    rows = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding="utf-8")
    names = list(rows.dtype.names)
    # The CSV stores theta/phi in degrees.
    theta = rows[names[2]].astype(float)
    phi = rows[names[3]].astype(float)
    er = np.column_stack([rows[names[4]], rows[names[6]], rows[names[8]]])
    ei = np.column_stack([rows[names[5]], rows[names[7]], rows[names[9]]])
    return theta, phi, er + 1j * ei


def rcs_components(theta_deg, phi_deg, rE, e_inc=1.0):
    th, ph = np.radians(theta_deg), np.radians(phi_deg)
    th_hat = np.column_stack([np.cos(th) * np.cos(ph), np.cos(th) * np.sin(ph),
                              -np.sin(th)])
    ph_hat = np.column_stack([-np.sin(ph), np.cos(ph), np.zeros_like(ph)])
    e_th = np.einsum("ij,ij->i", rE, th_hat)
    e_ph = np.einsum("ij,ij->i", rE, ph_hat)
    return 4.0 * np.pi * np.abs(e_th) ** 2 / e_inc**2, \
        4.0 * np.pi * np.abs(e_ph) ** 2 / e_inc**2


ff = WORKDIR / "postpro" / CASE / "farfield-rE.csv"
if ff.exists():
    theta_r, phi_r, rE = read_farfield(ff)
    palace = {}
    for phi in phis:
        m = np.isclose(phi_r, phi)
        s_th, s_ph = rcs_components(theta_r[m], phi_r[m], rE[m])
        palace[phi] = {"theta": s_th, "phi": s_ph, "theta_deg": theta_r[m]}

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), subplot_kw={"projection": "polar"})
    for ax, phi, label in zip(axes, phis, ["E-plane ($\\phi=0$)", "H-plane ($\\phi=90$)" ]):
        ax.plot(np.radians(thetas), 10 * np.log10(mie[phi]["theta"] + 1e-30),
                "k-", lw=1.2, label="Mie $\\sigma_{\\theta\\theta}$")
        ax.plot(np.radians(thetas), 10 * np.log10(mie[phi]["phi"] + 1e-30),
                "k--", lw=1.0, label="Mie $\\sigma_{\\phi\\phi}$")
        if phi in palace:
            p = palace[phi]
            ax.plot(np.radians(p["theta_deg"]), 10 * np.log10(p["theta"] + 1e-30),
                    "C0.", ms=2, label="Palace $\\sigma_{\\theta\\theta}$")
            ax.plot(np.radians(p["theta_deg"]), 10 * np.log10(p["phi"] + 1e-30),
                    "C1.", ms=2, label="Palace $\\sigma_{\\phi\\phi}$")
        ax.set_title(label)
        ax.set_theta_zero_location("N")
        ax.legend(loc="lower left", fontsize=7)
    fig.suptitle(f"Bistatic RCS, 1 m PEC sphere, {FREQ/1e9:.2f} GHz ($ka$={KA:.2f})")
    fig.tight_layout()
    fig.savefig(FIGDIR / "sphere_bistatic_1m_300MHz_polar.png")

    fig, ax = plt.subplots(figsize=(9, 5))
    for phi, color in zip(phis, ["C0", "C1"]):
        p = palace[phi]
        ax.plot(thetas, 10 * np.log10(mie[phi]["theta"] + 1e-30), "k-", lw=1.2)
        ax.plot(thetas, 10 * np.log10(mie[phi]["phi"] + 1e-30), "k--", lw=1.0)
        ax.plot(p["theta_deg"], 10 * np.log10(p["theta"] + 1e-30), color + ".", ms=3)
        ax.plot(p["theta_deg"], 10 * np.log10(p["phi"] + 1e-30), color + "x", ms=3)
    ax.set_xlabel("$\\theta$ (deg.)")
    ax.set_ylabel("RCS (dBsm)")
    ax.set_title("Bistatic RCS, Palace (markers) vs Mie (lines): "
                 "$\\sigma_{\\theta\\theta}$ solid, $\\sigma_{\\phi\\phi}$ dashed")
    ax.grid(alpha=0.3)
    ax.legend(["Mie $\\sigma_{\\theta\\theta}$", "Mie $\\sigma_{\\phi\\phi}$",
               "Palace E-plane", "Palace H-plane"], fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGDIR / "sphere_bistatic_1m_300MHz_db.png")
    print("bistatic figures written")
else:
    print(f"{ff} not found; run with RUN_PALACE=1 to generate it")

# %% [markdown]
# ## 3D RCS pattern
#
# Build a ParaView point-cloud pattern from a full-sphere Mie computation (fast)
# and, when Palace output is available, a dB-scaled pattern from its far-field
# CSV. The radial coordinate is scaled in dB over a 40 dB range.

# %%
import csv

def mie_pattern_csv(path, a, f, dtheta=2.0, dphi=2.0):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["f (GHz)", "exc", "theta (deg.)", "phi (deg.)",
                    "RCS (m^2)", "RCS (dBsm)"])
        for phi_deg in np.arange(0.0, 360.0, dphi):
            th = np.radians(np.arange(0.0, 181.0, dtheta))
            et, ep = mie_pec(a, f, th, np.radians(phi_deg))
            rcs = np.abs(et) ** 2 + np.abs(ep) ** 2
            for t, s in zip(np.arange(0.0, 181.0, dtheta), rcs):
                w.writerow([f / 1e9, 1, t, phi_deg, s,
                            10 * np.log10(max(s, 1e-30))])


mie_csv = WORKDIR / "sphere_mie_pattern.csv"
mie_pattern_csv(mie_csv, RADIUS, FREQ)
subprocess.run([sys.executable, str(TOOLS / "pattern_vtp.py"), str(mie_csv),
                str(WORKDIR / "sphere_mie_pattern.vtp")], check=True)

# 3D image from the pattern CSV.
rows = list(csv.reader(open(mie_csv)))[1:]
th = np.radians([float(r[2]) for r in rows])
ph = np.radians([float(r[3]) for r in rows])
db = np.array([float(r[5]) for r in rows])
peak = db.max()
r = 0.2 + 0.8 * np.clip((db - (peak - 40.0)) / 40.0, 0.0, 1.0)
fig = plt.figure(figsize=(9, 7))
ax = fig.add_subplot(111, projection="3d")
p = ax.scatter(r * np.sin(th) * np.cos(ph), r * np.sin(th) * np.sin(ph),
               r * np.cos(th), c=db, cmap="jet", s=4, vmin=peak - 40, vmax=peak)
ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
ax.set_title("PEC sphere RCS pattern from Mie theory (dB-scaled radius)\n"
             f"1 m radius, {FREQ/1e9:.0f} MHz")
fig.colorbar(p, ax=ax, shrink=0.6, label="RCS (dBsm)")
fig.tight_layout()
fig.savefig(FIGDIR / "sphere_rcs_pattern_3d.png")
print("3D pattern figure written")

# %% [markdown]
# ## ParaView-compatible output
#
# Palace writes total-field VTU/PVD files under
# `postpro/sphere1m/paraview/driven/`. The scattered field VTU is produced by
# subtracting the analytic incident field (`vtk` package; Palace writes VTK XML
# 2.2 which meshio cannot read), and the far-field pattern is exported as a
# dB-scaled point cloud for ParaView.

# %%
import glob

vtu = sorted(glob.glob(str(WORKDIR / f"postpro/{CASE}/paraview/driven/**/*.vtu"),
                       recursive=True))
if vtu:
    subprocess.run([sys.executable, str(TOOLS / "vtu_scattered.py"), vtu[-1],
                    str(WORKDIR / "vis" / "sphere1m_scattered.vtu"),
                    "--direction", "0", "0", "-1", "--polarization", "1", "0", "0",
                    "--frequency", str(FREQ)], check=True)
    subprocess.run([sys.executable, str(TOOLS / "pattern_vtp.py"),
                    str(ff), str(WORKDIR / "vis" / "sphere1m_pattern.vtp")], check=True)
    print("scattered VTU and pattern VTP written")
else:
    print("no VTU files found; run with RUN_PALACE=1")

# %% [markdown]
# Offscreen ParaView renders (requires `pvbatch3.12`):
#
# ```sh
# pvbatch3.12 --force-offscreen-rendering tools/rcs/render_paraview.py \
#   --input vis/sphere1m_scattered.vtu --output figures/sphere_scattered_slice.png \
#   --field E_scat_mag --log --normal 0 1 0
# pvbatch3.12 --force-offscreen-rendering tools/rcs/render_paraview.py \
#   --input vis/sphere1m_pattern.vtp --output figures/sphere_rcs_pattern_3d.png \
#   --field RCS_dB --points
# ```
