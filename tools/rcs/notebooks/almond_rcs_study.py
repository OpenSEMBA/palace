# %% [markdown]
# # NASA almond monostatic RCS study (scattered-field formulation)
#
# Monostatic RCS of the EMCC/NASA PEC almond (length 9.936 in) at 1.19 GHz using
# the scattered-field plane wave formulation, with a resolution study. The
# incident wave is injected through the PEC surface and the dielectric/volume
# contrast machinery is inactive (vacuum), so the ABC only sees the scattered
# field.
#
# Geometry conventions: the almond axis is x; azimuth `phi` rotates the
# incidence in the x-y plane, with `phi = 0` tip-on. The incident wave travels
# along `-Direction` for the physical setup used here (SF propagation is
# `+Direction`, so `Direction = -k`), and the backscatter is `+k`, i.e.
# `theta = 90`, `phi = phi_az`. Excitations alternate VV (z) and HH (in-plane)
# polarization.
#
# Reference: set `RCS_ALMOND_REFERENCE` to a CSV with columns
# `phi_deg,VV_dBsm,HH_dBsm` (e.g. parsed from the Austin RCS Benchmark
# "Problem Set IIIA-PEC Almonds" reference data). Without it, only the Palace
# results are plotted.
#
# This file is jupytext percent format.

# %%
import csv
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

matplotlib.rcParams.update({"figure.dpi": 110, "font.size": 10})

TOOLS = Path(__file__).resolve().parents[1] if "__file__" in globals() else Path(
    "/home/luis/palace/tools/rcs")
PALACE = os.environ.get("PALACE_BIN", "/home/luis/palace/build/bin/palace")
WORKDIR = Path(os.environ.get("RCS_WORKDIR", "./almond_work")).resolve()
FIGDIR = WORKDIR / "figures"
RUN_PALACE = os.environ.get("RUN_PALACE", "0") == "1"
NPROC = int(os.environ.get("RCS_NPROC", "8"))
FREQ_GHZ = float(os.environ.get("ALMOND_FREQ_GHZ", "1.19"))
OUTER_RADIUS = float(os.environ.get("ALMOND_OUTER_R", "0.8"))
REFERENCE_CSV = Path(os.environ.get("RCS_ALMOND_REFERENCE",
                                    str(WORKDIR / "reference_almond.csv")))
WORKDIR.mkdir(parents=True, exist_ok=True)
FIGDIR.mkdir(parents=True, exist_ok=True)

# Resolution levels: (name, h at the almond surface, h far).
RESOLUTIONS = {
    "L1": (0.015, 0.050),
    "L2": (0.009, 0.035),
    "L3": (0.005, 0.022),
}
AZIMUTHS = [0.0, 45.0, 90.0, 135.0, 180.0]  # tip, ..., broadside, ..., tail


def make_config(case, mesh):
    waves = []
    idx = 1
    for phi_deg in AZIMUTHS:
        phi = math.radians(phi_deg)
        k = [math.cos(phi), math.sin(phi), 0.0]
        hh = [-math.sin(phi), math.cos(phi), 0.0]
        for pol in ([0.0, 0.0, 1.0], hh):  # VV, HH
            # SF propagation is +Direction; -k gives physical incidence -k.
            waves.append({"Index": idx, "Attributes": [1], "Excitation": idx,
                          "Direction": [-k[0], -k[1], 0.0], "Polarization": pol,
                          "Amplitude": 1.0})
            idx += 1
    cfg = {
        "Problem": {"Type": "Driven", "Verbose": 1, "Output": f"postpro/{case}"},
        "Model": {"Mesh": mesh, "L0": 1.0},
        "Domains": {"Materials": [{"Attributes": [1]}]},
        "Boundaries": {
            "PEC": {"Attributes": [2]},
            "Absorbing": {"Attributes": [1], "Order": 2, "ScatteredField": True},
            "PlaneWave": waves,
            "Postprocessing": {"FarField": {"Attributes": [1],
                                            "ThetaPhis": [[90, 0]],
                                            "Scattered": True, "Monostatic": True}},
        },
        "Solver": {"Order": 2,
                   "Linear": {"Type": "Default", "KSPType": "GMRES", "Tol": 1.0e-6,
                              "MaxIts": 1000, "InitialGuess": True},
                   "Driven": {"Samples": [{"Type": "Point", "Freq": [FREQ_GHZ]}]}},
    }
    (WORKDIR / f"{case}.json").write_text(json.dumps(cfg, indent=1))


# %% [markdown]
# ## Run the resolution study (optional)

# %%
if RUN_PALACE:
    for case, (hmin, hmax) in RESOLUTIONS.items():
        dmax = max(0.3, OUTER_RADIUS - 0.2523744)
        subprocess.run([sys.executable, str(TOOLS / "almond_mesh.py"), f"{case}.msh",
                        str(OUTER_RADIUS), str(hmin), str(hmax), str(dmax), "2"],
                       cwd=WORKDIR, check=True)
        make_config(case, f"{case}.msh")
        subprocess.run([PALACE, "-np", str(NPROC), f"{case}.json"], cwd=WORKDIR,
                       check=True)

# %% [markdown]
# ## Monostatic RCS per resolution
#
# Each excitation writes a monostatic row at the backscatter direction
# (`theta = 90`, `phi = phi_az`); excitations alternate VV and HH.

# %%
def read_monostatic(case):
    path = WORKDIR / "postpro" / case / "farfield-rcs.csv"
    if not path.exists():
        return None
    with open(path) as f:
        reader = csv.reader(f)
        header = [h.strip() for h in next(reader)]
        ie = header.index("exc")
        it = header.index("theta (deg.)")
        ip = header.index("phi (deg.)")
        idb = header.index("RCS (dBsm)")
        rows = [(int(r[ie]), float(r[it]), float(r[ip]), float(r[idb])) for r in reader]
    out = {}
    for i, az in enumerate(AZIMUTHS):
        vv = hh = None
        for exc, th, ph, db in rows:
            dphi = abs((ph - az + 180.0) % 360.0 - 180.0)
            if abs(th - 90.0) < 1e-6 and dphi < 1e-6:
                if exc == 2 * i + 1:
                    vv = db
                elif exc == 2 * i + 2:
                    hh = db
        out[az] = {"VV": vv, "HH": hh}
    return out


results = {case: read_monostatic(case) for case in RESOLUTIONS}
print(f"{'case':>4} {'phi':>5} {'VV dBsm':>9} {'HH dBsm':>9}")
for case, res in results.items():
    if res is None:
        print(f"{case}: no output")
        continue
    for az in AZIMUTHS:
        print(f"{case:>4} {az:5.0f} {res[az]['VV']:9.2f} {res[az]['HH']:9.2f}")

# %% [markdown]
# ## Reference comparison

# %%
reference = None
if REFERENCE_CSV.exists():
    ref = {}
    with open(REFERENCE_CSV) as f:
        for row in csv.DictReader(f):
            ref[float(row["phi_deg"])] = (float(row["VV_dBsm"]), float(row["HH_dBsm"]))
    reference = ref
    print(f"Loaded reference with {len(ref)} azimuths from {REFERENCE_CSV}")
else:
    print(f"No reference CSV at {REFERENCE_CSV}; plotting Palace only")

fig, ax = plt.subplots(figsize=(8.5, 5.5))
colors = {"L1": "#1f77b4", "L2": "#d62728", "L3": "#2ca02c"}
for case, res in results.items():
    if res is None:
        continue
    ax.plot(AZIMUTHS, [res[p]["VV"] for p in AZIMUTHS], "o-", color=colors[case],
            label=f"{case} VV")
    ax.plot(AZIMUTHS, [res[p]["HH"] for p in AZIMUTHS], "s--", color=colors[case],
            label=f"{case} HH")
if reference:
    phis = sorted(reference)
    ax.plot(phis, [reference[p][0] for p in phis], "k-", lw=2.0, label="Reference VV")
    ax.plot(phis, [reference[p][1] for p in phis], "k--", lw=2.0, label="Reference HH")
ax.set_xlabel("azimuth $\\phi$ (deg)")
ax.set_ylabel("monostatic RCS (dBsm)")
ax.set_title(f"NASA PEC almond monostatic RCS at {FREQ_GHZ} GHz")
ax.grid(True, alpha=0.3)
ax.legend(fontsize=8, ncol=2)
fig.tight_layout()
fig.savefig(FIGDIR / f"almond_monostatic_{FREQ_GHZ}GHz.png")
print(f"Wrote {FIGDIR / f'almond_monostatic_{FREQ_GHZ}GHz.png'}")
