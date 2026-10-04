# %% [markdown]
# # PEC sphere monostatic RCS: Palace vs Mie theory
#
# Focused validation of Palace plane wave excitation for a 1 m radius PEC
# sphere at several frequencies, comparing the monostatic radar cross section
# normalized by the projected area, `sigma/(pi a^2)`, with the Mie series.
#
# Convention: `Direction = [0,0,1]` specifies the incidence direction; under the
# incident phasor convention consistent with Palace's Stratton-Chu far-field
# kernel, the wave travels along `-Direction`, so the monostatic (backscatter)
# direction is `theta = 0` in Palace's far-field angle convention. The Mie
# reference `monostatic_normalized(ka)` uses the same backscatter convention.
#
# Run with `RUN_PALACE=1` to generate the meshes and run the cases; otherwise
# existing outputs are parsed. This file is jupytext percent format.

# %%
import csv
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
from mie_pec import monostatic_normalized  # noqa: E402

C0 = 299792458.0
PALACE = os.environ.get("PALACE_BIN", "/home/luis/palace/build/bin/palace")
WORKDIR = Path(os.environ.get("RCS_WORKDIR", "./mono_study")).resolve()
FIGDIR = WORKDIR / "figures"
RUN_PALACE = os.environ.get("RUN_PALACE", "0") == "1"
NPROC = int(os.environ.get("RCS_NPROC", "16"))
RADIUS = 1.0
KA_VALUES = [0.5, 1.0, 2.0, 3.0]
# Spherical absorbing boundary radius (a = 1 m, so ~1.2 wavelengths at ka = 1). The
# scattered-field formulation keeps the incident wave off the absorbing boundary, so a
# modest clearance is sufficient.
R_BOUNDARY = 7.283185307
WORKDIR.mkdir(parents=True, exist_ok=True)
FIGDIR.mkdir(parents=True, exist_ok=True)


def make_config(case, f_hz):
    import json
    cfg = {
        "Problem": {"Type": "Driven", "Verbose": 1, "Output": f"postpro/{case}"},
        "Model": {"Mesh": "sphere.msh", "L0": 1.0},
        "Domains": {"Materials": [{"Attributes": [1]}]},
        "Boundaries": {
            "PEC": {"Attributes": [2]},
            # Scattered-field formulation: homogeneous ABC, incident wave imposed on the
            # PEC sphere; the spherical boundary activates the Bayliss-Turkel curvature
            # term 2/R.
            "Absorbing": {"Attributes": [1], "Order": 2, "Radius": R_BOUNDARY,
                          "ScatteredField": True},
            # The phasor exp(-i k·x) propagates along +Direction; -z matches the Mie
            # reference incidence and puts the monostatic return at theta = 0.
            "PlaneWave": [{"Index": 1, "Attributes": [1], "Excitation": 1,
                           "Direction": [0.0, 0.0, -1.0],
                           "Polarization": [1.0, 0.0, 0.0], "Amplitude": 1.0}],
            "Postprocessing": {"FarField": {"Attributes": [1],
                                            "ThetaPhis": [[t, 0] for t in range(0, 181, 15)],
                                            "Scattered": True, "Monostatic": True}},
        },
        "Solver": {"Order": 2, "Driven": {"Samples": [{"Type": "Point",
                                                       "Freq": [f_hz / 1e9]}]}},
    }
    (WORKDIR / case / f"{case}.json").write_text(json.dumps(cfg, indent=1))


# %% [markdown]
# ## Run the study (optional)

# %%
if RUN_PALACE:
    for ka in KA_VALUES:
        case = f"ka_{ka}"
        d = WORKDIR / case
        d.mkdir(exist_ok=True)
        f = ka * C0 / (2.0 * np.pi * RADIUS)
        lam = C0 / f
        # Spherical outer boundary with the Bayliss-Turkel curvature term (Radius in the
        # config) and second-order curved elements.
        R = R_BOUNDARY
        hmin = min(0.167, lam / 20.0)
        hmax = min(0.63, lam / 10.0)
        dmax = min(2.5, lam)
        subprocess.run([sys.executable, str(TOOLS / "sphere_mesh.py"), "sphere.msh",
                        str(RADIUS), str(R), str(hmin), str(hmax), str(dmax), "2",
                        "sphere"], cwd=d, check=True)
        make_config(case, f)
        subprocess.run([PALACE, "-np", str(NPROC), f"{case}.json"], cwd=d, check=True)

# %% [markdown]
# ## Extract the monostatic values

# %%
def read_pattern(case):
    path = WORKDIR / case / "postpro" / case / "farfield-rcs.csv"
    if not path.exists():
        return None
    with open(path) as fh:
        rows = list(csv.reader(fh))
    header = [h.strip() for h in rows[0]]
    it = header.index("theta (deg.)")
    ip = header.index("phi (deg.)")
    ir = header.index("RCS (m^2)")
    th, sig = [], []
    for row in rows[1:]:
        if abs(float(row[ip])) < 1e-6:
            th.append(float(row[it]))
            sig.append(float(row[ir]))
    th = np.array(th)
    sig = np.array(sig)
    order = np.argsort(th)
    return th[order], sig[order]


results = []
for ka in KA_VALUES:
    case = f"ka_{ka}"
    pat = read_pattern(case)
    if pat is None:
        print(f"ka={ka}: no output")
        continue
    th, sig = pat
    # Direction = -z matches the Mie reference incidence, so Palace(theta) compares
    # directly with Mie(theta); the monostatic return is at theta = 0.
    f_hz = ka * C0 / (2.0 * np.pi * RADIUS)
    et, ep = mie_pec(RADIUS, f_hz, np.radians(th), [0.0])
    msig = np.abs(et) ** 2
    err_db = 10 * np.log10(sig) - 10 * np.log10(msig)
    sigma = sig[int(np.argmin(np.abs(th)))]
    mie = float(monostatic_normalized([ka])[0]) * np.pi * RADIUS**2
    rms = float(np.sqrt(np.mean(err_db**2)))
    mx = float(np.max(np.abs(err_db)))
    results.append((ka, sigma, mie, 10 * np.log10(sigma / mie), rms, mx))
    print(f"ka={ka:4.1f}  monostatic Palace {10*np.log10(sigma):8.3f} dBsm  "
          f"Mie {10*np.log10(mie):8.3f} dBsm  err {results[-1][3]:+5.2f} dB  |  "
          f"pattern RMS {rms:.2f} dB, max {mx:.2f} dB")

# %% [markdown]
# ## Mie curve and Palace points

# %%
ka_curve = np.logspace(np.log10(0.01), np.log10(30.0), 1500)
mie_curve = monostatic_normalized(ka_curve)

fig, ax = plt.subplots(figsize=(8.5, 5.5))
ax.loglog(ka_curve, mie_curve, color="#1f77b4", lw=1.8, label="Mie theory")
if results:
    rms_mean = float(np.mean([r[4] for r in results]))
    ax.loglog([r[0] for r in results], [r[1] / (np.pi * RADIUS**2) for r in results],
              "o", ms=7, color="#d62728",
              label=f"Palace scattered-field (pattern RMS {rms_mean:.2f} dB)")
ax.axhline(1.0, color="0.4", lw=1.0, ls=":")
ax.set_xlim(0.1, 10.0)
ax.set_ylim(1.0e-2, 10.0)
ax.set_xlabel("circumference / wavelength  ($ka$)")
ax.set_ylabel(r"monostatic RCS / projected area  ($\sigma/(\pi a^2)$)")
ax.set_title("Monostatic RCS of a 1 m PEC sphere: Palace vs Mie theory")
ax.grid(True, which="both", alpha=0.3)
ax.legend(loc="upper right", fontsize=9)
fig.tight_layout()
fig.savefig(FIGDIR / "sphere_monostatic_palace_vs_mie.png")
print(f"Wrote {FIGDIR / 'sphere_monostatic_palace_vs_mie.png'}")

# %%
# Error table for the report.
if results:
    print(f"{'ka':>5} {'Palace (dBsm)':>14} {'Mie (dBsm)':>12} {'mono err':>9} "
          f"{'RMS':>7} {'max':>7}")
    for ka, sigma, mie, err, rms, mx in results:
        print(f"{ka:5.1f} {10*np.log10(sigma):14.3f} {10*np.log10(mie):12.3f} "
              f"{err:9.2f} {rms:7.2f} {mx:7.2f}")
