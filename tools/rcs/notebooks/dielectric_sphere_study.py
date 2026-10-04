# %% [markdown]
# # Dielectric sphere scattering: Palace scattered-field vs Mie theory
#
# Validates the scattered-field contrast sources with a homogeneous dielectric
# sphere (`a = 0.5 m`), lossless (`eps_r = 4`) and lossy (`eps_r = 4 - 0.1i`,
# via `LossTan = 0.025`), at `ka = 1, 2, 3`. The incident field is the free-space
# plane wave (`Direction = [0,0,-1]`, so the monostatic return is at
# `theta = 0`); the volume contrast sources live in the sphere and the ABC is
# homogeneous.
#
# Reference: `mie_dielectric.py` (Bohren-Huffman series, complex refractive
# index). This file is jupytext percent format.

# %%
import csv
import json
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
sys.path.insert(0, str(TOOLS))
from mie_dielectric import mie_dielectric  # noqa: E402

C0 = 299792458.0
PALACE = os.environ.get("PALACE_BIN", "/home/luis/palace/build/bin/palace")
WORKDIR = Path(os.environ.get("DIEL_WORKDIR", "./dielectric_sphere")).resolve()
FIGDIR = WORKDIR / "figures"
NPROC = int(os.environ.get("RCS_NPROC", "16"))
RUN = os.environ.get("RUN_PALACE", "1") == "1"
RADIUS = 0.5
KA_VALUES = [1.0, 2.0, 3.0]
# (tag, eps_r, tan delta): eps = eps_r (1 - i tan delta), so tan = 0.025 -> 4 - 0.1i.
CASES = [("lossless", 4.0, 0.0), ("lossy", 4.0, 0.025)]
THETAS = list(range(0, 181, 15))
WORKDIR.mkdir(parents=True, exist_ok=True)
FIGDIR.mkdir(parents=True, exist_ok=True)


def case_name(ka, tag):
    return f"ka_{ka:g}_{tag}"


def params(ka):
    f = ka * C0 / (2.0 * np.pi * RADIUS)
    lam = C0 / f
    return {
        "f": f,
        "lam": lam,
        "R": RADIUS + 1.5 * lam,
        "hmin": min(RADIUS / 6.0, lam / 16.0),
        "hmax": lam / 8.0,
        "dmax": min(0.75 * lam, 2.5),
    }


def make_mesh(ka):
    d = WORKDIR / f"ka_{ka:g}"
    d.mkdir(exist_ok=True)
    msh = d / "sphere.msh"
    if msh.exists():
        return msh
    prm = params(ka)
    env = dict(os.environ, GMSH_HIGH_ORDER_OPTIMIZE=os.environ.get(
        "GMSH_HIGH_ORDER_OPTIMIZE", "1"))
    subprocess.run([sys.executable, str(TOOLS / "dielectric_sphere_mesh.py"), "sphere.msh",
                    str(RADIUS), str(prm["R"]), str(prm["hmin"]), str(prm["hmax"]),
                    str(prm["dmax"]), "2"], cwd=d, env=env, check=True)
    return msh


def make_config(ka, tag, eps_r, tand):
    d = WORKDIR / f"ka_{ka:g}"
    prm = params(ka)
    mat = {"Attributes": [2], "Permittivity": eps_r}
    if tand > 0.0:
        mat["LossTan"] = tand
    cfg = {
        "Problem": {"Type": "Driven", "Verbose": 1,
                    "Output": f"postpro/{case_name(ka, tag)}"},
        "Model": {"Mesh": "sphere.msh", "L0": 1.0},
        "Domains": {"Materials": [{"Attributes": [1]}, mat]},
        "Boundaries": {
            "Absorbing": {"Attributes": [1], "Order": 2, "Radius": prm["R"],
                          "ScatteredField": True},
            "PlaneWave": [{"Index": 1, "Attributes": [1], "Excitation": 1,
                           "Direction": [0.0, 0.0, -1.0],
                           "Polarization": [1.0, 0.0, 0.0], "Amplitude": 1.0}],
            "Postprocessing": {"FarField": {"Attributes": [1],
                                            "ThetaPhis": [[t, 0] for t in THETAS],
                                            "Scattered": True, "Monostatic": True}},
        },
        "Solver": {"Order": 2,
                   "Linear": {"Type": "Default", "KSPType": "GMRES", "Tol": 1.0e-6,
                              "MaxIts": 1000, "InitialGuess": True},
                   "Driven": {"Samples": [{"Type": "Point",
                                           "Freq": [prm["f"] / 1e9]}]}},
    }
    (d / f"{case_name(ka, tag)}.json").write_text(json.dumps(cfg, indent=1))


def read_pattern(path):
    rows = list(csv.reader(open(path)))
    hdr = [h.strip() for h in rows[0]]
    it, ip, ir = hdr.index("theta (deg.)"), hdr.index("phi (deg.)"), hdr.index("RCS (m^2)")
    th, sig = [], []
    for r in rows[1:]:
        if abs(float(r[ip])) < 1e-6:
            th.append(float(r[it]))
            sig.append(float(r[ir]))
    th, sig = np.array(th), np.array(sig)
    order = np.argsort(th)
    return th[order], sig[order]


# %% [markdown]
# ## Run the study

# %%
if RUN:
    for ka in KA_VALUES:
        make_mesh(ka)
        print(f"mesh ka={ka:g} ready", flush=True)

results = []
patterns = {}
for ka in KA_VALUES:
    for tag, eps_r, tand in CASES:
        name = case_name(ka, tag)
        d = WORKDIR / f"ka_{ka:g}"
        if RUN:
            make_config(ka, tag, eps_r, tand)
            log = d / f"run_{name}.log"
            with open(log, "w") as fh:
                rc = subprocess.run([PALACE, "-np", str(NPROC), f"{name}.json"], cwd=d,
                                    stdout=fh, stderr=subprocess.STDOUT).returncode
            print(f"{name}: rc={rc}", flush=True)
        csv_path = d / "postpro" / name / "farfield-rcs.csv"
        if not csv_path.exists():
            print(f"{name}: missing output", flush=True)
            continue
        th, sig = read_pattern(csv_path)
        eps = eps_r * (1.0 - 1j * tand)
        et, _ = mie_dielectric(RADIUS, params(ka)["f"], np.radians(th), [0.0],
                               epsilon_r=eps)
        msig = np.abs(et) ** 2
        err = 10 * np.log10(sig) - 10 * np.log10(msig)
        results.append({
            "ka": ka, "case": tag, "eps": eps,
            "mono_err": float(10 * np.log10(sig[0] / msig[0])),
            "rms": float(np.sqrt(np.mean(err**2))),
            "max": float(np.max(np.abs(err))),
        })
        patterns[(ka, tag)] = (th, sig, msig)
        print(f"{name}: mono err {results[-1]['mono_err']:+.2f} dB  "
              f"RMS {results[-1]['rms']:.2f} dB  max {results[-1]['max']:.2f} dB",
              flush=True)

# %% [markdown]
# ## Table

# %%
print(f"{'ka':>4} {'case':>9} {'eps':>16} {'mono err':>9} {'RMS':>6} {'max':>6}")
for r in results:
    print(f"{r['ka']:4.0f} {r['case']:>9} {str(r['eps']):>16} "
          f"{r['mono_err']:9.2f} {r['rms']:6.2f} {r['max']:6.2f}")

# %% [markdown]
# ## Figures

# %%
if patterns:
    fig, axes = plt.subplots(1, len(KA_VALUES), figsize=(5.0 * len(KA_VALUES), 4.2),
                             squeeze=False)
    for ax, ka in zip(axes[0], KA_VALUES):
        for tag, _, _ in CASES:
            if (ka, tag) not in patterns:
                continue
            th, sig, msig = patterns[(ka, tag)]
            style = "-" if tag == "lossless" else "--"
            ax.plot(th, 10 * np.log10(msig), style, color="0.5", lw=1.4,
                    label="Mie" if tag == "lossless" else None)
            ax.plot(th, 10 * np.log10(sig), "o", ms=3,
                    label=f"Palace {tag}" if tag == "lossless" else f"Palace {tag}")
        ax.set_title(f"dielectric sphere, ka = {ka:g}")
        ax.set_xlabel(r"$\theta$ (deg)")
        ax.set_ylabel("RCS (dBsm)")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGDIR / "dielectric_sphere_palace_vs_mie.png")
    print(f"Wrote {FIGDIR / 'dielectric_sphere_palace_vs_mie.png'}")
