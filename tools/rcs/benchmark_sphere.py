# %% [markdown]
# # PEC sphere benchmark: accuracy vs DOFs and CPU time
#
# Scattered-field plane wave scattering from a 1 m PEC sphere over ten
# frequencies (`ka = 0.5 ... 5.0`) and three finite element orders (`p = 1, 2,
# 3`). For each frequency one second-order curved mesh is generated (shared by
# all orders) with a spherical absorbing boundary scaled with the wavelength,
# `R = a + 1.5*lambda`, `hmax = lambda/8`, `hmin = min(a/6, lambda/16)`,
# `dmax = min(0.75*lambda, 2.5)`. The Bayliss-Turkel curvature term is active
# (`Radius = R`) and the scattered-field formulation is used.
#
# Metrics recorded per run: monostatic error, pattern RMS/max error against the
# Mie series, ND true DOFs, GMRES iterations and residual, wall time and its
# breakdown, and peak memory.
#
# This file is jupytext percent format. Set `RUN_PALACE=0` to only parse
# existing outputs. `BENCH_KA_VALUES` and `BENCH_ORDERS` override the sweep.

# %%
import csv
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

matplotlib.rcParams.update({"figure.dpi": 110, "font.size": 10})

TOOLS = Path(__file__).resolve().parent if "__file__" in globals() else Path(
    "/home/luis/palace/tools/rcs")
sys.path.insert(0, str(TOOLS))
from mie_pec import mie_pec  # noqa: E402

C0 = 299792458.0
PALACE = os.environ.get("PALACE_BIN", "/home/luis/palace/build/bin/palace")
WORKDIR = Path(os.environ.get("BENCH_WORKDIR", "./bench_sphere")).resolve()
FIGDIR = WORKDIR / "figures"
NPROC = int(os.environ.get("RCS_NPROC", "16"))
RUN = os.environ.get("RUN_PALACE", "1") == "1"
KA_VALUES = [float(x) for x in os.environ.get(
    "BENCH_KA_VALUES", "0.5,1,1.5,2,2.5,3,3.5,4,4.5,5").split(",")]
ORDERS = [int(x) for x in os.environ.get("BENCH_ORDERS", "1,2,3").split(",")]
RADIUS = 1.0
CLEARANCE = 1.5          # boundary clearance in wavelengths
THETAS = list(range(0, 181, 15))
WORKDIR.mkdir(parents=True, exist_ok=True)
FIGDIR.mkdir(parents=True, exist_ok=True)


def case_name(ka, p):
    return f"ka_{ka:g}_p{p}"


def mesh_params(ka):
    f = ka * C0 / (2.0 * np.pi * RADIUS)
    lam = C0 / f
    return {
        "f": f,
        "lam": lam,
        "R": RADIUS + CLEARANCE * lam,
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
    prm = mesh_params(ka)
    env = dict(os.environ, GMSH_HIGH_ORDER_OPTIMIZE=os.environ.get(
        "GMSH_HIGH_ORDER_OPTIMIZE", "1"))
    subprocess.run([sys.executable, str(TOOLS / "sphere_mesh.py"), "sphere.msh",
                    str(RADIUS), str(prm["R"]), str(prm["hmin"]), str(prm["hmax"]),
                    str(prm["dmax"]), "2", "sphere"], cwd=d, env=env, check=True)
    return msh


def make_config(ka, p):
    d = WORKDIR / f"ka_{ka:g}"
    prm = mesh_params(ka)
    cfg = {
        "Problem": {"Type": "Driven", "Verbose": 1,
                    "Output": f"postpro/{case_name(ka, p)}"},
        "Model": {"Mesh": "sphere.msh", "L0": 1.0},
        "Domains": {"Materials": [{"Attributes": [1]}]},
        "Boundaries": {
            "PEC": {"Attributes": [2]},
            "Absorbing": {"Attributes": [1], "Order": 2, "Radius": prm["R"],
                          "ScatteredField": True},
            "PlaneWave": [{"Index": 1, "Attributes": [1], "Excitation": 1,
                           "Direction": [0.0, 0.0, -1.0],
                           "Polarization": [1.0, 0.0, 0.0], "Amplitude": 1.0}],
            "Postprocessing": {"FarField": {"Attributes": [1],
                                            "ThetaPhis": [[t, 0] for t in THETAS],
                                            "Scattered": True, "Monostatic": True}},
        },
        "Solver": {"Order": p,
                   "Linear": {"Type": "Default", "KSPType": "GMRES",
                              "Tol": 1.0e-6, "MaxIts": 1000,
                              "InitialGuess": True},
                   "Driven": {"Samples": [{"Type": "Point",
                                           "Freq": [prm["f"] / 1e9]}]}},
    }
    path = d / f"{case_name(ka, p)}.json"
    path.write_text(json.dumps(cfg, indent=1))
    return path


def parse_log(path):
    text = Path(path).read_text(errors="replace")
    out = {}
    m = re.search(r"ND \(p = (\d+)\): (\d+)", text)
    if m:
        out["nd_dofs"] = int(m.group(2))
    m = re.search(r"GMRES solver (converged|did NOT converge) in (\d+) iterations"
                  r"(?: \(avg\. reduction factor: ([\d.eE+-]+)\))?", text)
    if m:
        out["converged"] = m.group(1) == "converged"
        out["gmres_iters"] = int(m.group(2))
        out["gmres_reduction"] = float(m.group(3)) if m.group(3) else np.nan
    m = re.search(r"norm\(Ax-b\)/norm\(b\) = ([\d.eE+-]+)", text)
    out["residual"] = float(m.group(1)) if m else np.nan
    m = re.search(r"^Total\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*$", text, re.M)
    if m:
        out["wall_s"] = float(m.group(2))
    for key in ["Initialization", "Operator Construction", "Linear Solve", "Setup",
                "Preconditioner", "Coarse Solve", "Estimation", "Postprocessing"]:
        m = re.search(rf"^\s*{re.escape(key)}\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*$",
                      text, re.M)
        if m:
            out["t_" + key.lower().replace(" ", "_")] = float(m.group(2))
    m = re.search(r"^Total\s+([\d.]+)([GMK])\s+([\d.]+)([GMK])\s+([\d.]+)([GMK])\s*$",
                  text, re.M)
    if m:
        scale = {"K": 1e-6, "M": 1e-3, "G": 1.0}[m.group(6)]
        out["peak_mem_gb"] = float(m.group(5)) * scale
    return out


def read_pattern(path):
    rows = list(csv.reader(open(path)))
    hdr = [h.strip() for h in rows[0]]
    it, ip, ir = hdr.index("theta (deg.)"), hdr.index("phi (deg.)"), hdr.index("RCS (m^2)")
    th, sig = [], []
    for r in rows[1:]:
        if abs(float(r[ip])) < 1e-6:
            th.append(float(r[it]))
            sig.append(float(r[ir]))
    th = np.array(th)
    sig = np.array(sig)
    order = np.argsort(th)
    return th[order], sig[order]


def errors_vs_mie(ka, th, sig):
    et, ep = mie_pec(RADIUS, mesh_params(ka)["f"], np.radians(th), [0.0])
    msig = np.abs(et) ** 2
    err = 10 * np.log10(sig) - 10 * np.log10(msig)
    return {
        "mono_db": float(10 * np.log10(sig[0])),
        "mono_mie_db": float(10 * np.log10(msig[0])),
        "mono_err_db": float(10 * np.log10(sig[0] / msig[0])),
        "rms_db": float(np.sqrt(np.mean(err**2))),
        "max_db": float(np.max(np.abs(err))),
    }


# %% [markdown]
# ## Run the sweep

# %%
def run_case(ka, p):
    d = WORKDIR / f"ka_{ka:g}"
    log = d / f"run_p{p}.log"
    if RUN:
        cfg = make_config(ka, p)
        t0 = os.times().elapsed
        with open(log, "w") as fh:
            rc = subprocess.run([PALACE, "-np", str(NPROC), cfg.name], cwd=d,
                                stdout=fh, stderr=subprocess.STDOUT).returncode
        print(f"{case_name(ka, p)}: wall {os.times().elapsed - t0:.1f} s rc={rc}",
              flush=True)
    csv_path = d / "postpro" / case_name(ka, p) / "farfield-rcs.csv"
    if not log.exists() or not csv_path.exists():
        return None
    rec = {"ka": ka, "p": p, **mesh_params(ka)}
    rec.update(parse_log(log))
    th, sig = read_pattern(csv_path)
    rec.update(errors_vs_mie(ka, th, sig))
    return rec


if RUN:
    for ka in KA_VALUES:
        make_mesh(ka)
        print(f"mesh ka={ka:g} ready", flush=True)

results = []
for ka in KA_VALUES:
    for p in ORDERS:
        rec = run_case(ka, p)
        if rec:
            results.append(rec)
            print(f"  {case_name(ka, p)}: ND={rec.get('nd_dofs')} "
                  f"RMS={rec['rms_db']:.2f} dB iters={rec.get('gmres_iters')} "
                  f"wall={rec.get('wall_s'):.1f} s", flush=True)

# %% [markdown]
# ## Metrics table

# %%
fields = ["ka", "p", "f", "lam", "R", "nd_dofs", "gmres_iters", "converged",
          "residual", "wall_s", "peak_mem_gb", "mono_db", "mono_mie_db",
          "mono_err_db", "rms_db", "max_db", "t_linear_solve", "t_preconditioner",
          "t_coarse_solve", "t_operator_construction", "t_postprocessing"]
with open(WORKDIR / "benchmark_sphere.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
    w.writeheader()
    for rec in sorted(results, key=lambda r: (r["ka"], r["p"])):
        w.writerow(rec)
print(f"Wrote {WORKDIR / 'benchmark_sphere.csv'} ({len(results)} runs)")

# %% [markdown]
# ## Figures

# %%
def pcolor(p):
    return {1: "#1f77b4", 2: "#d62728", 3: "#2ca02c", 4: "#9467bd"}.get(p, "0.3")


def pmarker(p):
    return {1: "o", 2: "s", 3: "^", 4: "D"}.get(p, "x")


if results:
    # Accuracy vs DOFs and vs wall time.
    for ykey, ylabel, fname in [
        ("nd_dofs", "ND true DOFs", "bench_sphere_accuracy_vs_dofs.png"),
        ("wall_s", "wall time (s)", "bench_sphere_accuracy_vs_time.png"),
    ]:
        fig, ax = plt.subplots(figsize=(8.0, 5.5))
        for ka in KA_VALUES:
            pts = [r for r in results if r["ka"] == ka]
            if not pts:
                continue
            pts.sort(key=lambda r: r["p"])
            ax.plot([r[ykey] for r in pts], [r["rms_db"] for r in pts], "-",
                    color="0.7", lw=0.8, zorder=1)
            for r in pts:
                ax.plot(r[ykey], r["rms_db"], pmarker(r["p"]), ms=7,
                        color=pcolor(r["p"]), zorder=2)
        for p in ORDERS:
            ax.plot([], [], pmarker(p), color=pcolor(p), label=f"p = {p}")
        ax.set_xscale("log")
        ax.set_xlabel(ylabel)
        ax.set_ylabel("pattern RMS error (dB)")
        ax.set_title("1 m PEC sphere: accuracy vs " +
                     ("ND DOFs" if ykey == "nd_dofs" else "wall time"))
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(loc="best", fontsize=9)
        fig.tight_layout()
        fig.savefig(FIGDIR / fname)
        print(f"Wrote {FIGDIR / fname}")

    # DOFs and time vs ka.
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.6))
    for p in ORDERS:
        pts = sorted([r for r in results if r["p"] == p], key=lambda r: r["ka"])
        if not pts:
            continue
        axes[0].semilogy([r["ka"] for r in pts], [r["nd_dofs"] for r in pts],
                         "-" + pmarker(p), color=pcolor(p), label=f"p = {p}")
        axes[1].semilogy([r["ka"] for r in pts], [r["wall_s"] for r in pts],
                         "-" + pmarker(p), color=pcolor(p), label=f"p = {p}")
    axes[0].set_xlabel("$ka$")
    axes[0].set_ylabel("ND true DOFs")
    axes[0].grid(True, which="both", alpha=0.3)
    axes[1].set_xlabel("$ka$")
    axes[1].set_ylabel("wall time (s)")
    axes[1].grid(True, which="both", alpha=0.3)
    axes[1].legend(loc="best", fontsize=9)
    fig.suptitle("PEC sphere benchmark: cost vs frequency")
    fig.tight_layout()
    fig.savefig(FIGDIR / "bench_sphere_dofs_time_vs_ka.png")
    print(f"Wrote {FIGDIR / 'bench_sphere_dofs_time_vs_ka.png'}")
