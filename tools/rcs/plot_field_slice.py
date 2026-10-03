#!/usr/bin/env python3
"""Plot one or more field slices through a Palace/ParaView VTU with matplotlib.

Reads a ``.vtu`` (e.g. the output of ``vtu_scattered.py``, which contains
``E_total_mag``, ``E_scat_mag`` and ``E_inc_mag``), cuts it with a plane, and
saves a multi-panel PNG colored by the requested point-data scalars. Useful in
environments without a working ParaView Python interpreter.

Usage:
  plot_field_slice.py INPUT.vtu OUTPUT.png --fields E_total_mag E_scat_mag \
      [E_inc_mag] [--normal 0 1 0] [--origin 0 0 0] [--db] [--log] \
      [--shared-scale] [--title "..."]
"""
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.colors import LogNorm
import vtk


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input")
    ap.add_argument("output")
    ap.add_argument("--fields", nargs="+", default=["E_total_mag", "E_scat_mag"])
    ap.add_argument("--normal", nargs=3, type=float, default=[0.0, 1.0, 0.0])
    ap.add_argument("--origin", nargs=3, type=float, default=[0.0, 0.0, 0.0])
    ap.add_argument("--db", action="store_true",
                    help="plot 20*log10(field) in dB relative to 1 V/m")
    ap.add_argument("--log", action="store_true", help="logarithmic color scale")
    ap.add_argument("--shared-scale", action="store_true",
                    help="use one color range for all panels")
    ap.add_argument("--clim", nargs=2, type=float, default=None)
    ap.add_argument("--title", default=None)
    args = ap.parse_args()

    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(args.input)
    reader.Update()
    plane = vtk.vtkPlane()
    plane.SetOrigin(args.origin)
    plane.SetNormal(args.normal)
    cutter = vtk.vtkCutter()
    cutter.SetInputData(reader.GetOutput())
    cutter.SetCutFunction(plane)
    cutter.Update()
    poly = cutter.GetOutput()
    n = poly.GetNumberOfPoints()
    pts = np.array([poly.GetPoint(i) for i in range(n)], dtype=float).reshape(n, 3)

    def point_array(name):
        arr = poly.GetPointData().GetArray(name)
        if arr is None:
            names = [poly.GetPointData().GetArrayName(i)
                     for i in range(poly.GetPointData().GetNumberOfArrays())]
            raise SystemExit(f"Field {name} not found; available: {names}")
        return np.array([arr.GetTuple1(i) for i in range(n)], dtype=float)

    normal = np.asarray(args.normal, dtype=float)
    normal /= np.linalg.norm(normal)
    axis = int(np.argmax(np.abs(normal)))
    keep = [i for i in range(3) if i != axis]
    x, y = pts[:, keep[0]], pts[:, keep[1]]
    tris = np.array([[poly.GetCell(i).GetPointId(j) for j in range(3)]
                     for i in range(poly.GetNumberOfCells())])
    triang = mtri.Triangulation(x, y, tris)

    fields = [(name, point_array(name)) for name in args.fields]
    if args.db:
        fields = [(name, 20.0 * np.log10(np.maximum(v, 1e-12))) for name, v in fields]

    if args.clim is not None:
        vmin, vmax = args.clim
    elif args.shared_scale:
        vmin = min(v.min() for _, v in fields)
        vmax = max(v.max() for _, v in fields)
    else:
        vmin = vmax = None

    npan = len(fields)
    fig, axes = plt.subplots(1, npan, figsize=(6.0 * npan, 5.2), squeeze=False)
    for ax, (name, vals) in zip(axes[0], fields):
        kw = {}
        if vmin is not None and vmax is not None:
            if args.log and not args.db:
                kw["norm"] = LogNorm(vmin=max(vmin, 1e-12), vmax=vmax)
            else:
                kw["vmin"], kw["vmax"] = vmin, vmax
        tpc = ax.tripcolor(triang, vals, shading="gouraud", cmap="jet", **kw)
        fig.colorbar(tpc, ax=ax, shrink=0.7, label=name + (" (dB)" if args.db else ""))
        ax.set_aspect("equal")
        ax.set_xlabel("xyz"[keep[0]])
        ax.set_ylabel("xyz"[keep[1]])
        ax.set_title(name)
    if args.title:
        fig.suptitle(args.title)
    fig.tight_layout()
    fig.savefig(args.output, dpi=200)
    print(f"Wrote {args.output}: {n} slice points, fields {[f[0] for f in fields]}")


if __name__ == "__main__":
    main()
