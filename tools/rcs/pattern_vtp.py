#!/usr/bin/env python3
"""Build a ParaView-compatible 3D RCS pattern (.vtp) from Palace far-field CSV.

The radial coordinate of each far-field sample is scaled in dB relative to the
peak,

    r = R_MIN + (R_MAX - R_MIN) * clip((sigma_dB - (sigma_max - DYNRANGE)) / DYNRANGE, 0, 1),

so deep nulls are visible while the main lobes keep their shape. The scalars
``RCS`` (m^2), ``RCS_dB`` (dBsm) and ``radius`` are attached to the point cloud
for coloring in ParaView.

Usage:
  pattern_vtp.py farfield-rcs.csv pattern.vtp [--dynrange 40] [--rmin 0.2] [--rmax 1.0]
"""
import argparse
import csv

import numpy as np


def read_rcs_csv(path):
    theta, phi, rcs, rcs_db = [], [], [], []
    with open(path) as f:
        reader = csv.reader(f)
        header = [h.strip() for h in next(reader)]
        it = next(i for i, h in enumerate(header) if h.startswith("theta"))
        ip = next(i for i, h in enumerate(header) if h.startswith("phi"))
        ir = next(i for i, h in enumerate(header) if h.startswith("RCS (m^2)"))
        idb = next(i for i, h in enumerate(header) if h.startswith("RCS (dBsm)"))
        for row in reader:
            theta.append(float(row[it]))
            phi.append(float(row[ip]))
            rcs.append(float(row[ir]))
            rcs_db.append(float(row[idb]))
    return np.radians(theta), np.radians(phi), np.asarray(rcs), np.asarray(rcs_db)


def write_vtp(path, points, point_data):
    """Write a minimal ASCII VTK PolyData file with vertex cells."""
    n = points.shape[0]
    lines = ['<?xml version="1.0"?>',
             '<VTKFile type="PolyData" version="0.1" byte_order="LittleEndian">',
             '<PolyData>',
             f'<Piece NumberOfPoints="{n}" NumberOfVerts="{n}">',
             '<PointData Scalars="RCS_dB">']
    for name, values in point_data.items():
        vals = np.asarray(values).reshape(n, -1)
        lines.append(f'<DataArray type="Float64" Name="{name}" '
                     f'NumberOfComponents="{vals.shape[1]}" format="ascii">')
        lines.append(" ".join(f"{v:.6e}" for row in vals for v in row))
        lines.append("</DataArray>")
    lines.append("</PointData>")
    lines.append("<Points>")
    lines.append('<DataArray type="Float64" NumberOfComponents="3" format="ascii">')
    lines.append(" ".join(f"{v:.6e}" for row in points for v in row))
    lines.append("</DataArray>")
    lines.append("</Points>")
    lines.append("<Verts>")
    lines.append('<DataArray type="Int64" Name="connectivity" format="ascii">')
    lines.append(" ".join(str(i) for i in range(n)))
    lines.append("</DataArray>")
    lines.append('<DataArray type="Int64" Name="offsets" format="ascii">')
    lines.append(" ".join(str(i + 1) for i in range(n)))
    lines.append("</DataArray>")
    lines.append("</Verts>")
    lines.append("</Piece>")
    lines.append("</PolyData>")
    lines.append("</VTKFile>")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv")
    ap.add_argument("output")
    ap.add_argument("--dynrange", type=float, default=40.0, help="dB dynamic range")
    ap.add_argument("--rmin", type=float, default=0.2)
    ap.add_argument("--rmax", type=float, default=1.0)
    args = ap.parse_args()

    theta, phi, rcs, rcs_db = read_rcs_csv(args.csv)
    peak = rcs_db.max()
    t = np.clip((rcs_db - (peak - args.dynrange)) / args.dynrange, 0.0, 1.0)
    r = args.rmin + (args.rmax - args.rmin) * t

    rhat = np.column_stack([
        np.sin(theta) * np.cos(phi),
        np.sin(theta) * np.sin(phi),
        np.cos(theta),
    ])
    points = r[:, None] * rhat
    write_vtp(args.output, points,
              {"RCS": rcs, "RCS_dB": rcs_db, "radius": r})
    print(f"Wrote {args.output}: {len(points)} samples, peak {peak:.2f} dBsm, "
          f"min {rcs_db.min():.2f} dBsm")


if __name__ == "__main__":
    main()
