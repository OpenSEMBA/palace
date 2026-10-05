#!/usr/bin/env python3
"""Extract the scattered electric field from Palace ParaView output (pvbatch).

Palace writes VTK XML files with version 2.2 and one piece per MPI rank, which
``meshio`` cannot read. Run this script with ParaView's batch interpreter to
merge the pieces, subtract the analytic incident plane wave, and write a clean
single-file VTU (version 1.0) plus a legacy VTK file:

  pvbatch3.12 --force-offscreen-rendering extract_scattered_pv.py \
      --input postpro/case/paraview/driven/driven.pvd \
      --output vis/case_scattered.vtu \
      --direction 0 0 -1 --polarization 1 0 0 --frequency 0.3e9

The output contains E_scat_real, E_scat_imag, E_total_mag and E_scat_mag.
"""
import argparse
import sys

import numpy as np
from paraview import servermanager
from paraview.simple import OpenDataFile

C0 = 299792458.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--direction", nargs=3, type=float, required=True)
    ap.add_argument("--polarization", nargs=3, type=float, required=True)
    ap.add_argument("--amplitude", type=float, default=1.0)
    ap.add_argument("--phase", type=float, default=0.0)
    ap.add_argument("--frequency", type=float, required=True)
    ap.add_argument("--origin", nargs=3, type=float, default=[0.0, 0.0, 0.0])
    args = ap.parse_args(sys.argv[1:])

    src = OpenDataFile(args.input)
    data = servermanager.Fetch(src)
    n = data.GetNumberOfPoints()
    print(f"Loaded {args.input}: {n} points, {data.GetNumberOfCells()} cells")

    def point_array(name):
        arr = data.GetPointData().GetArray(name)
        if arr is None:
            raise SystemExit(f"Array {name} not found; available: "
                             f"{[data.GetPointData().GetArrayName(i) for i in range(data.GetPointData().GetNumberOfArrays())]}")
        a = np.array([arr.GetTuple(i) for i in range(n)], dtype=float)
        return a.reshape(n, -1)

    er = point_array("E_real")
    ei = point_array("E_imag")
    pts = np.array([data.GetPoint(i) for i in range(n)], dtype=float).reshape(n, -1)

    k = 2.0 * np.pi * args.frequency / C0
    khat = np.asarray(args.direction, dtype=float)
    khat /= np.linalg.norm(khat)
    pol = np.asarray(args.polarization, dtype=float)
    pol /= np.linalg.norm(pol)
    x = pts - np.asarray(args.origin, dtype=float)
    e = args.amplitude * np.exp(1j * (args.phase - k * (x @ khat)))
    einc = e[:, None] * pol[None, :]

    etot = er[:, :3] + 1j * ei[:, :3]
    esc = etot - einc

    from vtkmodules.vtkCommonCore import vtkDoubleArray
    from vtkmodules.vtkIOXML import vtkXMLUnstructuredGridWriter

    def add_vector(name, values):
        arr = vtkDoubleArray()
        arr.SetName(name)
        arr.SetNumberOfComponents(values.shape[1])
        for row in values:
            arr.InsertNextTuple(row)
        data.GetPointData().AddArray(arr)

    def add_scalar(name, values):
        arr = vtkDoubleArray()
        arr.SetName(name)
        arr.SetNumberOfComponents(1)
        for v in values:
            arr.InsertNextValue(float(v))
        data.GetPointData().AddArray(arr)

    add_vector("E_scat_real", esc.real)
    add_vector("E_scat_imag", esc.imag)
    add_scalar("E_total_mag", np.linalg.norm(etot, axis=1))
    add_scalar("E_scat_mag", np.linalg.norm(esc, axis=1))

    writer = vtkXMLUnstructuredGridWriter()
    writer.SetFileName(args.output)
    writer.SetInputData(data)
    writer.SetDataModeToBinary()
    writer.Write()
    print(f"Wrote {args.output}; |E_scat| max = {np.abs(esc).max():.4e} V/m")


if __name__ == "__main__":
    main()
