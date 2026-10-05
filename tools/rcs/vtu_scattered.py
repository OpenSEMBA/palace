#!/usr/bin/env python3
"""Subtract the analytic incident plane wave from a Palace ParaView field.

Palace writes the *total* electric field on the ParaView visualization points
(``E_real``/``E_imag``, in V/m, coordinates in the input mesh units). For plane
wave scattering it is useful to visualize the scattered field

    E_scat(x) = E_total(x) - E_inc(x),
    E_inc(x)  = E0 * p * exp(-i k k_hat . (x - x0)),   k = 2*pi*f/c,

with the same phasor convention as Palace (exp(+i omega t), so the incident
wave travels along +k_hat).

Palace's VTU files use VTK XML version 2.2, which ``meshio`` cannot read, so
this script uses the ``vtk`` Python package. It accepts a single ``.vtu``, a
parallel ``.pvtu``, or the ``.pvd`` collection written by Palace and writes a
merged single-file VTU with the scattered-field arrays.

Usage:
  vtu_scattered.py INPUT.{vtu,pvtu,pvd} OUTPUT.vtu --direction 0 0 -1 \
      --polarization 1 0 0 --frequency 0.3e9 [--amplitude 1] [--phase 0] \
      [--origin 0 0 0] [--l0 1]
"""
import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import vtk

C0 = 299792458.0


def resolve_input(path):
    """Resolve a .pvd collection to its first dataset file."""
    if str(path).endswith(".pvd"):
        tree = ET.parse(path)
        for ds in tree.iter("DataSet"):
            f = ds.get("file")
            if f:
                return str(Path(path).parent / f)
    return path


def read_dataset(path):
    path = resolve_input(path)
    reader = vtk.vtkXMLGenericDataObjectReader()
    reader.SetFileName(path)
    reader.Update()
    data = reader.GetOutput()
    if data is None or data.GetNumberOfPoints() == 0:
        raise SystemExit(f"Could not read {path}")
    return data


def incident_field(points, direction, polarization, amplitude, phase, frequency, origin,
                   l0):
    k = 2.0 * np.pi * frequency / C0
    khat = np.asarray(direction, dtype=float)
    khat /= np.linalg.norm(khat)
    pol = np.asarray(polarization, dtype=float)
    pol /= np.linalg.norm(pol)
    x = points * l0 - np.asarray(origin, dtype=float)
    e = amplitude * np.exp(1j * (phase - k * (x @ khat)))
    return e[:, None] * pol[None, :]


def array_to_numpy(data, name, n):
    arr = data.GetPointData().GetArray(name)
    if arr is None:
        names = [data.GetPointData().GetArrayName(i)
                 for i in range(data.GetPointData().GetNumberOfArrays())]
        raise SystemExit(f"Array {name} not found; available: {names}")
    return np.array([arr.GetTuple(i) for i in range(n)], dtype=float).reshape(n, -1)


def add_vector(data, name, values):
    arr = vtk.vtkDoubleArray()
    arr.SetName(name)
    arr.SetNumberOfComponents(values.shape[1])
    for row in values:
        arr.InsertNextTuple(row)
    data.GetPointData().AddArray(arr)


def add_scalar(data, name, values):
    arr = vtk.vtkDoubleArray()
    arr.SetName(name)
    arr.SetNumberOfComponents(1)
    for v in values:
        arr.InsertNextValue(float(v))
    data.GetPointData().AddArray(arr)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input")
    ap.add_argument("output")
    ap.add_argument("--direction", nargs=3, type=float, required=True)
    ap.add_argument("--polarization", nargs=3, type=float, required=True)
    ap.add_argument("--amplitude", type=float, default=1.0)
    ap.add_argument("--phase", type=float, default=0.0)
    ap.add_argument("--frequency", type=float, required=True, help="Hz")
    ap.add_argument("--origin", nargs=3, type=float, default=[0.0, 0.0, 0.0])
    ap.add_argument("--l0", type=float, default=1.0,
                    help="mesh length unit in meters (Model.L0)")
    args = ap.parse_args()

    data = read_dataset(args.input)
    n = data.GetNumberOfPoints()
    pts = np.array([data.GetPoint(i) for i in range(n)], dtype=float).reshape(n, -1)
    er = array_to_numpy(data, "E_real", n)[:, :3]
    ei = array_to_numpy(data, "E_imag", n)[:, :3]
    etot = er + 1j * ei
    einc = incident_field(pts, args.direction, args.polarization, args.amplitude,
                          args.phase, args.frequency, args.origin, args.l0)
    esc = etot - einc

    add_vector(data, "E_scat_real", esc.real)
    add_vector(data, "E_scat_imag", esc.imag)
    add_vector(data, "E_inc_real", einc.real)
    add_vector(data, "E_inc_imag", einc.imag)
    add_scalar(data, "E_total_mag", np.linalg.norm(etot, axis=1))
    add_scalar(data, "E_scat_mag", np.linalg.norm(esc, axis=1))
    add_scalar(data, "E_inc_mag", np.linalg.norm(einc, axis=1))

    writer = vtk.vtkXMLUnstructuredGridWriter()
    writer.SetFileName(args.output)
    writer.SetInputData(data)
    writer.SetDataModeToBinary()
    writer.Write()
    print(f"Wrote {args.output}: {n} points; "
          f"|E_scat| max = {np.abs(esc).max():.4e} V/m, "
          f"|E_total| max = {np.abs(etot).max():.4e} V/m")

    out = Path(args.output)
    pvd = out.with_suffix(".pvd")
    root = ET.Element("VTKFile", type="Collection", version="0.1",
                      byte_order="LittleEndian")
    coll = ET.SubElement(root, "Collection")
    ET.SubElement(coll, "DataSet", timestep="0", group="", part="0", file=out.name)
    ET.ElementTree(root).write(pvd, xml_declaration=True, encoding="utf-8")
    print(f"Wrote {pvd}")


if __name__ == "__main__":
    main()
