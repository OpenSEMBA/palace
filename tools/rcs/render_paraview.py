#!/usr/bin/env python3
"""Offscreen ParaView renderer for Palace scattered-field VTU and RCS pattern VTP.

Run with ParaView's batch Python:

  pvbatch3.12 --force-offscreen-rendering render_paraview.py \
      --input scattered.vtu --output slice.png --field E_scat_mag --log \
      --normal 0 1 0 --origin 0 0 0

  pvbatch3.12 --force-offscreen-rendering render_paraview.py \
      --input pattern.vtp --output pattern.png --field RCS_dB --points

For field VTUs a plane slice is extracted when --normal is given; otherwise the
full domain is shown. For point clouds (RCS patterns) use --points.
"""
import argparse
import sys

from paraview.simple import (GetActiveViewOrCreate, GetColorTransferFunction, Hide,
                             OpenDataFile, Render, SaveScreenshot, Show, Slice)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--field", default="E_scat_mag")
    ap.add_argument("--log", action="store_true")
    ap.add_argument("--normal", nargs=3, type=float, default=None)
    ap.add_argument("--origin", nargs=3, type=float, default=[0.0, 0.0, 0.0])
    ap.add_argument("--points", action="store_true",
                    help="render a point cloud (RCS pattern) with Point Gaussian")
    ap.add_argument("--radius", type=float, default=0.02)
    ap.add_argument("--view", nargs=3, type=float, default=None,
                    help="camera position (default: automatic)")
    ap.add_argument("--zoom", type=float, default=1.0)
    args = ap.parse_args(sys.argv[1:])

    reader = OpenDataFile(args.input)
    view = GetActiveViewOrCreate("RenderView")

    source = reader
    if args.normal is not None and not args.points:
        sl = Slice(Input=reader)
        sl.SliceType = "Plane"
        sl.SliceType.Origin = args.origin
        sl.SliceType.Normal = args.normal
        source = sl

    display = Show(source, view)
    if args.points:
        display.Representation = "Point Gaussian"
        display.GaussianRadius = args.radius
    ColorBy(display, ("POINTS", args.field))
    display.SetScalarBarVisibility(view, True)

    lut = GetColorTransferFunction(args.field)
    if args.log:
        lut.UseLogScale = 1
        lut.MapControlPointsToLogSpace()
        lut.RescaleTransferFunctionToDataRange(True)

    view.ViewSize = [1600, 1200]
    view.OrientationAxesVisibility = 1
    if args.view is not None:
        view.CameraPosition = args.view
        view.CameraFocalPoint = [0.0, 0.0, 0.0]
    view.ResetCamera()
    if args.zoom != 1.0:
        view.CameraParallelScale = view.CameraParallelScale / args.zoom
    Render()
    SaveScreenshot(args.output, view, ImageResolution=[1600, 1200])
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
