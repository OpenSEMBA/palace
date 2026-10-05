#!/usr/bin/env python3
"""Generate a 2D PEC circular cylinder scattering mesh with Gmsh.

An annulus: inner circle of radius RADIUS (PEC, physical curve tag 2) and outer
circle of radius OUTER_R (absorbing, physical curve tag 1), with a single
domain (physical surface tag 1). Second-order curved elements are the default.

Usage: cylinder_mesh.py OUTPUT.msh [RADIUS] [OUTER_R] [HMIN] [HMAX] [DIST_MAX]
                     [ORDER]
"""
import os
import sys
import gmsh

out = sys.argv[1] if len(sys.argv) > 1 else "cylinder.msh"
a = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
R = float(sys.argv[3]) if len(sys.argv) > 3 else 2.0
hmin = float(sys.argv[4]) if len(sys.argv) > 4 else 0.05
hmax = float(sys.argv[5]) if len(sys.argv) > 5 else 0.2
dist_max = float(sys.argv[6]) if len(sys.argv) > 6 else max(0.3, R - a)
order = int(sys.argv[7]) if len(sys.argv) > 7 else 2
ho_opt = int(os.environ.get("GMSH_HIGH_ORDER_OPTIMIZE", "2"))

gmsh.initialize()
gmsh.model.add("cylinder")
occ = gmsh.model.occ
outer = occ.addDisk(0, 0, 0, R, R)
inner = occ.addDisk(0, 0, 0, a, a)
occ.cut([(2, outer)], [(2, inner)])
occ.synchronize()


def bbox_extent(dim, tag):
    b = gmsh.model.getBoundingBox(dim, tag)
    return max(b[i + 3] - b[i] for i in range(3))


curves = gmsh.model.getEntities(1)
inner_c = [tag for _, tag in curves if bbox_extent(1, tag) < R]
outer_c = [tag for _, tag in curves if tag not in inner_c]
vols = gmsh.model.getEntities(2)
assert inner_c and outer_c and vols, (inner_c, outer_c, vols)

gmsh.model.addPhysicalGroup(2, [tag for _, tag in vols], 1)
gmsh.model.addPhysicalGroup(1, outer_c, 1)
gmsh.model.addPhysicalGroup(1, inner_c, 2)

f1 = gmsh.model.mesh.field.add("Distance")
gmsh.model.mesh.field.setNumbers(f1, "CurvesList", inner_c)
gmsh.model.mesh.field.setNumber(f1, "Sampling", 400)
f2 = gmsh.model.mesh.field.add("Threshold")
gmsh.model.mesh.field.setNumber(f2, "InField", f1)
gmsh.model.mesh.field.setNumber(f2, "SizeMin", hmin)
gmsh.model.mesh.field.setNumber(f2, "SizeMax", hmax)
gmsh.model.mesh.field.setNumber(f2, "DistMin", 0.0)
gmsh.model.mesh.field.setNumber(f2, "DistMax", dist_max)
gmsh.model.mesh.field.setAsBackgroundMesh(f2)

gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
gmsh.option.setNumber("Mesh.ElementOrder", order)
if order > 1:
    gmsh.option.setNumber("Mesh.HighOrderOptimize", ho_opt)
gmsh.model.mesh.generate(2)
if order == 1:
    gmsh.model.mesh.optimize("Netgen")
gmsh.write(out)
print(f"Wrote {out}: {gmsh.model.mesh.getNodes()[0].size} nodes, "
      f"inner {len(inner_c)}, outer {len(outer_c)}, order {order}")
gmsh.finalize()
