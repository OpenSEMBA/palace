#!/usr/bin/env python3
"""Generate a dielectric sphere scattering mesh with Gmsh.

Two conforming volumes sharing the sphere interface: the outer vacuum shell
(domain physical tag 1) and the inner dielectric ball (domain physical tag 2).
The outer spherical boundary is physical tag 1 (absorbing) and is meant to be
used with ``Boundaries.Absorbing.Radius = OUTER_R``. Second-order curved
elements are the default.

Usage: dielectric_sphere_mesh.py OUTPUT.msh [RADIUS] [OUTER_R] [HMIN] [HMAX]
                             [DIST_MAX] [ORDER]
"""
import os
import sys
import gmsh

out = sys.argv[1] if len(sys.argv) > 1 else "dielectric_sphere.msh"
a = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
R = float(sys.argv[3]) if len(sys.argv) > 3 else 2.0
hmin = float(sys.argv[4]) if len(sys.argv) > 4 else 0.05
hmax = float(sys.argv[5]) if len(sys.argv) > 5 else 0.2
dist_max = float(sys.argv[6]) if len(sys.argv) > 6 else max(0.3, min(R - a, 2.0 * a))
order = int(sys.argv[7]) if len(sys.argv) > 7 else 2
ho_opt = int(os.environ.get("GMSH_HIGH_ORDER_OPTIMIZE", "2"))

gmsh.initialize()
gmsh.model.add("dielectric_sphere")
occ = gmsh.model.occ
outer = occ.addSphere(0, 0, 0, R)
inner = occ.addSphere(0, 0, 0, a)
occ.fragment([(3, outer)], [(3, inner)])
occ.synchronize()


def bbox_extent(dim, tag):
    b = gmsh.model.getBoundingBox(dim, tag)
    return max(b[i + 3] - b[i] for i in range(3))


vols = gmsh.model.getEntities(3)
ball = [tag for _, tag in vols if bbox_extent(3, tag) < R]
shell = [tag for _, tag in vols if tag not in ball]
surfs = gmsh.model.getEntities(2)
outer_s = [tag for _, tag in surfs if bbox_extent(2, tag) > R]
assert ball and shell and outer_s, (ball, shell, outer_s)

gmsh.model.addPhysicalGroup(3, shell, 1)
gmsh.model.addPhysicalGroup(3, ball, 2)
gmsh.model.addPhysicalGroup(2, outer_s, 1)

f1 = gmsh.model.mesh.field.add("Distance")
gmsh.model.mesh.field.setNumbers(f1, "SurfacesList", [tag for _, tag in surfs
                                                      if bbox_extent(2, tag) < R])
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
gmsh.model.mesh.generate(3)
if order == 1:
    gmsh.model.mesh.optimize("Netgen")
gmsh.write(out)
print(f"Wrote {out}: {gmsh.model.mesh.getNodes()[0].size} nodes, "
      f"ball {ball}, shell {shell}, outer {len(outer_s)}, order {order}")
gmsh.finalize()
