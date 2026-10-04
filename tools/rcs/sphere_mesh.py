#!/usr/bin/env python3
"""Generate a PEC sphere scattering mesh with Gmsh.

The inner surface is physical tag 2 (PEC) and the outer boundary is physical
tag 1 (absorbing). Two outer boundary shapes are available:

  box    -- flat faces; no curvature term in the ABC. At normal incidence the
            ABC is accurate, but grazing incidence on the side walls reflects.
  sphere -- spherical boundary; use Boundaries.Absorbing.Radius = BOX_HALF in
            the Palace config to activate the Bayliss-Turkel curvature term
            2/R in the second-order ABC.

Meshes are generated with second-order (curved) elements by default; pass
ORDER = 1 for the straight-sided comparison arm.

Usage: sphere_mesh.py OUTPUT.msh [RADIUS] [BOX_HALF] [HMIN] [HMAX] [DIST_MAX]
                      [ORDER] [OUTER]
"""
import sys
import gmsh

out = sys.argv[1] if len(sys.argv) > 1 else "sphere.msh"
a = float(sys.argv[2]) if len(sys.argv) > 2 else 0.159154943
R = float(sys.argv[3]) if len(sys.argv) > 3 else 0.6
hmin = float(sys.argv[4]) if len(sys.argv) > 4 else 0.035
hmax = float(sys.argv[5]) if len(sys.argv) > 5 else 0.15
dist_max = float(sys.argv[6]) if len(sys.argv) > 6 else max(0.3, min(R - a, 2.0 * a))
order = int(sys.argv[7]) if len(sys.argv) > 7 else 2
outer = sys.argv[8] if len(sys.argv) > 8 else "box"

gmsh.initialize()
gmsh.model.add("sphere")
occ = gmsh.model.occ
if outer == "sphere":
    outer_shape = occ.addSphere(0, 0, 0, R)
elif outer == "box":
    outer_shape = occ.addBox(-R, -R, -R, 2 * R, 2 * R, 2 * R)
else:
    raise SystemExit(f"Unknown outer boundary '{outer}' (use 'box' or 'sphere')")
inner = occ.addSphere(0, 0, 0, a)
occ.cut([(3, outer_shape)], [(3, inner)])
occ.synchronize()


def bbox_extent(dim, tag):
    b = gmsh.model.getBoundingBox(dim, tag)
    return max(b[i + 3] - b[i] for i in range(3))


surfs = gmsh.model.getEntities(2)
inner_s = [tag for _, tag in surfs if bbox_extent(2, tag) < 2.1 * a]
outer_s = [tag for _, tag in surfs if tag not in inner_s]
vols = gmsh.model.getEntities(3)
shell = [tag for _, tag in vols if bbox_extent(3, tag) > 2.1 * a]
assert inner_s and outer_s and shell, (inner_s, outer_s, shell)

gmsh.model.addPhysicalGroup(3, shell, 1)
gmsh.model.addPhysicalGroup(2, outer_s, 1)
gmsh.model.addPhysicalGroup(2, inner_s, 2)

f1 = gmsh.model.mesh.field.add("Distance")
gmsh.model.mesh.field.setNumbers(f1, "SurfacesList", inner_s)
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
    # Elastic + optimization of the high-order nodes; keeps curved elements valid.
    gmsh.option.setNumber("Mesh.HighOrderOptimize", 2)
gmsh.model.mesh.generate(3)
if order == 1:
    gmsh.model.mesh.optimize("Netgen")
gmsh.write(out)
print(f"Wrote {out}: {gmsh.model.mesh.getNodes()[0].size} nodes, "
      f"inner {len(inner_s)}, outer {len(outer_s)}, order {order}, outer {outer}")
gmsh.finalize()
