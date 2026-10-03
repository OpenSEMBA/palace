#!/usr/bin/env python3
"""Generate a PEC sphere scattering mesh (sphere inside an outer sphere) with Gmsh.

A spherical outer boundary is used instead of a box: the incident wave impinges
close to normal incidence everywhere, which minimizes the reflection of the
absorbing boundary condition. The inner surface is physical tag 2 (PEC) and the
outer spherical boundary is physical tag 1 (absorbing).

Usage: sphere_mesh.py OUTPUT.msh [RADIUS] [OUTER_RADIUS] [HMIN] [HMAX]
"""
import sys
import gmsh

out = sys.argv[1] if len(sys.argv) > 1 else "sphere.msh"
a = float(sys.argv[2]) if len(sys.argv) > 2 else 0.159154943
R = float(sys.argv[3]) if len(sys.argv) > 3 else 0.6
hmin = float(sys.argv[4]) if len(sys.argv) > 4 else 0.035
hmax = float(sys.argv[5]) if len(sys.argv) > 5 else 0.15
# Distance over which the mesh coarsens from hmin at the sphere to hmax. Keep
# it modest so the fine region does not fill the whole domain.
dist_max = float(sys.argv[6]) if len(sys.argv) > 6 else max(0.3, min(R - a, 2.0 * a))

gmsh.initialize()
gmsh.model.add("sphere")
occ = gmsh.model.occ
outer = occ.addSphere(0, 0, 0, R)
inner = occ.addSphere(0, 0, 0, a)
occ.cut([(3, outer)], [(3, inner)])
occ.synchronize()


def bbox_extent(dim, tag):
    xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(dim, tag)
    return max(xmax - xmin, ymax - ymin, zmax - zmin)


surfs = gmsh.model.getEntities(2)
inner_s = [tag for _, tag in surfs if bbox_extent(2, tag) < 2.1 * a]
outer_s = [tag for _, tag in surfs if tag not in inner_s]
vols = gmsh.model.getEntities(3)
shell = [tag for _, tag in vols if bbox_extent(3, tag) > 2.1 * a]
assert len(inner_s) == 1 and len(outer_s) == 1 and len(shell) == 1, (
    inner_s, outer_s, shell)

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
gmsh.model.mesh.field.setNumber(f2, "DistMax", max(0.3, R - a))
gmsh.model.mesh.field.setAsBackgroundMesh(f2)

gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
gmsh.model.mesh.generate(3)
gmsh.model.mesh.optimize("Netgen")
gmsh.write(out)
print(f"Wrote {out}: {gmsh.model.mesh.getNodes()[0].size} nodes, "
      f"inner {inner_s}, outer {outer_s}")
gmsh.finalize()
