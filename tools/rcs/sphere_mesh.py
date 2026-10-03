#!/usr/bin/env python3
"""Generate a PEC sphere scattering mesh (sphere inside a box) with Gmsh.

Usage: sphere_mesh.py OUTPUT.msh [RADIUS] [BOX_HALF] [HMIN] [HMAX]
"""
import sys
import gmsh

out = sys.argv[1] if len(sys.argv) > 1 else "sphere.msh"
a = float(sys.argv[2]) if len(sys.argv) > 2 else 0.159154943
R = float(sys.argv[3]) if len(sys.argv) > 3 else 0.6
hmin = float(sys.argv[4]) if len(sys.argv) > 4 else 0.035
hmax = float(sys.argv[5]) if len(sys.argv) > 5 else 0.15

gmsh.initialize()
gmsh.model.add("sphere")
box = gmsh.model.occ.addBox(-R, -R, -R, 2 * R, 2 * R, 2 * R)
sph = gmsh.model.occ.addSphere(0, 0, 0, a)
gmsh.model.occ.fragment([(3, box)], [(3, sph)])
gmsh.model.occ.synchronize()

# Identify the inner (sphere) and outer (box) surfaces by their bounding boxes.
def bbox_extent(dim, tag):
    xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(dim, tag)
    return max(xmax - xmin, ymax - ymin, zmax - zmin)

surfs = gmsh.model.getEntities(2)
inner = [t for _, t in surfs if bbox_extent(2, t) < 2.1 * a]
outer = [t for _, t in surfs if t not in inner]
vols = gmsh.model.getEntities(3)
shell = [t for _, t in vols if bbox_extent(3, t) > 2.1 * a]
inner_vol = [t for _, t in vols if t not in shell]
assert len(inner) == 1, inner
assert len(shell) == 1, (vols, shell)
if inner_vol:
    # Remove the sphere volume but keep the hole surface.
    gmsh.model.occ.remove([(3, t) for t in inner_vol], recursive=False)
    gmsh.model.occ.synchronize()

gmsh.model.addPhysicalGroup(3, shell, 1)
gmsh.model.addPhysicalGroup(2, outer, 1)
gmsh.model.addPhysicalGroup(2, inner, 2)

f1 = gmsh.model.mesh.field.add("Distance")
gmsh.model.mesh.field.setNumbers(f1, "SurfacesList", inner)
gmsh.model.mesh.field.setNumber(f1, "Sampling", 300)
f2 = gmsh.model.mesh.field.add("Threshold")
gmsh.model.mesh.field.setNumber(f2, "InField", f1)
gmsh.model.mesh.field.setNumber(f2, "SizeMin", hmin)
gmsh.model.mesh.field.setNumber(f2, "SizeMax", hmax)
gmsh.model.mesh.field.setNumber(f2, "DistMin", 0.0)
gmsh.model.mesh.field.setNumber(f2, "DistMax", 0.45)
gmsh.model.mesh.field.setAsBackgroundMesh(f2)

gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
gmsh.model.mesh.generate(3)
gmsh.write(out)
n = gmsh.model.mesh.getNodes()[0].size
print(f"Wrote {out}: {n} nodes, inner surfaces {inner}, outer {len(outer)} faces")
gmsh.finalize()
