#!/usr/bin/env python3
"""Generate a PEC NASA almond scattering mesh (almond inside a box) with Gmsh.

The almond geometry follows the EMCC/NASA benchmark definition with the
corrected constants (the published expressions in some application notes use
0.13333 for the tip half-width and 1.91115 for the tail thickness factor, which
introduce a discontinuity at x=0; the correct values are 0.193333 and 1.61115,
giving the published 3.84 in width and 1.28 in thickness and a smooth surface).

The cross sections are ellipses with a 3:1 width-to-thickness ratio, so the
solid is a body of revolution about the x axis scaled by 1/3 in z. The mesh is
generated in the axisymmetric geometry and all nodes are then scaled in z, which
also flattens the outer box (its z half-size becomes BOX_HALF/3).

Usage: almond_mesh.py OUTPUT.msh [BOX_HALF] [HMIN] [HMAX]
"""
import sys
import numpy as np
import gmsh

out = sys.argv[1] if len(sys.argv) > 1 else "almond.msh"
box_half = float(sys.argv[2]) if len(sys.argv) > 2 else 0.6
hmin = float(sys.argv[3]) if len(sys.argv) > 3 else 0.012
hmax = float(sys.argv[4]) if len(sys.argv) > 4 else 0.06

# Almond dimensions [m].
d = 9.936 * 0.0254  # total length = 0.2523744 m
t_tip = -0.416667
t_tail = 0.583333


def profile(t):
    """Half-width r(t) of the almond in the (x, y) plane; z = r/3."""
    if t < 0.0:
        return 0.193333 * d * np.sqrt(max(0.0, 1.0 - (t / 0.416667) ** 2))
    return 4.83345 * d * (np.sqrt(max(0.0, 1.0 - (t / 2.08335) ** 2)) - 0.96)


# Sample the profile with clustering near the tip and tail.
#
# NOTE: the almond has sharp tips where the surface has a singular point. The
# current generator produces a valid geometry but the volume mesh quality near
# the tips is not yet sufficient for MFEM (degenerate/sliver elements). Until
# the tips are capped with conformal disks, use a benchmark mesh (for example
# the EMCC/NASA almond meshes) for quantitative studies. The geometry equations
# and dimensions below are verified against the benchmark.
t = np.concatenate(
    [
        t_tip + (0.0 - t_tip) * np.linspace(0.0, 1.0, 40) ** 2,
        0.0 + t_tail * np.linspace(0.0, 1.0, 40) ** 2,
    ]
)
keep = np.concatenate([[True], np.diff(t) > 1e-12])
t = t[keep]
x = d * t
r = np.array([profile(ti) for ti in t])

gmsh.initialize()
gmsh.model.add("almond")
occ = gmsh.model.occ

# Revolve the half-profile about the x axis to obtain the surface of revolution.
pts = [occ.addPoint(xi, ri, 0.0, hmin) for xi, ri in zip(x, r)]
spline = occ.addSpline(pts)
out_tags = occ.revolve([(1, spline)], 0, 0, 0, 1, 0, 0, 2 * np.pi)
shell = sorted({tag for dim, tag in out_tags if dim == 2})
loop = occ.addSurfaceLoop(shell)
almond = occ.addVolume([loop])
occ.synchronize()

box = occ.addBox(-box_half, -box_half, -box_half, 2 * box_half, 2 * box_half,
                 2 * box_half)
occ.cut([(3, box)], [(3, almond)])
occ.synchronize()


def bbox_extent(dim, tag):
    xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(dim, tag)
    return max(xmax - xmin, ymax - ymin, zmax - zmin)


almond_len = x[-1] - x[0]
surfs = gmsh.model.getEntities(2)
inner = [tag for _, tag in surfs if bbox_extent(2, tag) < 1.5 * almond_len]
outer = [tag for _, tag in surfs if tag not in inner]
vols = gmsh.model.getEntities(3)
shell_vols = [tag for _, tag in vols if bbox_extent(3, tag) > 1.5 * almond_len]
assert inner and shell_vols, (inner, shell_vols, vols)

gmsh.model.addPhysicalGroup(3, shell_vols, 1)
gmsh.model.addPhysicalGroup(2, outer, 1)
gmsh.model.addPhysicalGroup(2, inner, 2)

f1 = gmsh.model.mesh.field.add("Distance")
gmsh.model.mesh.field.setNumbers(f1, "SurfacesList", inner)
gmsh.model.mesh.field.setNumber(f1, "Sampling", 400)
f2 = gmsh.model.mesh.field.add("Threshold")
gmsh.model.mesh.field.setNumber(f2, "InField", f1)
gmsh.model.mesh.field.setNumber(f2, "SizeMin", hmin)
gmsh.model.mesh.field.setNumber(f2, "SizeMax", hmax)
gmsh.model.mesh.field.setNumber(f2, "DistMin", 0.0)
gmsh.model.mesh.field.setNumber(f2, "DistMax", 0.3)
gmsh.model.mesh.field.setAsBackgroundMesh(f2)

gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
gmsh.option.setNumber("Mesh.Algorithm3D", 1)  # Delaunay
gmsh.model.mesh.generate(3)
gmsh.model.mesh.optimize("Netgen")

# Apply the 1/3 scaling in z to all nodes.
node_tags, coords, _ = gmsh.model.mesh.getNodes()
coords = coords.reshape(-1, 3)
for tag, (xx, yy, zz) in zip(node_tags, coords):
    gmsh.model.mesh.setNode(int(tag), [float(xx), float(yy), float(zz) / 3.0], [])

gmsh.write(out)
print(f"Wrote {out}: length {almond_len:.4f} m, "
      f"z half-size {box_half / 3.0:.4f} m, inner surfaces {inner}")
gmsh.finalize()
