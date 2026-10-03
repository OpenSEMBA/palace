#!/usr/bin/env python3
"""Generate a PEC NASA almond scattering mesh (almond inside an outer sphere).

The almond geometry follows the EMCC/NASA benchmark definition with the
corrected constants (the published expressions in some application notes use
0.13333 for the tip half-width and 1.91115 for the tail thickness factor, which
introduce a discontinuity at x=0; the correct values are 0.193333 and 1.61115,
giving the published 3.84 in width and 1.28 in thickness and a smooth surface).

Construction: the half-profile is revolved about the x axis and the small
openings left by truncating the sharp tips are capped with conformal disks
built from the tube's boundary edges. The cross sections are ellipses with a
3:1 width-to-thickness ratio, obtained by scaling all mesh nodes by 1/3 in z;
this also flattens the outer spherical boundary into an ellipsoid with z
semi-axis OUTER_RADIUS/3 (still adequate clearance for the absorbing boundary).

Usage: almond_mesh.py OUTPUT.msh [OUTER_RADIUS] [HMIN] [HMAX]
"""
import sys
import numpy as np
import gmsh

out = sys.argv[1] if len(sys.argv) > 1 else "almond.msh"
R = float(sys.argv[2]) if len(sys.argv) > 2 else 1.2
hmin = float(sys.argv[3]) if len(sys.argv) > 3 else 0.01
hmax = float(sys.argv[4]) if len(sys.argv) > 4 else 0.06

d = 9.936 * 0.0254  # total length = 0.2523744 m
t_tip = -0.416667
t_tail = 0.583333
r_min = 0.0005  # tip truncation radius [m]


def profile(t):
    if t < 0.0:
        return 0.193333 * d * np.sqrt(max(0.0, 1.0 - (t / 0.416667) ** 2))
    return 4.83345 * d * (np.sqrt(max(0.0, 1.0 - (t / 2.08335) ** 2)) - 0.96)


# Truncate at the tip radius and sample the profile with clustering.
t_tip_cut = -0.416667 * np.sqrt(1.0 - (r_min / (0.193333 * d)) ** 2)
t_tail_cut = 2.08335 * np.sqrt(1.0 - (0.96 + r_min / (4.83345 * d)) ** 2)
t = np.concatenate([
    t_tip_cut + (0.0 - t_tip_cut) * np.linspace(0.0, 1.0, 40) ** 2,
    0.0 + t_tail_cut * np.linspace(0.0, 1.0, 40) ** 2,
])
keep = np.concatenate([[True], np.diff(t) > 1e-12])
t = t[keep]
x = d * t
r = np.array([profile(ti) for ti in t])

gmsh.initialize()
gmsh.model.add("almond")
occ = gmsh.model.occ

# Revolve the half-profile about x, then cap the truncated openings with
# conformal plane surfaces built from the tube boundary edges.
pts = [occ.addPoint(xi, ri, 0.0, hmin) for xi, ri in zip(x, r)]
spline = occ.addSpline(pts)
out_tags = occ.revolve([(1, spline)], 0, 0, 0, 1, 0, 0, 2 * np.pi)
surface = [tag for dim, tag in out_tags if dim == 2][0]
occ.synchronize()
boundary = gmsh.model.getBoundary([(2, surface)], oriented=False, recursive=False)
edges = sorted(tag for dim, tag in boundary if dim == 1)
caps = [occ.addPlaneSurface([occ.addCurveLoop([edge])]) for edge in edges]
almond = occ.addVolume([occ.addSurfaceLoop([surface] + caps)])
occ.synchronize()

# Outer spherical domain.
outer = occ.addSphere(0, 0, 0, R)
occ.cut([(3, outer)], [(3, almond)])
occ.synchronize()


def bbox_extent(dim, tag):
    b = gmsh.model.getBoundingBox(dim, tag)
    return max(b[i + 3] - b[i] for i in range(3))


# Use only the surfaces bounding the resulting domain volume; remove any
# leftover tool surfaces so they are not written as duplicate boundary elements.
almond_len = x[-1] - x[0]
vols = gmsh.model.getEntities(3)
shell = [tag for _, tag in vols if bbox_extent(3, tag) > 1.5 * almond_len]
assert len(shell) == 1, (vols, shell)
boundary = gmsh.model.getBoundary([(3, shell[0])], oriented=False, recursive=False)
shell_surfs = {tag for dim, tag in boundary if dim == 2}
extra = [tag for _, tag in gmsh.model.getEntities(2) if tag not in shell_surfs]
if extra:
    occ.remove([(2, tag) for tag in extra], recursive=False)
    occ.synchronize()
inner = [tag for tag in sorted(shell_surfs) if bbox_extent(2, tag) < 1.5 * almond_len]
outer_s = [tag for tag in sorted(shell_surfs) if tag not in inner]
assert inner and outer_s, (inner, outer_s, shell_surfs)

gmsh.model.addPhysicalGroup(3, shell, 1)
gmsh.model.addPhysicalGroup(2, outer_s, 1)
gmsh.model.addPhysicalGroup(2, inner, 2)

f1 = gmsh.model.mesh.field.add("Distance")
gmsh.model.mesh.field.setNumbers(f1, "SurfacesList", inner)
gmsh.model.mesh.field.setNumber(f1, "Sampling", 400)
f2 = gmsh.model.mesh.field.add("Threshold")
gmsh.model.mesh.field.setNumber(f2, "InField", f1)
gmsh.model.mesh.field.setNumber(f2, "SizeMin", hmin)
gmsh.model.mesh.field.setNumber(f2, "SizeMax", hmax)
gmsh.model.mesh.field.setNumber(f2, "DistMin", 0.0)
gmsh.model.mesh.field.setNumber(f2, "DistMax", max(0.3, R - almond_len))
gmsh.model.mesh.field.setAsBackgroundMesh(f2)

gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
gmsh.option.setNumber("Mesh.Algorithm3D", 1)
gmsh.model.mesh.generate(3)
gmsh.model.mesh.optimize("Netgen")

# Apply the 1/3 scaling in z to obtain the elliptical cross sections. This also
# scales the outer spherical boundary into an ellipsoid with z semi-axis R/3.
node_tags, coords, _ = gmsh.model.mesh.getNodes()
coords = coords.reshape(-1, 3)
for tag, (xx, yy, zz) in zip(node_tags, coords):
    gmsh.model.mesh.setNode(int(tag), [float(xx), float(yy), float(zz) / 3.0], [])

gmsh.write(out)
print(f"Wrote {out}: {len(node_tags)} nodes, length {almond_len:.4f} m, "
      f"z half-size {R / 3.0:.4f} m, inner surfaces {len(inner)}")
gmsh.finalize()
