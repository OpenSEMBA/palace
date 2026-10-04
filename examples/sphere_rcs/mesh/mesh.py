# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Gmsh mesh generator for radar cross section (RCS) computations of a perfectly conducting
sphere.

The mesh discretizes the air region between a sphere of radius `radius` and a concentric
outer sphere of radius `radius_farfield`, on which absorbing boundary conditions are applied
and the near-to-far-field transformation is evaluated. For a perfectly conducting (PEC)
sphere, the interior of the sphere is removed from the mesh. For a dielectric sphere
(`dielectric=True`), the interior is meshed as a separate domain.

The generated mesh contains the following regions (printed on generation):
  1. The 3D air region (domain attribute 1)
  2. The 3D sphere interior, only for a dielectric sphere (domain attribute 2)
  3. The outer spherical boundary for the absorbing boundary condition (boundary attribute 1)
  4. The PEC sphere surface, only for a PEC sphere (boundary attribute 2)

Requirements: the Gmsh Python API (`pip install gmsh` or `conda install python-gmsh`).

Usage, from this directory:

    python mesh.py --radius 0.5 --radius-farfield 1.25 --h-sphere 0.06 \
        --h-farfield 0.12 --filename sphere.msh

or from Python:

    from mesh import generate_sphere_mesh
    generate_sphere_mesh(filename="sphere.msh", radius=0.5, radius_farfield=1.25,
                         h_sphere=0.06, h_farfield=0.12)
"""

import argparse

import gmsh

DOMAIN_ATTRIBUTE = 1
DIELECTRIC_ATTRIBUTE = 2
FARFIELD_ATTRIBUTE = 1
PEC_ATTRIBUTE = 2


def generate_sphere_mesh(
    filename,
    radius=0.5,
    radius_farfield=1.25,
    h_sphere=0.06,
    h_farfield=0.12,
    h_interior=None,
    dielectric=False,
    order=2,
    verbose=2,
    gui=False,
):
    """
    Generate the mesh of the air region between a sphere and a concentric outer boundary.

    Arguments:
      filename: Output mesh file (Gmsh MSH 2.2 format, as read by Palace).
      radius: Radius of the scattering (PEC) sphere, in mesh units (m).
      radius_farfield: Radius of the outer boundary sphere, in mesh units (m).
      h_sphere: Target element size on the surface of the scattering sphere.
      h_farfield: Target element size on the outer boundary sphere.
      h_interior: Target element size inside a dielectric sphere (default: h_sphere).
      dielectric: If True, mesh the interior of the sphere (domain attribute 2) instead of
        removing it and defining its surface as a PEC boundary.
      order: Geometric order of the (curved) elements.
      verbose: Gmsh verbosity level (0-5).
      gui: Launch the Gmsh GUI after mesh generation.

    Returns a dictionary with the number of elements and the attribute numbers.
    """
    assert radius_farfield > radius > 0.0

    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", verbose)
    if "sphere_rcs" in gmsh.model.list():
        gmsh.model.setCurrent("sphere_rcs")
        gmsh.model.remove()
    gmsh.model.add("sphere_rcs")
    kernel = gmsh.model.occ

    # The air region is the outer sphere minus the scattering sphere. For a dielectric
    # sphere, the interior is kept as a second volume sharing the interface.
    sphere = kernel.addSphere(0.0, 0.0, 0.0, radius)
    sphere_farfield = kernel.addSphere(0.0, 0.0, 0.0, radius_farfield)
    if dielectric:
        # Fragment to obtain two volumes with a shared (conforming) interface.
        out_dimtags, _ = kernel.fragment([(3, sphere_farfield)], [(3, sphere)])
        kernel.synchronize()
        volumes = [tag for dim, tag in out_dimtags if dim == 3]
        assert len(volumes) == 2

        def extent(tag):
            xmin, _, _, xmax, _, _ = gmsh.model.getBoundingBox(3, tag)
            return xmax - xmin

        sphere, domain = sorted(volumes, key=extent)
    else:
        cut_dimtags, _ = kernel.cut([(3, sphere_farfield)], [(3, sphere)])
        assert len(cut_dimtags) == 1 and cut_dimtags[0][0] == 3
        domain = cut_dimtags[0][1]
        kernel.synchronize()

    # Identify the inner and outer spherical surfaces by their bounding boxes.
    surfaces = gmsh.model.getBoundary([(3, domain)], oriented=False)
    inner, outer = [], []
    for dim, tag in surfaces:
        xmin, _, _, xmax, _, _ = gmsh.model.getBoundingBox(dim, tag)
        (inner if 0.5 * (xmax - xmin) < 0.5 * (radius + radius_farfield) else outer).append(
            tag
        )
    assert len(inner) == 1 and len(outer) == 1

    # Physical groups become attributes in Palace. Boundary attributes are numbered
    # independently from domain attributes.
    gmsh.model.addPhysicalGroup(3, [domain], DOMAIN_ATTRIBUTE, "air")
    gmsh.model.addPhysicalGroup(2, outer, FARFIELD_ATTRIBUTE, "farfield")
    if dielectric:
        gmsh.model.addPhysicalGroup(3, [sphere], DIELECTRIC_ATTRIBUTE, "dielectric")
    else:
        gmsh.model.addPhysicalGroup(2, inner, PEC_ATTRIBUTE, "pec")

    # Element size grows linearly from h_sphere on the scatterer to h_farfield on the outer
    # boundary.
    gmsh.model.mesh.field.add("Distance", 1)
    gmsh.model.mesh.field.setNumbers(1, "SurfacesList", inner)
    gmsh.model.mesh.field.setNumber(1, "Sampling", 100)
    gmsh.model.mesh.field.add("Threshold", 2)
    gmsh.model.mesh.field.setNumber(2, "InField", 1)
    gmsh.model.mesh.field.setNumber(2, "SizeMin", h_sphere)
    gmsh.model.mesh.field.setNumber(2, "SizeMax", h_farfield)
    gmsh.model.mesh.field.setNumber(2, "DistMin", 0.0)
    gmsh.model.mesh.field.setNumber(2, "DistMax", radius_farfield - radius)
    if dielectric:
        # Constant size inside the dielectric, to resolve the shorter wavelength.
        gmsh.model.mesh.field.add("Constant", 3)
        gmsh.model.mesh.field.setNumber(3, "VIn", h_interior or h_sphere)
        gmsh.model.mesh.field.setNumber(3, "VOut", 1.0e22)
        gmsh.model.mesh.field.setNumbers(3, "VolumesList", [sphere])
        gmsh.model.mesh.field.add("Min", 4)
        gmsh.model.mesh.field.setNumbers(4, "FieldsList", [2, 3])
        gmsh.model.mesh.field.setAsBackgroundMesh(4)
    else:
        gmsh.model.mesh.field.setAsBackgroundMesh(2)
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)

    # Use deterministic (non-parallel) algorithms so the mesh is reproducible.
    gmsh.option.setNumber("Mesh.Algorithm", 6)
    gmsh.option.setNumber("Mesh.Algorithm3D", 1)
    gmsh.model.mesh.generate(3)
    gmsh.model.mesh.setOrder(order)

    # Palace reads the MSH 2.2 format; save only the elements in physical groups.
    gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
    gmsh.option.setNumber("Mesh.Binary", 0)
    gmsh.option.setNumber("Mesh.SaveAll", 0)
    gmsh.write(filename)

    _, tets, _ = gmsh.model.mesh.getElements(3)
    info = {
        "num_tetrahedra": int(sum(len(t) for t in tets)),
        "domain": DOMAIN_ATTRIBUTE,
        "farfield": FARFIELD_ATTRIBUTE,
    }
    if dielectric:
        info["dielectric"] = DIELECTRIC_ATTRIBUTE
    else:
        info["pec"] = PEC_ATTRIBUTE
    if verbose > 0:
        print(f"Wrote {filename} with {info['num_tetrahedra']} tetrahedra")
        print(f"  Domain attribute (air):            {DOMAIN_ATTRIBUTE}")
        if dielectric:
            print(f"  Domain attribute (dielectric):     {DIELECTRIC_ATTRIBUTE}")
        print(f"  Boundary attribute (absorbing):    {FARFIELD_ATTRIBUTE}")
        if not dielectric:
            print(f"  Boundary attribute (PEC sphere):   {PEC_ATTRIBUTE}")

    if gui:
        gmsh.fltk.run()
    gmsh.finalize()
    return info


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[1])
    parser.add_argument("--filename", default="sphere.msh")
    parser.add_argument("--radius", type=float, default=0.5)
    parser.add_argument("--radius-farfield", type=float, default=1.25)
    parser.add_argument("--h-sphere", type=float, default=0.06)
    parser.add_argument("--h-farfield", type=float, default=0.12)
    parser.add_argument("--h-interior", type=float, default=None)
    parser.add_argument("--dielectric", action="store_true")
    parser.add_argument("--order", type=int, default=2)
    parser.add_argument("--gui", action="store_true")
    args = parser.parse_args()
    generate_sphere_mesh(
        filename=args.filename,
        radius=args.radius,
        radius_farfield=args.radius_farfield,
        h_sphere=args.h_sphere,
        h_farfield=args.h_farfield,
        h_interior=args.h_interior,
        dielectric=args.dielectric,
        order=args.order,
        gui=args.gui,
    )
