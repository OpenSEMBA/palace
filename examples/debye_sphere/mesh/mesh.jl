# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

#=
# README

This Julia script uses Gmsh to create a mesh for plane wave scattering from a homogeneous
dielectric (Debye) sphere.

The generated mesh contains four distinct regions:
1. A 3D shell region (free space, physical tag 1)
2. A 3D ball region (the dielectric sphere, physical tag 2)
3. An inner spherical interface (between the two volume regions)
4. An outer spherical boundary (the absorbing boundary, physical tag 1)

## Prerequisites

This script requires the Gmsh Julia package. If you don't already have it installed, you can
install it with

```bash
julia -e 'using Pkg; Pkg.add("Gmsh")'
```

## How to run

From this directory, run:
```bash
julia -e 'include("mesh.jl"); generate_dielectric_sphere_mesh(; filename="debye_sphere.msh")'
```
This generates the mesh used in the example.

To visualize the mesh in Gmsh's graphical interface, add the `gui=true` parameter:
```bash
julia -e 'include("mesh.jl"); generate_dielectric_sphere_mesh(; filename="debye_sphere.msh", gui=true)'
```

The script will generate a mesh file and print the "attribute" numbers for each region.
These attributes are needed when configuring Palace simulations.
=#

using Gmsh: gmsh

# Convenience function to extract the tag (ID) from the return values of several Gmsh
# functions
extract_tag(entity) = first(entity)[2]

# Maximum extent of the bounding box of a Gmsh entity.
function bbox_extent(dim::Integer, tag::Integer)
    b = gmsh.model.getBoundingBox(dim, tag)
    return maximum(b[i + 3] - b[i] for i = 1:3)
end

"""
    generate_dielectric_sphere_mesh(;
        filename::AbstractString,
        radius::Real = 0.5,
        outer_radius::Real = 5.2124,
        hmin::Real = 0.1,
        hmax::Real = 0.45,
        dist_max::Real = 1.5,
        order::Integer = 2,
        verbose::Integer = 5,
        gui::Bool = false
    )

Generate a mesh for the dielectric sphere scattering example using Gmsh.

# Arguments

  - filename - the filename to use for the generated mesh
  - radius - radius of the dielectric sphere
  - outer_radius - radius of the outer absorbing boundary (a sphere)
  - hmin - element size on the inner spherical interface
  - hmax - maximum element size at the outer boundary
  - dist_max - distance over which the element size grows from `hmin` to `hmax`
  - order - polynomial order of the generated (curved) elements
  - verbose - flag to dictate the level of print to REPL, passed to Gmsh
    (0-5, higher = more verbose)
  - gui - whether to launch the Gmsh GUI on mesh generation
"""
function generate_dielectric_sphere_mesh(;
    filename::AbstractString,
    radius::Real=0.5,
    outer_radius::Real=5.2124,
    hmin::Real=0.1,
    hmax::Real=0.45,
    dist_max::Real=1.5,
    order::Integer=2,
    verbose::Integer=5,
    gui::Bool=false
)
    # Boilerplate
    kernel = gmsh.model.occ
    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", verbose)

    if "dielectric_sphere" in gmsh.model.list()
        gmsh.model.setCurrent("dielectric_sphere")
        gmsh.model.remove()
    end
    gmsh.model.add("dielectric_sphere")

    # Create the dielectric ball and the surrounding free-space shell. The fragment
    # operation keeps the ball and shell as separate volumes sharing a conformal
    # interface.
    outer = kernel.addSphere(0.0, 0.0, 0.0, outer_radius)
    inner = kernel.addSphere(0.0, 0.0, 0.0, radius)
    kernel.fragment([(3, outer)], [(3, inner)])
    gmsh.model.occ.synchronize()

    # Identify the ball and shell volumes, and the outer boundary surface
    volumes = gmsh.model.getEntities(3)
    ball = [tag for (_, tag) in volumes if bbox_extent(3, tag) < outer_radius]
    shell = [tag for (_, tag) in volumes if !(tag in ball)]
    surfaces = gmsh.model.getEntities(2)
    outer_surfaces = [tag for (_, tag) in surfaces if bbox_extent(2, tag) > outer_radius]
    @assert !isempty(ball) && !isempty(shell) && !isempty(outer_surfaces)

    # Physical groups: shell (free space) = 1, ball (dielectric) = 2,
    # outer absorbing boundary = 1
    gmsh.model.addPhysicalGroup(3, shell, 1)
    gmsh.model.addPhysicalGroup(3, ball, 2)
    gmsh.model.addPhysicalGroup(2, outer_surfaces, 1)

    # Refine the mesh near the dielectric interface using a distance-based size field
    inner_surfaces = [tag for (_, tag) in surfaces if bbox_extent(2, tag) < outer_radius]
    f1 = gmsh.model.mesh.field.add("Distance")
    gmsh.model.mesh.field.setNumbers(f1, "SurfacesList", inner_surfaces)
    gmsh.model.mesh.field.setNumber(f1, "Sampling", 400)
    f2 = gmsh.model.mesh.field.add("Threshold")
    gmsh.model.mesh.field.setNumber(f2, "InField", f1)
    gmsh.model.mesh.field.setNumber(f2, "SizeMin", hmin)
    gmsh.model.mesh.field.setNumber(f2, "SizeMax", hmax)
    gmsh.model.mesh.field.setNumber(f2, "DistMin", 0.0)
    gmsh.model.mesh.field.setNumber(f2, "DistMax", dist_max)
    gmsh.model.mesh.field.setAsBackgroundMesh(f2)

    # Generate the mesh and save it using the format expected by Palace
    gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
    gmsh.option.setNumber("Mesh.ElementOrder", order)
    if order > 1
        gmsh.option.setNumber("Mesh.HighOrderOptimize", 1)
    end
    gmsh.model.mesh.generate(3)
    if order == 1
        gmsh.model.mesh.optimize("Netgen")
    end
    gmsh.write(joinpath(@__DIR__, filename))

    println("\nFinished generating mesh. Physical group tags:")
    println("Free space: ", 1)
    println("Dielectric sphere: ", 2)
    println("Absorbing boundary: ", 1)
    println()

    # Optionally launch the Gmsh GUI
    if gui
        gmsh.fltk.run()
    end

    # Clean up Gmsh resources
    return gmsh.finalize()
end
