# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

#=
# README

This Julia script uses Gmsh to create a mesh for plane wave scattering from a perfectly
conducting sphere.

The generated mesh contains three distinct regions:
1. A 3D volume region (free space)
2. An inner spherical boundary (the PEC sphere)
3. An outer spherical boundary (the absorbing boundary)

## Prerequisites

This script requires the Gmsh Julia package. If you don't already have it installed, you can
install it with

```bash
julia -e 'using Pkg; Pkg.add("Gmsh")'
```

## How to run

From this directory, run:
```bash
julia -e 'include("mesh.jl"); generate_sphere_mesh(; filename="sphere.msh")'
```
This generates the mesh used in the example.

To visualize the mesh in Gmsh's graphical interface, add the `gui=true` parameter:
```bash
julia -e 'include("mesh.jl"); generate_sphere_mesh(; filename="sphere.msh", gui=true)'
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
    generate_sphere_mesh(;
        filename::AbstractString,
        radius::Real = 0.5,
        outer_radius::Real = 2.5944,
        hmin::Real = 0.10,
        hmax::Real = 0.25,
        dist_max::Real = 0.6,
        order::Integer = 2,
        verbose::Integer = 5,
        gui::Bool = false
    )

Generate a mesh for the PEC sphere scattering example using Gmsh.

# Arguments

  - filename - the filename to use for the generated mesh
  - radius - radius of the PEC sphere
  - outer_radius - radius of the outer absorbing boundary (a sphere)
  - hmin - element size on the inner (PEC) surface
  - hmax - maximum element size at the outer boundary
  - dist_max - distance over which the element size grows from `hmin` to `hmax`
  - order - polynomial order of the generated (curved) elements
  - verbose - flag to dictate the level of print to REPL, passed to Gmsh
    (0-5, higher = more verbose)
  - gui - whether to launch the Gmsh GUI on mesh generation
"""
function generate_sphere_mesh(;
    filename::AbstractString,
    radius::Real=0.5,
    outer_radius::Real=2.5944,
    hmin::Real=0.10,
    hmax::Real=0.25,
    dist_max::Real=0.6,
    order::Integer=2,
    verbose::Integer=5,
    gui::Bool=false
)
    # Boilerplate
    kernel = gmsh.model.occ
    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", verbose)

    if "sphere" in gmsh.model.list()
        gmsh.model.setCurrent("sphere")
        gmsh.model.remove()
    end
    gmsh.model.add("sphere")

    # Create the free-space region by subtracting the PEC sphere from the outer sphere
    outer = kernel.addSphere(0.0, 0.0, 0.0, outer_radius)
    inner = kernel.addSphere(0.0, 0.0, 0.0, radius)
    cut_dimtags, _ = kernel.cut([(3, outer)], [(3, inner)])
    gmsh.model.occ.synchronize()

    @assert length(cut_dimtags) == 1 && first(cut_dimtags)[1] == 3
    domain = extract_tag(cut_dimtags)

    # Identify the inner and outer surfaces after the boolean operation
    surfaces = gmsh.model.getEntities(2)
    inner_surfaces = [tag for (_, tag) in surfaces if bbox_extent(2, tag) < outer_radius]
    outer_surfaces = [tag for (_, tag) in surfaces if !(tag in inner_surfaces)]
    @assert !isempty(inner_surfaces) && !isempty(outer_surfaces)

    # Physical groups: domain = 1, outer absorbing boundary = 1, PEC sphere = 2
    gmsh.model.addPhysicalGroup(3, [domain], 1)
    gmsh.model.addPhysicalGroup(2, outer_surfaces, 1)
    gmsh.model.addPhysicalGroup(2, inner_surfaces, 2)

    # Refine the mesh near the PEC sphere using a distance-based size field
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
    println("Domain: ", 1)
    println("Absorbing boundary: ", 1)
    println("PEC sphere: ", 2)
    println()

    # Optionally launch the Gmsh GUI
    if gui
        gmsh.fltk.run()
    end

    # Clean up Gmsh resources
    return gmsh.finalize()
end
