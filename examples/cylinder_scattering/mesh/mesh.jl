# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

#=
# README

This Julia script uses Gmsh to create a two-dimensional mesh for plane wave scattering
from a perfectly conducting circular cylinder.

The generated mesh contains three distinct regions:
1. A 2D annular domain (free space)
2. An inner circular boundary (the PEC cylinder)
3. An outer circular boundary (the absorbing boundary)

## Prerequisites

This script requires the Gmsh Julia package. If you don't already have it installed, you can
install it with

```bash
julia -e 'using Pkg; Pkg.add("Gmsh")'
```

## How to run

From this directory, run:
```bash
julia -e 'include("mesh.jl"); generate_cylinder_mesh(; filename="cylinder.msh")'
```
This generates the mesh used in the example.

To visualize the mesh in Gmsh's graphical interface, add the `gui=true` parameter:
```bash
julia -e 'include("mesh.jl"); generate_cylinder_mesh(; filename="cylinder.msh", gui=true)'
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
    generate_cylinder_mesh(;
        filename::AbstractString,
        radius::Real = 0.5,
        outer_radius::Real = 5.21,
        hmin::Real = 0.13,
        hmax::Real = 0.39,
        dist_max::Real = 2.36,
        order::Integer = 2,
        verbose::Integer = 5,
        gui::Bool = false
    )

Generate a mesh for the 2D cylinder scattering example using Gmsh.

# Arguments

  - filename - the filename to use for the generated mesh
  - radius - radius of the PEC cylinder
  - outer_radius - radius of the outer absorbing boundary
  - hmin - element size on the inner (PEC) boundary
  - hmax - maximum element size at the outer boundary
  - dist_max - distance over which the element size grows from `hmin` to `hmax`
  - order - polynomial order of the generated (curved) elements
  - verbose - flag to dictate the level of print to REPL, passed to Gmsh
    (0-5, higher = more verbose)
  - gui - whether to launch the Gmsh GUI on mesh generation
"""
function generate_cylinder_mesh(;
    filename::AbstractString,
    radius::Real=0.5,
    outer_radius::Real=5.21,
    hmin::Real=0.13,
    hmax::Real=0.39,
    dist_max::Real=2.36,
    order::Integer=2,
    verbose::Integer=5,
    gui::Bool=false
)
    # Boilerplate
    kernel = gmsh.model.occ
    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", verbose)

    if "cylinder" in gmsh.model.list()
        gmsh.model.setCurrent("cylinder")
        gmsh.model.remove()
    end
    gmsh.model.add("cylinder")

    # Create the annulus by subtracting the inner disk from the outer disk
    outer = kernel.addDisk(0.0, 0.0, 0.0, outer_radius, outer_radius)
    inner = kernel.addDisk(0.0, 0.0, 0.0, radius, radius)
    cut_dimtags, _ = kernel.cut([(2, outer)], [(2, inner)])
    gmsh.model.occ.synchronize()

    @assert length(cut_dimtags) == 1 && first(cut_dimtags)[1] == 2
    domain = extract_tag(cut_dimtags)

    # Identify the inner and outer curves after the boolean operation
    curves = gmsh.model.getEntities(1)
    inner_curves = [tag for (_, tag) in curves if bbox_extent(1, tag) < outer_radius]
    outer_curves = [tag for (_, tag) in curves if !(tag in inner_curves)]
    @assert !isempty(inner_curves) && !isempty(outer_curves)

    # Physical groups: domain = 1, outer absorbing boundary = 1, PEC cylinder = 2
    gmsh.model.addPhysicalGroup(2, [domain], 1)
    gmsh.model.addPhysicalGroup(1, outer_curves, 1)
    gmsh.model.addPhysicalGroup(1, inner_curves, 2)

    # Refine the mesh near the PEC cylinder using a distance-based size field
    f1 = gmsh.model.mesh.field.add("Distance")
    gmsh.model.mesh.field.setNumbers(f1, "CurvesList", inner_curves)
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
    gmsh.model.mesh.generate(2)
    if order == 1
        gmsh.model.mesh.optimize("Netgen")
    end
    gmsh.write(joinpath(@__DIR__, filename))

    println("\nFinished generating mesh. Physical group tags:")
    println("Domain: ", 1)
    println("Absorbing boundary: ", 1)
    println("PEC cylinder: ", 2)
    println()

    # Optionally launch the Gmsh GUI
    if gui
        gmsh.fltk.run()
    end

    # Clean up Gmsh resources
    return gmsh.finalize()
end
