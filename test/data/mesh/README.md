<!---
Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
SPDX-License-Identifier: Apache-2.0
--->

These mesh files originate from the MFEM repository, see
https://github.com/mfem/mfem/tree/master/data.

The exception is `sphere-shell-p2.msh`, a second-order (curved) tetrahedral mesh of the region
between concentric spheres of radii 0.5 and 1.0 (boundary attribute 1: outer sphere, boundary
attribute 2: inner sphere), generated with the Gmsh script in `examples/sphere_rcs/mesh`:

```bash
python examples/sphere_rcs/mesh/mesh.py --filename test/data/mesh/sphere-shell-p2.msh \
  --radius 0.5 --radius-farfield 1.0 --h-sphere 0.25 --h-farfield 0.35 --order 2
```
