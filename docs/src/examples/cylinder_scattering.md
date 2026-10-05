```@raw html
<!---
Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
SPDX-License-Identifier: Apache-2.0
--->
```

# Plane Wave Scattering from a 2D PEC Cylinder

!!! note

    The files for this example can be found in the
    [`examples/cylinder_scattering/`](https://github.com/awslabs/palace/blob/main/examples/cylinder_scattering)
    directory of the *Palace* source code.

This example demonstrates *Palace*'s two-dimensional plane wave scattering and RCS
postprocessing for a perfectly conducting circular cylinder. The scattered field is
compared with the exact TE Mie series [[1]](#References).

## Problem Setup

A PEC circular cylinder of radius ``a = 0.5\text{ m}`` is illuminated by a plane wave
traveling in the ``-x`` direction with the electric field in the simulation plane,

```math
E_{\text{inc}} = \hat{y} \, e^{-ikx}, \qquad H_{\text{inc}} = \hat{z} \, e^{-ikx},
```

with unit amplitude. This is the two-dimensional TE polarization: the electric field is
in-plane and the magnetic field is out-of-plane. The operating frequency is
``f = 0.0954269\text{ GHz}``, so that ``ka = 1``.

The cylinder is surrounded by a circular absorbing boundary of radius
``R = a + 1.5\lambda = 5.21\text{ m}``. The far-field postprocessing for two-dimensional
problems uses the equivalence-principle line integral on the absorbing boundary and reports
the scattering width ``\sigma_{2D} = 2\pi |E_\infty|^2 / |E_{\text{inc}}|^2`` in meters. All
observation directions must lie in the simulation plane, so ``\theta = 90^\circ`` and the
azimuth ``\phi`` spans the full circle:

```json
"Postprocessing":
{
    "FarField":
    {
        "Attributes": [1],
        "ThetaPhis": [[90, 0], [90, 5], ..., [90, 355]],
        "Monostatic": true
    }
}
```

The full configuration is available in
[`examples/cylinder_scattering/cylinder_scattering.json`](https://github.com/awslabs/palace/blob/main/examples/cylinder_scattering/cylinder_scattering.json),
and the mesh can be regenerated with
[`examples/cylinder_scattering/mesh/mesh.jl`](https://github.com/awslabs/palace/blob/main/examples/cylinder_scattering/mesh/mesh.jl).

## Results

The figure below compares the bistatic scattering width with the TE Mie series. The
monostatic (backscatter) value is also shown.

```@raw html
<p align="center">
  <img src="../../assets/examples/cylinder_scattering-1.png" width="70%" />
</p>
```

The *Palace* solution matches the Mie series over the full circle, with a monostatic error
of ``0.24\text{ dB}`` and a pattern RMS error of ``0.29\text{ dB}``. The deep nulls near
``\phi \approx 120^\circ`` and ``\phi \approx 240^\circ``, where the scattering width drops
more than ``20\text{ dB}`` below the forward-scattering peak, are resolved to within
``0.5\text{ dB}``.

## References

[1] C. A. Balanis, *Advanced Engineering Electromagnetics*, 2nd ed., Wiley, Hoboken, NJ,
2012, Ch. 11.\
[2] G. T. Ruck, D. E. Barrick, W. D. Stuart, and C. K. Krichbaum, *Radar Cross Section
Handbook*, Plenum Press, New York, 1970.\
[3] J. J. Bowman, T. B. A. Senior, and P. L. E. Uslenghi, *Electromagnetic and Acoustic
Scattering by Simple Shapes*, North-Holland, Amsterdam, 1969.
