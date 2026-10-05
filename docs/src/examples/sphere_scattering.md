```@raw html
<!---
Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
SPDX-License-Identifier: Apache-2.0
--->
```

# Plane Wave Scattering from a PEC Sphere

!!! note

    The files for this example can be found in the
    [`examples/sphere_scattering/`](https://github.com/awslabs/palace/blob/main/examples/sphere_scattering)
    directory of the *Palace* source code.

This example demonstrates *Palace*'s plane wave excitation and radar cross section (RCS)
postprocessing for a perfectly conducting (PEC) sphere. The scattered far field computed
by *Palace* is compared with the exact Mie series solution, which is the classical
benchmark for three-dimensional scattering codes [[1]](#References).

## Problem Setup

A PEC sphere of radius ``a = 0.5\text{ m}`` is illuminated by a ``z``-polarized plane wave
traveling in the ``-z`` direction,

```math
E_{\text{inc}} = \hat{x} \, e^{-ikz}, \qquad H_{\text{inc}} = \hat{y} \, e^{-ikz},
```

with unit amplitude (``1\text{ V/m}``). The operating frequency is ``f = 0.2863\text{ GHz}``,
so that ``ka = 2\pi f a / c = 3``. The scatterer is surrounded by a spherical absorbing
boundary of radius ``R = a + 2\lambda = 2.5944\text{ m}``, which uses the second-order
Bayliss-Turkel absorbing boundary condition with the curvature term enabled through
[`config["Boundaries"]["Absorbing"]["Radius"]`](../config/reference.md#config-boundaries-absorbing).

The solution is computed with the [scattered-field
formulation](../features/farfield.md#Scattered-field-formulation): the incident wave is
imposed on the PEC surface through the boundary condition ``n \times E_s = -n \times E_{\text{inc}}``, while the absorbing boundary condition is homogeneous. The
[far-field postprocessing](../features/farfield.md#Incident-Plane-Waves-and-Scattering)
then evaluates the scattered far field on the absorbing boundary and writes the bistatic
RCS and the monostatic (backscatter) RCS.

## Configuration

The plane wave and far-field postprocessing are configured as follows:

```json
"PlaneWave":
[
    {
        "Index": 1,
        "Attributes": [1],
        "Excitation": 1,
        "Direction": [0.0, 0.0, -1.0],
        "Polarization": [1.0, 0.0, 0.0],
        "Amplitude": 1.0
    }
],
"Postprocessing":
{
    "FarField":
    {
        "Attributes": [1],
        "ThetaPhis": [[0, 0], [5, 0], ..., [180, 0]],
        "Monostatic": true
    }
}
```

The full configuration is available in
[`examples/sphere_scattering/sphere_scattering.json`](https://github.com/awslabs/palace/blob/main/examples/sphere_scattering/sphere_scattering.json),
and the mesh can be regenerated with
[`examples/sphere_scattering/mesh/mesh.jl`](https://github.com/awslabs/palace/blob/main/examples/sphere_scattering/mesh/mesh.jl).

## Results

The figure below compares the bistatic RCS in the ``\phi = 0`` (E-) plane with the Mie
series. The monostatic value is also shown.

```@raw html
<p align="center">
  <img src="../../assets/examples/sphere_scattering-1.png" width="70%" />
</p>
```

The *Palace* solution follows the Mie series over the full angular range, including the
interference nulls near ``\theta \approx 85^\circ`` and ``\theta \approx 105^\circ``. The
monostatic RCS is within ``1.3\text{ dB}`` of the analytic value and the pattern RMS error
is ``0.7\text{ dB}``. This level of agreement is representative of a scattered-field
simulation with a modest mesh; convergence studies with finer meshes and larger boundary
distances give errors below ``0.3\text{ dB}``.

## References

[1] G. T. Ruck, D. E. Barrick, W. D. Stuart, and C. K. Krichbaum, *Radar Cross Section
Handbook*, Plenum Press, New York, 1970.
