```@raw html
<!---
Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
SPDX-License-Identifier: Apache-2.0
--->
```

# Plane Wave Scattering from a Lossy Debye Sphere

!!! note

    The files for this example can be found in the
    [`examples/debye_sphere/`](https://github.com/awslabs/palace/blob/main/examples/debye_sphere)
    directory of the *Palace* source code.

This example demonstrates plane wave scattering from a homogeneous, lossy dielectric
sphere. The frequency-dependent permittivity is described by the single-pole Debye
relaxation model [[1]](#References),

```math
\varepsilon(\omega) = \varepsilon_\infty + \frac{\Delta\varepsilon}{1 + i\omega\tau},
```

and the scattered far field is compared with the exact Mie series for a complex
permittivity [[2]](#References).

## Problem Setup

A dielectric sphere of radius ``a = 0.5\text{ m}`` is illuminated by a ``z``-polarized
plane wave traveling in the ``-z`` direction with unit amplitude. The Debye parameters are
chosen as

```math
\varepsilon_\infty = 2, \qquad \Delta\varepsilon = 2, \qquad \tau = \frac{1}{2\pi f_0},
```

so that at the operating frequency ``f_0 = 0.0954269\text{ GHz}`` (``ka = 1``) the complex
relative permittivity is

```math
\varepsilon(\omega_0) = 2 + \frac{2}{1 + i} = 3 - i .
```

In *Palace*, a complex permittivity ``\varepsilon = \varepsilon_r (1 - i\tan\delta)`` is
specified by the real part and the loss tangent, so the example uses

```json
"Materials":
[
    { "Attributes": [1] },
    { "Attributes": [2], "Permittivity": 3.0, "LossTan": 0.3333333333333333 }
]
```

The sphere (attribute 2) is embedded in a free-space shell (attribute 1) with a spherical
absorbing boundary of radius ``R = a + 1.5\lambda = 5.2124\text{ m}``. The incident wave
enters the dielectric through the volume contrast sources of the [scattered-field
formulation](../features/farfield.md#Scattered-field-formulation), which support complex
(lossy) permittivities.

## Results

The figure below compares the bistatic RCS in the ``\phi = 0`` plane with the Mie series
for ``\varepsilon = 3 - i``. The monostatic value is also shown.

```@raw html
<p align="center">
  <img src="../../assets/examples/debye_sphere-1.png" width="70%" />
</p>
```

The *Palace* solution tracks the Mie series over the full angular range, with a monostatic
error of ``0.4\text{ dB}`` and a pattern RMS error of ``0.4\text{ dB}``. The deep null near
``\theta \approx 90^\circ`` is resolved to within about ``0.8\text{ dB}``.

## References

[1] P. Debye, *Polar Molecules*, Chemical Catalog Company, New York, 1929.\
[2] C. F. Bohren and D. R. Huffman, *Absorption and Scattering of Light by Small
Particles*, Wiley, New York, 1983.
