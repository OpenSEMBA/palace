```@raw html
<!---
Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
SPDX-License-Identifier: Apache-2.0
--->
```

# Extracting Fields in the Radiation Zone

When run in the [`Driven` or `Eigenmode` modes](../guide/problem.md), *Palace*
can extrapolate fields from the near-field region (being simulated) to the
far-field zone. This capability can be used to study the radiative properties of
a system, or, combined with an incident plane wave excitation, its scattering properties
such as the [radar cross section](#Plane-wave-scattering-and-radar-cross-section).

The mathematical details on how this is accomplished are available in the
[reference](../reference.md#Far-field-extraction). The key points are as
follows:

  - Computing fields requires evaluation of surface integrals, so users must
    specify a surface of integration (more on this later).
  - For the result to be physically accurate, it is important to properly model
    the propagation of waves to infinity. This can be accomplished by enclosing
    the system inside a sphere or box and applying
    [`config["Boundaries"]["Absorbing"]` boundary
    conditions](../config/reference.md#config-boundaries-absorbing).
  - The result is provided as complex vectors ``r \mathbf{E}(\theta, \phi)``,
    where ``(\theta, \phi)`` identify a point on sphere at infinite distance and
    the result is defined up to a global phase.

*Palace* outputs ``r\mathbf{E}`` because this is a well-defined, finite quantity
(``\mathbf{E}`` itself goes to zero at infinity as 1/r). You can also think of
this as the electric field measured at a distance of one unit of length. With
this output, you can immediately compute various quantities of interest. For
instance, ``|r\mathbf{E}|^2`` gives the relative radiative power.

!!! warning "Limitations"

    *Palace* only supports propagation of fields to infinity when the integration
    surface

     1. is an external boundary
     2. does not cross anisotropic materials

## Setup

A typical setup consists of starting from the system under consideration and
enclosing the system inside an outer boundary (typically a sphere or a box), if
it is not already. Then, we set `"Absorbing" ` boundary conditions on this
surface and choose it as the surface for the integration. For best accuracy, it
is a good idea to make sure that this outer boundary is meshed finely enough to
resolve the expected wavelength.

Turning on far-field extraction requires activating the feature in the
configuration JSON file. To do so, we look at the `"FarField"` section under
`"Postprocessing"` in `"Boundaries"`. Here, we need to specify the identifier of
the integration surface in `"Attributes"` and specify how we want to sample the
sphere at infinity. As in many other parts of *Palace*, `"Attributes"` expects a
vector, as it can happen the boundary is split in multiple pieces.

Once we define the surface of integration, we need to specify where we want to
evaluate target far-field points. The simplest way to do this is by setting
`"NSample"`, so that the far-field sphere is uniformly sampled with `NSample`
points. (Note that uniform on a sphere means that the polar angle ``\theta`` of
the sampled points is not uniformly distributed, to avoid bunching of points on
the poles.)

Often, uniform sampling on the far-field sphere might be a good first step, but
not the best way to accurately capture the radiative properties (e.g., when the
radiation is highly directional). In this case, you can specify at what angles
``\theta, \phi`` you want to evaluate the fields. This is done by passing
2-vectors to the `"ThetaPhis"` array with the angular coordinates of your
choosing (in degrees).

This can be combined with scripts to target specific regions. For instance, the
following Python code produces the required `"ThetaPhis"` section to sample over
the xy plane:

```python
import json
dphi = 1  # degree
print(json.dumps({"ThetaPhis": [[90.0, p] for p in range(0, 361, dphi)]}))
```

Both `"NSample"` and `"ThetaPhis"` can be specified simultaneously and the
results will be combined and duplicates removed.

## Output

Once a simulation is run, *Palace* generates a CSV file named `farfield-rE.csv`
in the folder specified by the
[`config["Problem"]["Output"]`](../config/reference.md#config-problem)
configuration.

The CSV contains a header line that describes the columns, and one row for each
``(\theta, \phi)`` pair (per excitation, for driven simulations with multiple
port excitations). The columns are:

  - `f (GHz)` : Frequency (driven). For eigenmode simulations this is replaced
    by `m` (mode index) followed by `f_re (GHz)` and `f_im (GHz)`.
  - `exc` : Excitation index. Constant for single-excitation simulations; varies
    across rows in multi-port driven port sweeps so each row uniquely identifies
    a `(frequency, angle, excitation)` triple.
  - `theta (deg.)` : Polar angle in degrees
  - `phi (deg.)` : Azimuthal angle in degrees
  - `r*Re{E_x} (V)`, `r*Im{E_x} (V)` : Real and imaginary parts of ``r E_x`` component
  - `r*Re{E_y} (V)`, `r*Im{E_y} (V)` : Real and imaginary parts of ``r E_y`` component
  - `r*Re{E_z} (V)`, `r*Im{E_z} (V)` : Real and imaginary parts of ``r E_z`` component

All field values are in SI units.

To obtain the magnetic fields, you can assume that propagation occurs in free
space, so that

```math
r \mathbf{H}_p(\mathbf{r}_0) = \frac{\mathbf{r}_0 \times r \mathbf{E}_p(\mathbf{r}_0)}{Z_0}
```

with ``Z_0`` impedance of free space.

This type of output can be processed by several different packages and
languages. For instance, see the [antenna example](../examples/antenna.md) for
an example of a [Julia](https://julialang.org/) script that plots antenna
patterns.

## Plane wave scattering and radar cross section

In `"Driven"` simulations, the model can be illuminated by an incident plane wave, which
is the typical setup for computing the scattering properties of an object such as its
[radar cross section](https://en.wikipedia.org/wiki/Radar_cross_section) (RCS). Plane waves
are specified in the [`config["Domains"]["PlaneWave"]`](../config/reference.md#config-domains-planewave)
array, with each entry defining the incident field

```math
\mathbf{E}_{inc}(\mathbf{x}) = E_0 \, \hat{\mathbf{p}} \, e^{-ik\hat{\mathbf{k}}\cdot\mathbf{x}}
```

with propagation direction ``\hat{\mathbf{k}}`` (`"Direction"`), polarization
``\hat{\mathbf{p}} \perp \hat{\mathbf{k}}`` (`"Polarization"`), and amplitude ``E_0``
in V/m (`"Amplitude"`, default 1 V/m), where ``k`` is the wavenumber in the background
medium, the material adjacent to the absorbing boundary. Each plane wave defines its own
excitation, with excitation index equal to its `"Index"`, so that several incidence
directions or polarizations (for example for a monostatic RCS as a function of the
incidence angle) can be computed in a single simulation, reusing the same discretization.

*Palace* uses a scattered field formulation (see the
[reference](../reference.md#Plane-wave-excitation)): the computed electric field is the
total field ``\mathbf{E} = \mathbf{E}_{inc} + \mathbf{E}_s``, but the absorbing boundary
condition acts only on the scattered field ``\mathbf{E}_s``, which is the only field
radiating to infinity. Scatterers can be any combination of PEC boundaries, impedance or
conductivity boundaries, and (possibly anisotropic or lossy) materials, as long as the
background medium adjacent to the absorbing boundary is isotropic and lossless. The fields
written for visualization are the total fields.

For excitations with an incident plane wave, the far field written to `farfield-rE.csv` is
the far field of the scattered field ``r \mathbf{E}_s``. When the plane wave is the only
source of the excitation, *Palace* additionally writes the bistatic radar cross section

```math
\sigma(\theta, \phi) = \lim_{r\to\infty} 4 \pi r^2 \frac{|\mathbf{E}_s|^2}{|\mathbf{E}_{inc}|^2}
  = 4 \pi \frac{|r \mathbf{E}_s|^2}{E_0^2}
```

to the file `farfield-rcs.csv`, with columns:

  - `f (GHz)` : Frequency
  - `exc` : Excitation index (equal to the plane wave index)
  - `theta (deg.)`, `phi (deg.)` : Observation direction
  - `RCS (m^2)` : Total RCS ``\sigma = \sigma_\theta + \sigma_\phi``
  - `RCS_theta (m^2)`, `RCS_phi (m^2)` : RCS of the ``\hat{\boldsymbol{\theta}}`` and
    ``\hat{\boldsymbol{\phi}}`` polarized components of the scattered field,
    ``\sigma_\theta = 4 \pi |r E_{s,\theta}|^2 / E_0^2`` and
    ``\sigma_\phi = 4 \pi |r E_{s,\phi}|^2 / E_0^2``

The RCS in dBsm is ``10 \log_{10}(\sigma / 1\text{ m}^2)``. The monostatic
(backscattering) RCS is obtained for the observation direction ``-\hat{\mathbf{k}}``.

The accuracy of the computed RCS is often limited by the spurious reflections of the
absorbing boundary condition, which pollute the weak scattered field, rather than by the
discretization. It is therefore recommended to truncate the domain with a sphere centered
on the scatterer and to use the exact spherical Dirichlet-to-Neumann boundary condition by
setting `"DtN": true` in
[`config["Boundaries"]["Absorbing"]`](../config/reference.md#config-boundaries-absorbing)
(see [Absorbing boundary](../guide/boundaries.md#Absorbing-(scattering)-boundary)). With it,
the outer sphere can be placed a fraction of a wavelength away from the scatterer. A typical
configuration is:

```json
"Domains":
{
  "Materials": [{"Attributes": [1]}],
  "PlaneWave":
  [
    {
      "Index": 1,
      "Direction": "-Z",
      "Polarization": "+X",
      "Amplitude": 1.0
    }
  ]
},
"Boundaries":
{
  "PEC": {"Attributes": [2]},
  "Absorbing": {"Attributes": [1], "DtN": true},
  "Postprocessing":
  {
    "FarField": {"Attributes": [1], "NSample": 1000}
  }
}
```

The [sphere RCS example](../examples/sphere_rcs.md) validates this capability against the
Mie series for perfectly conducting and dielectric spheres.
