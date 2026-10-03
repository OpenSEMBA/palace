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
a system.

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

## Incident Plane Waves and Scattering

For scattering and radar cross section (RCS) problems, *Palace* can illuminate a
structure with an incident plane wave and extract the scattered far field. The
incident field is injected through the [`Absorbing` boundary
conditions](../config/reference.md#config-boundaries-absorbing) as a
non-homogeneous radiation condition: the scattered field satisfies the
homogeneous absorbing boundary condition, which corresponds to the classic
total-field formulation of scattering with absorbing boundary conditions (see
Jin, *The Finite Element Method in Electromagnetics*, Section 9.3, and Monk,
*Finite Element Methods for Maxwell's Equations*, Eq. (1.25)).

A plane wave is configured under `"Boundaries"`:

```json
"PlaneWave":
[
    {
        "Index": 1,
        "Attributes": [1],
        "Excitation": 1,
        "Direction": [0.0, 0.0, 1.0],
        "Polarization": [1.0, 0.0, 0.0],
        "Amplitude": 1.0
    }
]
```

  - `"Attributes"` must be absorbing boundary attributes through which the
    incident wave enters the domain.
  - `"Direction"` is the propagation direction `k̂` (normalized internally), and
    `"Polarization"` is the electric field direction, which must be orthogonal
    to `"Direction"`.
  - `"Amplitude"` is the incident electric field magnitude in V/m; the RCS is
    independent of this value for linear problems. `"Phase"` (radians) and
    `"ReferencePoint"` (mesh length units) optionally define the phase
    reference.
  - The `"Excitation"` index follows the same conventions as ports, so multiple
    plane waves (for example, different incidence angles or polarizations) can be
    swept in a single simulation.

Plane waves are only available for frequency domain driven simulations, and the
absorbing boundary should be placed in a homogeneous, isotropic medium (vacuum
for the usual free-space scattering setup).

To extract the scattered far field and RCS, set `"Scattered"` and optionally
`"Monostatic"` in the `"FarField"` postprocessing block:

```json
"Postprocessing":
{
    "FarField":
    {
        "Attributes": [1],
        "ThetaPhis": [[90, 0], [90, 90], [90, 180]],
        "Scattered": true,
        "Monostatic": true
    }
}
```

  - With `"Scattered": true`, the incident plane wave contribution is subtracted
    from the total field on the integration surface, so `farfield-rE.csv`
    contains the scattered far field and a new `farfield-rcs.csv` file is
    written with the bistatic RCS in m² and dBsm,
    `σ = 4π |r E_scat|² / |E_inc|²`.
  - With `"Monostatic": true`, the scattered far field is also evaluated in the
    backscatter direction `-k̂` of each excited plane wave and added to
    `farfield-rcs.csv`.

For a plane wave incident on a PEC sphere, the scattered far field computed this
way agrees with the analytic Mie series, which provides a useful validation case.

!!! warning "Limitations"

    The accuracy of the scattered field is limited by the reflection of the
    absorbing boundary condition. Use a second-order `"Order": 2` absorbing
    boundary, place it sufficiently far from the scatterer, and verify
    convergence for low-RCS (deep null) directions. Perfectly matched layers are
    not yet implemented.

