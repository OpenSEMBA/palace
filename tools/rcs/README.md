# RCS validation tools

Python helpers for validating Palace plane wave excitation, scattered far-field
output, and ParaView-compatible field visualization. These are research and
validation scripts; they are not part of the Palace build or the CI test suite.

## Files

- `sphere_mesh.py` — generate a PEC sphere with Gmsh. The inner surface is
  physical tag 2 (PEC) and the outer boundary is physical tag 1 (absorbing).
  Both a box (default) and a spherical outer boundary are available; second-order
  curved elements are the default, with `ORDER = 1` for the straight-sided
  comparison arm. For the spherical boundary set
  `Boundaries.Absorbing.Radius` to the boundary radius to activate the
  Bayliss-Turkel curvature term in the second-order ABC.
- `mie_pec.py` — Mie series for a PEC sphere following Ruck et al., *Radar Cross
  Section Handbook*. Time convention `exp(j*omega*t)`, incident wave along `-z`
  (backscatter at `theta = 0`), electric field along `x`. Also provides
  `monostatic_normalized(ka)` for the classic normalized monostatic RCS curve.
- `compare_rcs.py` — compare a Palace `farfield-rcs.csv` against `mie_pec.py`.
- `vtu_scattered.py` — subtract the analytic incident plane wave from a Palace
  ParaView field (`E_real`/`E_imag`) and write a single-file VTU with
  `E_scat_real`, `E_scat_imag`, `E_total_mag`, `E_scat_mag`. Uses the `vtk`
  package because Palace writes VTK XML version 2.2 (meshio cannot read it).
  Accepts a `.vtu`, `.pvtu`, or `.pvd`.
- `pattern_vtp.py` — convert `farfield-rcs.csv` into a dB-scaled 3D point-cloud
  pattern (`.vtp`) for ParaView, with `RCS`, `RCS_dB` and `radius` scalars.
- `render_paraview.py` — offscreen ParaView renderer for slices/patterns
  (`pvbatch3.12`), when a working ParaView Python environment is available.
- `plot_field_slice.py` — cut a field VTU with a plane and save a matplotlib
  PNG colored by a scalar (works without a ParaView Python interpreter).
- `notebooks/sphere_rcs_study.py` — jupytext notebook: monostatic Mie figure,
  bistatic 1 m PEC sphere at 300 MHz (Palace vs Mie), ParaView exports.

## Dependencies

```sh
python3 -m venv venv && ./venv/bin/pip install -r requirements.txt
```

## Example: PEC sphere

```sh
python3 sphere_mesh.py sphere.msh 1.0 2.2 0.028 0.09
palace sphere.json
python3 compare_rcs.py postpro/sphere/farfield-rcs.csv
```

A `sphere.json` uses `Boundaries.PlaneWave` with `Direction = [0, 0, 1]` (the propagation direction),
`Polarization = [1, 0, 0]`; the monostatic (backscatter) direction is
`theta = 180`, and `Postprocessing.FarField` with `Scattered` and `Monostatic`
enabled. With a well-resolved mesh (`h ~ lambda/30`) the main bistatic features
agree with Mie to about 1 dB; deep nulls and the forward peak are limited by the
absorbing-boundary reflection and require a PML for higher accuracy.

## Validation status

The scattered-field plane wave formulation (`Boundaries.Absorbing.ScatteredField`)
is validated against the Mie series for a 1 m PEC sphere at `ka = 0.5, 1, 2, 3`.
The setup is a spherical absorbing boundary at `R = 7.28 m` with the
Bayliss-Turkel curvature term (`"Radius"`), second-order curved elements
(`h ~ lambda/10` to `lambda/20`), and quadratic finite elements. Palace points are
compared with the analytic bistatic pattern at 15-degree resolution:

| ka | f (MHz) | monostatic error (dB) | pattern RMS (dB) | pattern max (dB) |
|---|---|---|---|---|
| 0.5 | 23.9 | +0.06 | 0.15 | 0.32 |
| 1.0 | 47.7 | +0.17 | 0.17 | 0.26 |
| 2.0 | 95.4 | +0.08 | 0.08 | 0.14 |
| 3.0 | 143.1 | +0.22 | 0.18 | 0.36 |

All solves use `"MaxIts": 600` (`Tol 1e-6`) so that the fields are fully
converged; the default 100 GMRES iterations are not sufficient for these
problem sizes.

Mesh convergence at `ka = 2` (same boundary and polynomial order, converged
solves):

| hmax | monostatic error (dB) | pattern RMS (dB) | pattern max (dB) |
|---|---|---|---|
| lambda/6.3 | -0.11 | 0.15 | 0.27 |
| lambda/10 | +0.08 | 0.08 | 0.14 |

All results are within **0.36 dB** (pattern RMS 0.08–0.18 dB), far inside the
2 dB validation target, including the deep-null directions that the total-field
formulation could not resolve. The earlier total-field failures (several dB to
tens of dB, frequency dependent) were caused by the absorbing boundary
reflection of the strong incident wave; the scattered-field formulation keeps
the incident wave off the absorbing boundary entirely, so only the (much weaker)
scattered field interacts with it.

The total-field formulation is still the default and is unchanged; it remains
appropriate for port and waveguide excitations. A known discrepancy remains
between the total-field and scattered-field plane wave injections: the
total-field pattern appears mirrored relative to the scattered-field pattern for
the same `Direction`, which points to an inconsistency in the total-field
non-homogeneous absorbing-boundary injection that requires a dedicated
investigation. Use the scattered-field formulation for RCS validation.

## Figures

`figures/` contains reference images generated by the notebooks:

- `sphere_monostatic_mie.png` — monostatic `sigma/(pi a^2)` vs `ka` (the classic
  Mie theory figure, with Rayleigh and optical asymptotes).
- `sphere_monostatic_palace_vs_mie.png` — Palace points at `ka = 0.5, 1, 2, 3`
  for a 1 m PEC sphere over the Mie curve (see the validation-status section;
  only the `ka = 1` point is converged).
- `sphere_bistatic_1m_300MHz_polar.png`, `..._db.png` — bistatic patterns of a
  1 m PEC sphere at 300 MHz, Palace vs Mie.
- `sphere_rcs_pattern_3d.png` — 3D dB-scaled RCS pattern (Mie, full sphere).
- `sphere_scattered_slice_y0.png` — scattered field `|E_scat|` slice through a
  1 m PEC sphere at 300 MHz (from a coarse visualization run).
