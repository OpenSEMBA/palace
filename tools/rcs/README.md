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

## Benchmark: accuracy vs DOFs and CPU time

`benchmark_sphere.py` runs the scattered-field sphere problem over ten
frequencies (`ka = 0.5 ... 5.0`) and three finite element orders (`p = 1, 2, 3`)
on a wavelength-scaled spherical boundary (`R = a + 1.5*lambda`,
`hmax = lambda/8`, order-2 curved elements, one mesh per frequency shared by all
orders). All solves use `GMRES Tol 1e-6`, `MaxIts 1000`, 16 MPI ranks, and run
sequentially. Full metrics: `benchmark_sphere.csv`.

| order | ND DOFs | pattern RMS (dB) | wall time (s) | peak memory (GB) | GMRES iterations |
|---|---|---|---|---|---|
| p=1 | 109k-250k | 1.04-2.06 | 22-94 | 3.3-7.7 | 306-547 |
| p=2 | 582k-1.34M | 0.13-0.44 | 59-223 | 6.2-15.8 | 172-277 |
| p=3 | 1.69M-3.89M | 0.04-0.34 | 213-1036 | 12.1-33.3 | 148-262 |

Observations:

- `p=1` is dispersion-limited at `h = lambda/8` (about 1-2 dB RMS roughly
  independent of frequency); it is by far the cheapest (~100-250k DOFs, seconds
  to 1.5 min).
- `p=2` gives roughly an order of magnitude better accuracy for ~5-6x the DOFs;
  `p=3` adds another ~2-3x for ~3x the DOFs, with diminishing returns at high
  `ka` (the `ka = 3` and `4.5` points are slightly worse than `p=2`, likely
  near-null sensitivity rather than a monotone convergence failure).
- GMRES iterations decrease with order (about 300-550 at `p=1` versus 150-280 at
  `p=3`), so higher-order cost is dominated by assembly, preconditioner setup,
  and memory rather than iteration count.
- DOFs scale roughly linearly in `ka` at fixed order (the wavelength-scaled
  boundary keeps the element count nearly constant); wall time grows faster
  because the iteration count grows with `ka` (`p=2`: 175 iterations at `ka=1`
  to 277 at `ka=5`).

## Dielectric sphere validation

`dielectric_sphere_study.py` validates the scattered-field **volume contrast
sources** with a homogeneous dielectric sphere (`a = 0.5 m`), lossless
(`eps_r = 4`) and lossy (`eps_r = 4 - 0.1i`, set via `LossTan = 0.025`), at
`ka = 1, 2, 3`. Reference: `mie_dielectric.py` (Bohren-Huffman series with
complex refractive index).

| ka | case | monostatic error (dB) | pattern RMS (dB) | pattern max (dB) |
|---|---|---|---|---|
| 1 | lossless | +0.30 | 0.25 | 0.30 |
| 1 | lossy | +0.31 | 0.25 | 0.31 |
| 2 | lossless | +0.26 | 0.24 | 0.35 |
| 2 | lossy | +0.26 | 0.22 | 0.32 |
| 3 | lossless | +0.59 | 0.44 | 1.04 |
| 3 | lossy | +0.59 | 0.46 | 1.20 |

The same result is obtained in serial and parallel (4 and 16 ranks). The
formulation supports arbitrary shaped, piecewise-homogeneous, complex
(nonmagnetic) dielectrics; magnetic contrast and dispersive (Lorentz) material
models are future work.

## Figures

`figures/` contains reference images generated by the notebooks:

- `sphere_monostatic_mie.png` — monostatic `sigma/(pi a^2)` vs `ka` (the classic
  Mie theory figure, with Rayleigh and optical asymptotes).
- `sphere_monostatic_palace_vs_mie.png` — Palace points at `ka = 0.5, 1, 2, 3`
  for a 1 m PEC sphere over the Mie curve (scattered-field formulation; all
  points within 0.36 dB).
- `sphere_bistatic_1m_300MHz_polar.png`, `..._db.png` — bistatic patterns of a
  1 m PEC sphere at 300 MHz, Palace vs Mie.
- `sphere_rcs_pattern_3d.png` — 3D dB-scaled RCS pattern (Mie, full sphere).
- `sphere_scattered_slice_y0.png` — scattered field `|E_scat|` slice through a
  1 m PEC sphere at 300 MHz (from a coarse visualization run).
- `bench_sphere_accuracy_vs_dofs.png`, `bench_sphere_accuracy_vs_time.png`,
  `bench_sphere_dofs_time_vs_ka.png` — accuracy vs DOFs, accuracy vs wall time,
  and cost vs frequency for the `p = 1, 2, 3` benchmark.
- `dielectric_sphere_palace_vs_mie.png` — bistatic patterns of the dielectric
  sphere (lossless and lossy) vs the Mie series at `ka = 1, 2, 3`.
