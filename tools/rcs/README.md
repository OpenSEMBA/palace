# RCS validation tools

Python helpers for validating Palace plane wave excitation and scattered
far-field / RCS output. These are research and validation scripts; they are not
part of the Palace build or the CI test suite.

## Files

- `sphere_mesh.py` — generate a PEC sphere inside a box (or larger sphere) with
  Gmsh. The inner surface is physical tag 2 (PEC) and the outer boundary is
  physical tag 1 (absorbing).
- `mie_pec.py` — Mie series for a PEC sphere following Ruck et al., *Radar Cross
  Section Handbook*. Time convention `exp(j*omega*t)`, incident wave along +z
  with the electric field along +x, matching Palace's `Direction`/`Polarization`
  convention.
- `compare_rcs.py` — compare a Palace `farfield-rcs.csv` against `mie_pec.py`.
- `almond_mesh.py` — generate the EMCC/NASA almond benchmark geometry with
  Gmsh. The geometry equations and dimensions are verified (length 9.936 in,
  width 3.84 in, thickness 1.28 in), but the volume mesh quality at the sharp
  tips is not yet sufficient for MFEM; use a published benchmark mesh for
  quantitative almond studies until the tips are capped conformally.

## Example: PEC sphere

```sh
python3 sphere_mesh.py sphere.msh 0.159154943 0.6 0.018 0.07
palace sphere.json
python3 compare_rcs.py postpro/sphere/farfield-rcs.csv
```

A reference `sphere.json` uses `Boundaries.PlaneWave` with
`Direction = [0, 0, 1]`, `Polarization = [1, 0, 0]`, and
`Postprocessing.FarField` with `Scattered` and `Monostatic` enabled. For a
well-resolved mesh the bistatic pattern agrees with Mie to better than ~1 dB at
most angles (the forward direction agrees to <0.1 dB); deep co-polarized nulls
show larger relative errors and are limited by the absorbing boundary
reflection.

## NASA almond benchmark

The benchmark (Woo et al., *IEEE Antennas Propagat. Mag.* 35(1), 1993) is a PEC
almond of length 9.936 in with monostatic RCS at 1.19, 7, and 9.92 GHz for
horizontal (HH) and vertical (VV) polarization, azimuth 0-180 degrees. The
reference data are the published measurement curves, or the CC-BY-SA-4.0 Austin
RCS Benchmark Suite (`UTAustinCEMGroup/AustinCEMBenchmarks`, Problem Set IIIA).
Use one `PlaneWave` excitation per azimuth and polarization with
`Postprocessing.FarField.Monostatic = true`; the backscatter RCS for each
excitation is written to `farfield-rcs.csv`.

Notes:

- The sharp almond tips require local mesh refinement and a low-reflection
  truncation boundary (second-order absorbing, or a PML when available) placed
  sufficiently far from the target; the truncation error dominates the
  low-RCS tip-on directions.
- Monostatic sweeps are naturally parallelized as multiple plane wave
  excitations in a single run, which reuses the system matrix across
  right-hand sides.
