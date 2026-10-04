// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#ifndef PALACE_MODELS_PLANE_WAVE_OPERATOR_HPP
#define PALACE_MODELS_PLANE_WAVE_OPERATOR_HPP

#include <array>
#include <complex>
#include <map>
#include <mfem.hpp>
#include "linalg/vector.hpp"

namespace palace
{

class FiniteElementSpace;
class IoData;
class Units;

namespace config
{

struct PlaneWaveData;

}  // namespace config

enum class ProblemType : char;

// Helper class for incident plane wave sources in a model.
class PlaneWaveData
{
public:
  // Plane wave properties (stored in nondimensional form).
  std::array<double, 3> direction;     // Unit propagation direction k̂
  std::array<double, 3> polarization;  // Unit electric field polarization p̂ ⟂ k̂
  double amplitude;                    // Electric field amplitude E₀

  PlaneWaveData(const config::PlaneWaveData &data, const Units &units);

  // Evaluate the incident electric field E_inc(x) = E₀ p̂ exp(-i k k̂·x) at the
  // (nondimensional) point x, for the (nondimensional) background wavenumber k. The phase
  // convention matches the exp(+iωt) time dependence of the frequency domain solver.
  std::array<std::complex<double>, 3> E(const double *x, double k) const;

  // Evaluate the incident magnetic flux density B_inc = (k / ω) k̂ × E_inc(x).
  std::array<std::complex<double>, 3> B(const double *x, double k, double omega) const;
};

//
// A class handling incident plane wave excitations for scattering problems.
//
// Plane waves use a scattered-field formulation expressed in terms of the total field. The
// solver computes the total field E = E_s + E_inc, with homogeneous PEC conditions on E,
// driven by the source A_bg Π(E_inc), where Π is the Nédélec interpolant and A_bg is the
// system operator obtained by replacing every material with the background medium (the
// medium adjacent to the absorbing boundary), keeping only the absorbing boundary
// condition. With this source, the scattered field E_s = E - Π(E_inc) satisfies the
// discrete scattered-field equations exactly: it is sourced only where the system differs
// from the background (inhomogeneous materials, PEC, impedance surfaces, ...) and is the
// field which the absorbing boundary condition acts upon.
//
class PlaneWaveOperator
{
private:
  // Mapping from plane wave (and excitation) index to plane wave data.
  std::map<int, PlaneWaveData> plane_waves;

  void PrintPlaneWaveInfo(const Units &units) const;

public:
  PlaneWaveOperator(const std::map<int, config::PlaneWaveData> &plane_wave,
                    ProblemType problem_type, const Units &units);
  PlaneWaveOperator(const IoData &iodata);

  // Access data structures for plane waves.
  auto begin() const { return plane_waves.begin(); }
  auto end() const { return plane_waves.end(); }
  auto Size() const { return plane_waves.size(); }
  bool Empty() const { return plane_waves.empty(); }

  // Check if the excitation with the given index contains an incident plane wave.
  bool HasExcitation(int excitation_idx) const
  {
    return plane_waves.find(excitation_idx) != plane_waves.end();
  }
  const PlaneWaveData &GetPlaneWave(int excitation_idx) const;

  // Compute the true dof vector of the Nédélec interpolant of the incident electric field
  // for the given excitation and background wavenumber k. If the excitation has no plane
  // wave, the returned vector is zero and the function returns false.
  bool GetIncidentField(int excitation_idx, double k, FiniteElementSpace &nd_fespace,
                        ComplexVector &E_inc) const;
};

}  // namespace palace

#endif  // PALACE_MODELS_PLANE_WAVE_OPERATOR_HPP
