// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#ifndef PALACE_MODELS_PLANEWAVE_OPERATOR_HPP
#define PALACE_MODELS_PLANEWAVE_OPERATOR_HPP

#include <array>
#include <complex>
#include <map>
#include <mfem.hpp>
#include "linalg/vector.hpp"

namespace palace
{

class MaterialOperator;
class SumVectorCoefficient;

namespace config
{
struct PlaneWaveData;
}  // namespace config

//
// Runtime data for a single incident plane wave.
//
class PlaneWaveData
{
public:
  // Excitation index (0 if not excited).
  int excitation;

  // Boundary attributes where the incident wave enters the domain.
  mfem::Array<int> attr_list;

  // Unit propagation direction k̂ and electric field polarization p.
  mfem::Vector direction, polarization;

  // Complex nondimensional amplitude E0 * exp(i phase).
  std::complex<double> amplitude;

  // Phase reference point in internal length units.
  mfem::Vector origin;

  // Per-boundary-attribute local wave speed and inverse impedance (internal units).
  mfem::Array<double> attr_c, attr_invz;

  PlaneWaveData(const config::PlaneWaveData &data, const MaterialOperator &mat_op,
                const mfem::ParMesh &mesh);

  [[nodiscard]] bool HasExcitation() const { return excitation != 0; }

  // Evaluate the incident electric field phasor (internal units) at a physical point x,
  // using the local wave speed for the given boundary attribute (or vacuum if attr <= 0).
  void GetElectricField(const mfem::Vector &x, double omega, int attr,
                        std::complex<double> E[3]) const;
  void GetElectricField(const mfem::Vector &x, double omega,
                        std::complex<double> E[3]) const;

  // Evaluate the incident electric field using the background medium (internal vacuum,
  // c0 = 1) independent of the local material. Used by the scattered-field formulation,
  // where the incident field is the free-space plane wave and the material response is
  // carried by the scattered field.
  void GetBackgroundElectricField(const mfem::Vector &x, double omega,
                                  std::complex<double> E[3]) const;

private:
  void GetElectricFieldImpl(const mfem::Vector &x, double omega, double c,
                            std::complex<double> E[3]) const;
};

//
// A class handling incident plane wave excitation on absorbing boundaries.
//
class PlaneWaveOperator
{
private:
  std::map<int, PlaneWaveData> waves;

  void SetUpBoundaryProperties(const std::map<int, config::PlaneWaveData> &planewave,
                               const MaterialOperator &mat_op, const mfem::ParMesh &mesh);
  void PrintBoundaryInfo(const mfem::ParMesh &mesh);

public:
  PlaneWaveOperator(const std::map<int, config::PlaneWaveData> &planewave,
                    const MaterialOperator &mat_op, const mfem::ParMesh &mesh);

  auto begin() const { return waves.begin(); }
  auto end() const { return waves.end(); }
  auto Size() const { return waves.size(); }
  bool Empty() const { return waves.empty(); }

  const PlaneWaveData &GetWave(int idx) const;

  // Add the boundary excitation coefficient pair Re/Im{(1/Z) E_inc} for the given
  // excitation index (first-order absorbing boundary contribution).
  void AddExcitationBdrCoefficients(int excitation_idx, double omega,
                                    SumVectorCoefficient &fbr,
                                    SumVectorCoefficient &fbi) const;

  // Add the incident electric field coefficient pair Re/Im{E_inc} (without the 1/Z
  // factor), used for the second-order absorbing boundary correction and, with
  // scale = -1, for the tangential PEC boundary condition in the scattered-field
  // formulation. If `attr_scale` is provided, the field is evaluated in the background
  // medium and multiplied by the per-domain-attribute complex scale (used for the
  // dielectric contrast sources in the scattered-field formulation).
  void AddFieldCoefficients(int excitation_idx, double omega, SumVectorCoefficient &fr,
                            SumVectorCoefficient &fi, double scale = 1.0,
                            const std::vector<std::complex<double>> *attr_scale =
                                nullptr,
                            bool background = false) const;

  // Return the first plane wave with the given excitation index, or nullptr if none.
  const PlaneWaveData *GetExcitationWave(int excitation_idx) const;

  // Boundary attribute marker for the given excitation index.
  mfem::Array<int> GetExcitationBdrMarker(int excitation_idx) const;

  // Combined boundary attribute marker for all configured plane waves.
  mfem::Array<int> GetAttrMarker() const;
};

}  // namespace palace

#endif  // PALACE_MODELS_PLANEWAVE_OPERATOR_HPP
