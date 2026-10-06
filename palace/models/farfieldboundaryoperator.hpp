// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#ifndef PALACE_MODELS_FARFIELD_BOUNDARY_OPERATOR_HPP
#define PALACE_MODELS_FARFIELD_BOUNDARY_OPERATOR_HPP

#include <complex>
#include <memory>
#include <vector>
#include <mfem.hpp>
#include "linalg/operator.hpp"
#include "models/sphericaldtn.hpp"

namespace palace
{

class FiniteElementSpace;
class IoData;
class MaterialOperator;
class MaterialPropertyCoefficient;

namespace config
{

struct FarfieldBoundaryData;

}  // namespace config

enum class ProblemType : char;

//
// A class handling farfield, or absorbing, boundaries.
//
class FarfieldBoundaryOperator
{
private:
  // Reference to material property data (not owned).
  const MaterialOperator &mat_op;

  // List of all absorbing boundary condition attributes.
  mfem::Array<int> farfield_attr;

  // First- or second-order absorbing boundary condition.
  int order;

  // Optional exact DtN correction for a spherical absorbing boundary, and the (background)
  // material index adjacent to it.
  std::unique_ptr<SphericalDtNData> dtn_data;
  int dtn_max_degree = 0, dtn_mat = -1;
  void SetUpDtN(const config::FarfieldBoundaryData &farfield, ProblemType problem_type,
                const mfem::ParMesh &mesh);

  mfem::Array<int> SetUpBoundaryProperties(const config::FarfieldBoundaryData &farfield,
                                           ProblemType problem_type,
                                           const mfem::ParMesh &mesh);

public:
  FarfieldBoundaryOperator(const config::FarfieldBoundaryData &farfield,
                           ProblemType problem_type, const MaterialOperator &mat_op,
                           const mfem::ParMesh &mesh);
  FarfieldBoundaryOperator(const IoData &iodata, const MaterialOperator &mat_op,
                           const mfem::ParMesh &mesh);

  // Returns array of farfield BC attributes.
  const auto &GetAttrList() const { return farfield_attr; }

  // Returns order of absorbing BC approximation.
  int GetOrder() const { return order; }

  // Returns whether the exact spherical DtN boundary condition is used.
  bool HasDtN() const { return dtn_data != nullptr; }
  const SphericalDtNData *GetDtNData() const { return dtn_data.get(); }

  // Maximum vector spherical harmonic degree of the DtN boundary condition at frequency ω.
  int GetDtNMaxDegree(double omega) const;

  // Get the low-rank DtN correction to the first-order absorbing boundary condition at
  // frequency ω (null if not using the DtN boundary condition). The rows and columns of the
  // given essential true dofs are eliminated.
  std::unique_ptr<ComplexOperator> GetDtNOperator(double omega,
                                                  FiniteElementSpace &nd_fespace,
                                                  const mfem::Array<int> *dbc_tdof_list);

  // Add contributions to system matrices from first- or second-order absorbing boundary
  // condition.
  void AddDampingBdrCoefficients(double coeff, MaterialPropertyCoefficient &fb);
  void AddExtraSystemBdrCoefficients(double omega, MaterialPropertyCoefficient &dfbr,
                                     MaterialPropertyCoefficient &dfbi);

  // Complex-ω overload, used by the eigenmode nonlinear solve where the second-order
  // absorbing-BC coefficient i·(0.5/ω) is evaluated at the genuinely complex eigenvalue
  // (ω = -i·λ). For real ω (imag = 0) this produces the same (dfbr, dfbi) split as the
  // double overload (the whole term on dfbi).
  void AddExtraSystemBdrCoefficients(std::complex<double> omega,
                                     MaterialPropertyCoefficient &dfbr,
                                     MaterialPropertyCoefficient &dfbi);

  // Add the ω-independent boundary curl-curl coefficient of the second-order absorbing
  // BC (the μ⁻¹·c₀ normal-projected term that the real-ω stamping scales by 0.5/ω),
  // scaled by `coeff`. Pulled out so both the real-ω and complex-ω overloads share the
  // material-coefficient construction. No-op when order < 2 or there are no farfield
  // attributes.
  void AddExtraSystemBoundaryCurlCurlBdrCoefficients(double coeff,
                                                     MaterialPropertyCoefficient &df) const;
};

}  // namespace palace

#endif  // PALACE_MODELS_FARFIELD_BOUNDARY_OPERATOR_HPP
