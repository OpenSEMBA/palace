// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include "farfieldboundaryoperator.hpp"

#include <cmath>
#include <limits>
#include <set>
#include "fem/fespace.hpp"
#include "linalg/densematrix.hpp"
#include "models/materialoperator.hpp"
#include "utils/communication.hpp"
#include "utils/geodata.hpp"
#include "utils/iodata.hpp"
#include "utils/prettyprint.hpp"

namespace palace
{

FarfieldBoundaryOperator::FarfieldBoundaryOperator(const IoData &iodata,
                                                   const MaterialOperator &mat_op,
                                                   const mfem::ParMesh &mesh)
  : FarfieldBoundaryOperator(iodata.boundaries.farfield, iodata.problem.type, mat_op, mesh)
{
}

FarfieldBoundaryOperator::FarfieldBoundaryOperator(
    const config::FarfieldBoundaryData &farfield, ProblemType problem_type,
    const MaterialOperator &mat_op, const mfem::ParMesh &mesh)
  : mat_op(mat_op), farfield_attr(SetUpBoundaryProperties(farfield, problem_type, mesh))
{
  // Print out BC info for all farfield attributes.
  if (farfield_attr.Size())
  {
    Mpi::Print("\nConfiguring Robin absorbing BC (order {:d}) at attributes:\n", order);
    std::sort(farfield_attr.begin(), farfield_attr.end());
    utils::PrettyPrint(farfield_attr);
  }
  SetUpDtN(farfield, problem_type, mesh);
}

void FarfieldBoundaryOperator::SetUpDtN(const config::FarfieldBoundaryData &farfield,
                                        ProblemType problem_type, const mfem::ParMesh &mesh)
{
  if (!farfield.dtn || farfield_attr.Size() == 0)
  {
    return;
  }
  MFEM_VERIFY(problem_type == ProblemType::DRIVEN,
              "Spherical DtN absorbing boundaries are only available for driven "
              "simulations!");
  MFEM_VERIFY(order == 1, "Spherical DtN absorbing boundaries require \"Order\": 1!");

  // The medium adjacent to the absorbing boundary must be unique and isotropic.
  const auto bdr_attr_mat = mat_op.GetBdrAttributeToMaterial();
  int mat_range[2] = {std::numeric_limits<int>::max(), std::numeric_limits<int>::max()};
  for (auto attr : mat_op.GetCeedBdrAttributes(farfield_attr))
  {
    const int k = bdr_attr_mat[attr - 1];
    if (k >= 0)
    {
      mat_range[0] = std::min(mat_range[0], k);
      mat_range[1] = std::min(mat_range[1], -k);
    }
  }
  Mpi::GlobalMin(2, mat_range, mesh.GetComm());
  MFEM_VERIFY(mat_range[0] < std::numeric_limits<int>::max() &&
                  mat_range[0] == -mat_range[1],
              "Spherical DtN absorbing boundaries must be adjacent to a single material!");
  dtn_mat = mat_range[0];
  const auto &muinv = mat_op.GetInvPermeability()(dtn_mat);
  const auto &eps = mat_op.GetPermittivityReal()(dtn_mat);
  MFEM_VERIFY(muinv(0, 0) == muinv(1, 1) && muinv(0, 0) == muinv(2, 2) &&
                  eps(0, 0) == eps(1, 1) && eps(0, 0) == eps(2, 2),
              "Spherical DtN absorbing boundaries require an isotropic adjacent material!");

  dtn_max_degree = farfield.dtn_max_degree;
  dtn_data = std::make_unique<SphericalDtNData>(mesh, farfield_attr, farfield.dtn_center);
  const auto &c = dtn_data->GetCenter();
  Mpi::Print(" Exact spherical DtN boundary condition: center = ({:.3e}, {:.3e}, {:.3e}), "
             "radius = {:.3e} (nondimensional), maximum degree = {}\n",
             c[0], c[1], c[2], dtn_data->GetRadius(),
             (dtn_max_degree > 0) ? std::to_string(dtn_max_degree) : "automatic");
}

int FarfieldBoundaryOperator::GetDtNMaxDegree(double omega) const
{
  MFEM_VERIFY(dtn_data, "No spherical DtN boundary condition!");
  if (dtn_max_degree > 0)
  {
    return dtn_max_degree;
  }
  const double k = omega * std::sqrt(mat_op.GetPermittivityReal()(dtn_mat)(0, 0) /
                                     mat_op.GetInvPermeability()(dtn_mat)(0, 0));
  return static_cast<int>(std::ceil(k * dtn_data->GetRadius())) + 6;
}

std::unique_ptr<ComplexOperator>
FarfieldBoundaryOperator::GetDtNOperator(double omega, FiniteElementSpace &nd_fespace,
                                         const mfem::Array<int> *dbc_tdof_list)
{
  if (!dtn_data)
  {
    return {};
  }
  // The DtN correction to the first-order condition is μ⁻¹ Σ_j (λ_j - ik) w_j w_jᵀ, where
  // ik/μ = iω/Z is the coefficient of the first-order absorbing boundary condition.
  const double muinv = mat_op.GetInvPermeability()(dtn_mat)(0, 0);
  const double k = omega * std::sqrt(mat_op.GetPermittivityReal()(dtn_mat)(0, 0) / muinv);
  const double R = dtn_data->GetRadius();
  const int n_max = GetDtNMaxDegree(omega);
  dtn_data->SetMaxDegree(n_max, nd_fespace);
  const int n_modes = dtn::NumModes(n_max);
  std::vector<std::complex<double>> g(n_modes);
  for (int n = 1; n <= n_max; n++)
  {
    const std::complex<double> g_te =
        muinv * (dtn::EigenvalueTE(n, k, R) - std::complex<double>(0.0, k));
    const std::complex<double> g_tm =
        muinv * (dtn::EigenvalueTM(n, k, R) - std::complex<double>(0.0, k));
    for (int m = -n; m <= n; m++)
    {
      const int j = n * n - 1 + m + n;
      g[2 * j] = g_te;
      g[2 * j + 1] = g_tm;
    }
  }
  return std::make_unique<SphericalDtNOperator>(*dtn_data, n_modes, std::move(g),
                                                nd_fespace.GetTrueVSize(), dbc_tdof_list);
}

mfem::Array<int> FarfieldBoundaryOperator::SetUpBoundaryProperties(
    const config::FarfieldBoundaryData &farfield, ProblemType problem_type,
    const mfem::ParMesh &mesh)
{
  // Check that impedance boundary attributes have been specified correctly.
  int bdr_attr_max = mesh.bdr_attributes.Size() ? mesh.bdr_attributes.Max() : 0;
  mfem::Array<int> bdr_attr_marker;
  if (!farfield.empty())
  {
    bdr_attr_marker.SetSize(bdr_attr_max);
    bdr_attr_marker = 0;
    for (auto attr : mesh.bdr_attributes)
    {
      bdr_attr_marker[attr - 1] = 1;
    }
    std::set<int> bdr_warn_list;
    for (auto attr : farfield.attributes)
    {
      if (attr <= 0 || attr > bdr_attr_max || !bdr_attr_marker[attr - 1])
      {
        bdr_warn_list.insert(attr);
      }
      if (!bdr_warn_list.empty())
      {
        Mpi::Print("\n");
        Mpi::Warning(
            "Unknown absorbing boundary attributes!\nSolver will just ignore them!");
        utils::PrettyPrint(bdr_warn_list, "Boundary attribute list:");
        Mpi::Print("\n");
      }
    }
  }

  // Set the order of the farfield boundary condition.
  order = farfield.order;

  // Mark selected boundary attributes from the mesh as farfield.
  mfem::Array<int> farfield_bcs;
  farfield_bcs.Reserve(static_cast<int>(farfield.attributes.size()));
  for (auto attr : farfield.attributes)
  {
    if (attr <= 0 || attr > bdr_attr_max || !bdr_attr_marker[attr - 1])
    {
      continue;  // Can just ignore if wrong
    }
    farfield_bcs.Append(attr);
  }
  MFEM_VERIFY(farfield_bcs.Size() == 0 || order < 2 ||
                  problem_type == ProblemType::DRIVEN ||
                  problem_type == ProblemType::EIGENMODE ||
                  problem_type == ProblemType::BOUNDARYMODE,
              "Second-order farfield boundaries are only available for frequency "
              "domain simulations!");
  return farfield_bcs;
}

void FarfieldBoundaryOperator::AddDampingBdrCoefficients(double coeff,
                                                         MaterialPropertyCoefficient &fb)
{
  // First-order absorbing boundary condition.
  if (farfield_attr.Size())
  {
    MaterialPropertyCoefficient invz0_func(mat_op.GetBdrAttributeToMaterial(),
                                           mat_op.GetInvImpedance());
    invz0_func.RestrictCoefficient(mat_op.GetCeedBdrAttributes(farfield_attr));
    fb.AddCoefficient(invz0_func.GetAttributeToMaterial(),
                      invz0_func.GetMaterialProperties(), coeff);
  }
}

void FarfieldBoundaryOperator::AddExtraSystemBoundaryCurlCurlBdrCoefficients(
    double coeff, MaterialPropertyCoefficient &df) const
{
  // Contribution for second-order absorbing BC. See Jin Section 9.3 for reference. The β
  // coefficient for the second-order ABC is 1/(2ik+2/r). Taking the radius of curvature
  // as infinity (plane wave scattering), the r-dependence vanishes and the contribution
  // is purely imaginary. Multiplying through by μ⁻¹ we get the material coefficient to ω
  // as 1 / (μ √(με)). Also, this implementation ignores the divergence term ∇⋅Eₜ, as
  // COMSOL does as well. The ω-dependence enters only through the scalar `coeff` (= 0.5/ω
  // for the imaginary slot in the real-ω stamping), so the material coefficient is pulled
  // out here and reused by both the real-ω and complex-ω overloads.
  if (farfield_attr.Size() && order > 1)
  {
    mfem::DenseTensor muinvc0 =
        linalg::Mult(mat_op.GetInvPermeability(), mat_op.GetLightSpeed());
    MaterialPropertyCoefficient muinvc0_func(mat_op.GetBdrAttributeToMaterial(), muinvc0);
    muinvc0_func.RestrictCoefficient(mat_op.GetCeedBdrAttributes(farfield_attr));

    // Instead getting the correct normal of farfield boundary elements, just pick the
    // the first element normal. This is fine as long as the farfield material properties
    // are not anisotropic.
    mfem::Vector normal(mat_op.SpaceDimension());
    normal = 0.0;
    normal(0) = 1.0;
    muinvc0_func.NormalProjectedCoefficient(normal);

    df.AddCoefficient(muinvc0_func.GetAttributeToMaterial(),
                      muinvc0_func.GetMaterialProperties(), coeff);
  }
}

void FarfieldBoundaryOperator::AddExtraSystemBdrCoefficients(
    double omega, MaterialPropertyCoefficient &dfbr, MaterialPropertyCoefficient &dfbi)
{
  // Real-ω second-order ABC: the term is i·(0.5/ω)·(curl-curl). At real ω the scalar
  // 0.5/ω is real, so the whole contribution is imaginary and lands on dfbi.
  AddExtraSystemBoundaryCurlCurlBdrCoefficients(0.5 / omega, dfbi);
}

void FarfieldBoundaryOperator::AddExtraSystemBdrCoefficients(
    std::complex<double> omega, MaterialPropertyCoefficient &dfbr,
    MaterialPropertyCoefficient &dfbi)
{
  // Complex-ω second-order ABC for the eigenmode nonlinear solve. The contribution is
  // i·(0.5/ω)·(curl-curl); with ω complex, s = 0.5/ω is complex and i·s splits as
  // i·(s.real() + i·s.imag()) = -s.imag() + i·s.real(), so the real part -s.imag() goes
  // on dfbr and the imaginary part s.real() on dfbi. For real ω (s.imag() == 0) this
  // reduces to the double overload (whole term on dfbi).
  const std::complex<double> s = 0.5 / omega;
  AddExtraSystemBoundaryCurlCurlBdrCoefficients(s.real(), dfbi);
  if (s.imag() != 0.0)
  {
    AddExtraSystemBoundaryCurlCurlBdrCoefficients(-s.imag(), dfbr);
  }
}

}  // namespace palace
