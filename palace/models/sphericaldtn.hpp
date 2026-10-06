// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#ifndef PALACE_MODELS_SPHERICAL_DTN_HPP
#define PALACE_MODELS_SPHERICAL_DTN_HPP

#include <array>
#include <complex>
#include <memory>
#include <vector>
#include <Eigen/Dense>
#include <mfem.hpp>
#include "linalg/operator.hpp"
#include "linalg/vector.hpp"

namespace palace
{

class FiniteElementSpace;

namespace dtn
{

// Number of vector spherical harmonic modes (TE and TM, all orders m) of degree 1 to n_max.
constexpr int NumModes(int n_max)
{
  return 2 * n_max * (n_max + 2);
}

// Surface gradients ∇ₛY = θ̂ ∂Y/∂θ + φ̂ (1/sin θ) ∂Y/∂φ of the real orthonormal spherical
// harmonics Y_nm (∮ Y_nm Y_n'm' dΩ = δ_nn' δ_mm'), for degrees n = 1, ..., n_max and orders
// m = -n, ..., n. The (θ̂, φ̂) components are returned in grad, ordered by n then m, with
// n² - 1 + (m + n) the index of (n, m).
void SphericalHarmonicSurfaceGradients(int n_max, double theta, double phi,
                                       std::vector<std::array<double, 2>> &grad);

// Real orthonormal spherical harmonics Y_nm, ordered as above, for n = 0, ..., n_max (used
// for testing).
void SphericalHarmonics(int n_max, double theta, double phi, std::vector<double> &Y);

// Eigenvalues λ of the exact Dirichlet-to-Neumann map r̂ × ∇ × E = λ Eₜ for outgoing TE
// (∇ₜ·Eₜ = 0) and TM ((∇ × E)·r̂ = 0) spherical modes of degree n at radius R, with
// wavenumber k, for the exp(+iωt) time convention: λ_TE = -k ζₙ'(kR) / ζₙ(kR) and
// λ_TM = k ζₙ(kR) / ζₙ'(kR), with ζₙ(x) = x hₙ⁽²⁾(x) the Riccati-Hankel function. As
// kR → ∞, λ → ik, the first-order absorbing boundary condition.
std::complex<double> EigenvalueTE(int n, double k, double R);
std::complex<double> EigenvalueTM(int n, double k, double R);

}  // namespace dtn

//
// Data for the exact Dirichlet-to-Neumann (DtN) boundary condition on a spherical
// absorbing boundary of radius R. On the sphere, the vector spherical harmonics
//   Ψ_TE,nm = r̂ × ∇ₛY_nm / (R √(n(n+1))) and Ψ_TM,nm = ∇ₛY_nm / (R √(n(n+1)))
// form an orthonormal basis for tangential fields, and outgoing fields satisfy
// r̂ × ∇ × E = Σ λ_j (∮ Ψ_j·E dS) Ψ_j exactly. The first-order absorbing boundary condition
// approximates all λ_j by ik, so the exact condition is obtained from it by adding the
// low-rank correction
//              A_DtN(ω) = μ⁻¹ Σ_j (λ_j(ω) - ik) w_j w_jᵀ ,  (w_j)_i = ∮ Ψ_j·φ_i dS ,
// to the system matrix, where the sum runs over all modes of degree n ≤ n_max.
//
class SphericalDtNData
{
private:
  MPI_Comm comm;

  // Sphere center and radius (nondimensional).
  std::array<double, 3> center;
  double radius;

  // Maximum degree for which the mode vectors have been computed (0 if none).
  int n_max = 0;

  // Local true dofs with support on the absorbing boundary, and the mode vectors restricted
  // to these rows (column j is w_j, ordered as TE and TM for each (n, m)).
  mfem::Array<int> rows;
  Eigen::MatrixXd W;

  // Boundary attribute marker for the absorbing boundary.
  mfem::Array<int> attr_marker;

public:
  SphericalDtNData(const mfem::ParMesh &mesh, const mfem::Array<int> &attr_list,
                   const std::vector<double> &center);

  double GetRadius() const { return radius; }
  const auto &GetCenter() const { return center; }
  int GetMaxDegree() const { return n_max; }
  const auto &GetRows() const { return rows; }
  const auto &GetModeVectors() const { return W; }
  MPI_Comm GetComm() const { return comm; }

  // Compute (or recompute, if a larger degree is requested) the mode vectors for all
  // degrees up to n.
  void SetMaxDegree(int n, FiniteElementSpace &nd_fespace);
};

//
// Matrix-free operator applying the DtN correction y += a Σ_j g_j w_j (w_jᵀ x), with
// optional elimination of essential boundary condition true dofs (zero rows and columns).
// All mode projections are reduced with a single MPI reduction.
//
class SphericalDtNOperator : public ComplexOperator
{
private:
  const SphericalDtNData &data;
  std::vector<std::complex<double>> g;
  std::vector<bool> keep;  // Per-row: false for eliminated (essential) rows
  mutable Eigen::VectorXd xr, xi, c;

public:
  SphericalDtNOperator(const SphericalDtNData &data, int n_modes,
                       std::vector<std::complex<double>> &&g, int height,
                       const mfem::Array<int> *dbc_tdof_list);

  void Mult(const ComplexVector &x, ComplexVector &y) const override;
  void AddMult(const ComplexVector &x, ComplexVector &y,
               const std::complex<double> a = 1.0) const override;
};

}  // namespace palace

#endif  // PALACE_MODELS_SPHERICAL_DTN_HPP
