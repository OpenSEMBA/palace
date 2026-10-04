// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include "sphericaldtn.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include "fem/fespace.hpp"
#include "utils/communication.hpp"
#include "utils/geodata.hpp"

namespace palace
{

namespace dtn
{

namespace
{

// Fully normalized associated Legendre functions P̄_n^m(cos θ) (without the Condon-Shortley
// phase), such that Y_nm = P̄_n^|m|(cos θ) T_m(φ) with T_0 = 1, T_m = √2 cos(mφ) for m > 0
// and T_m = √2 sin(|m|φ) for m < 0 are orthonormal on the unit sphere. Returns P[n][m] and,
// for m ≥ 1, S[n][m] = P̄_n^m / sin θ, which is regular at the poles.
void Legendre(int n_max, double theta, std::vector<std::vector<double>> &P,
              std::vector<std::vector<double>> &S)
{
  const double x = std::cos(theta), s = std::sin(theta);
  P.assign(n_max + 2, std::vector<double>(n_max + 2, 0.0));
  S.assign(n_max + 2, std::vector<double>(n_max + 2, 0.0));
  P[0][0] = 1.0 / std::sqrt(4.0 * M_PI);
  for (int m = 0; m <= n_max; m++)
  {
    // Diagonal seed: P̄_m^m = √((2m+1)/(2m)) sin θ P̄_{m-1}^{m-1}, and S_m^m without the
    // sin θ factor.
    if (m > 0)
    {
      const double f = std::sqrt((2.0 * m + 1.0) / (2.0 * m));
      S[m][m] = f * P[m - 1][m - 1];
      P[m][m] = s * S[m][m];
    }
    // Three-term recurrence in n for fixed m, applied to P̄ and S = P̄ / sin θ.
    for (int n = m + 1; n <= n_max + 1; n++)
    {
      const double a = std::sqrt((4.0 * n * n - 1.0) / (double(n) * n - double(m) * m));
      const double b = (n - 1 > m) ? std::sqrt(((n - 1.0) * (n - 1.0) - double(m) * m) /
                                               (4.0 * (n - 1.0) * (n - 1.0) - 1.0))
                                   : 0.0;
      const double P2 = (n - 2 >= m) ? P[n - 2][m] : 0.0;
      const double S2 = (n - 2 >= m) ? S[n - 2][m] : 0.0;
      P[n][m] = a * (x * P[n - 1][m] - b * P2);
      if (m > 0)
      {
        S[n][m] = a * (x * S[n - 1][m] - b * S2);
      }
    }
  }
}

}  // namespace

void SphericalHarmonics(int n_max, double theta, double phi, std::vector<double> &Y)
{
  std::vector<std::vector<double>> P, S;
  Legendre(n_max, theta, P, S);
  Y.assign((n_max + 1) * (n_max + 1), 0.0);
  for (int n = 0; n <= n_max; n++)
  {
    for (int m = -n; m <= n; m++)
    {
      const int am = std::abs(m);
      const double T = (m == 0)  ? 1.0
                       : (m > 0) ? M_SQRT2 * std::cos(am * phi)
                                 : M_SQRT2 * std::sin(am * phi);
      Y[n * n + m + n] = P[n][am] * T;
    }
  }
}

void SphericalHarmonicSurfaceGradients(int n_max, double theta, double phi,
                                       std::vector<std::array<double, 2>> &grad)
{
  std::vector<std::vector<double>> P, S;
  Legendre(n_max, theta, P, S);
  const double x = std::cos(theta), s = std::sin(theta);
  grad.resize(n_max * (n_max + 2));
  for (int n = 1; n <= n_max; n++)
  {
    for (int m = -n; m <= n; m++)
    {
      const int am = std::abs(m);
      double dP;  // dP̄_n^|m| / dθ
      if (am == 0)
      {
        dP = -std::sqrt(n * (n + 1.0)) * s * S[n][1];
      }
      else
      {
        // dP̄_n^m / dθ = (n cos θ P̄_n^m - f_nm P̄_{n-1}^m) / sin θ.
        const double f = std::sqrt((double(n) * n - double(am) * am) * (2.0 * n + 1.0) /
                                   (2.0 * n - 1.0));
        dP = n * x * S[n][am] - f * ((n - 1 >= am) ? S[n - 1][am] : 0.0);
      }
      double T, dT;
      if (m == 0)
      {
        T = 1.0;
        dT = 0.0;
      }
      else if (m > 0)
      {
        T = M_SQRT2 * std::cos(am * phi);
        dT = -M_SQRT2 * am * std::sin(am * phi);
      }
      else
      {
        T = M_SQRT2 * std::sin(am * phi);
        dT = M_SQRT2 * am * std::cos(am * phi);
      }
      // ∇ₛY = θ̂ ∂Y/∂θ + φ̂ (1/sin θ) ∂Y/∂φ, with P̄_n^m / sin θ = S_n^m.
      grad[n * n - 1 + m + n] = {dP * T, (am == 0) ? 0.0 : S[n][am] * dT};
    }
  }
}

namespace
{

// Ratio ζₙ'(x) / ζₙ(x) of the Riccati-Hankel function ζₙ(x) = x hₙ⁽²⁾(x), from the stable
// upward recurrence for rₙ = hₙ₋₁ / hₙ: ζₙ'/ζₙ = rₙ - n / x.
std::complex<double> RiccatiHankelLogDerivative(int n, double x)
{
  using namespace std::complex_literals;
  MFEM_VERIFY(n >= 1 && x > 0.0, "Invalid arguments for Riccati-Hankel function!");
  const std::complex<double> e = std::exp(-1i * x);
  const std::complex<double> h0 = 1i * e / x;
  const std::complex<double> h1 = e * (-1.0 / x + 1i / (x * x));
  std::complex<double> r = h0 / h1;  // r₁
  for (int l = 2; l <= n; l++)
  {
    // hₗ = (2l - 1) / x hₗ₋₁ - hₗ₋₂  ⟹  1 / rₗ = (2l - 1) / x - rₗ₋₁.
    r = 1.0 / ((2.0 * l - 1.0) / x - r);
  }
  return r - double(n) / x;
}

}  // namespace

std::complex<double> EigenvalueTE(int n, double k, double R)
{
  return -k * RiccatiHankelLogDerivative(n, k * R);
}

std::complex<double> EigenvalueTM(int n, double k, double R)
{
  return k / RiccatiHankelLogDerivative(n, k * R);
}

}  // namespace dtn

SphericalDtNData::SphericalDtNData(const mfem::ParMesh &mesh,
                                   const mfem::Array<int> &attr_list,
                                   const std::vector<double> &center_cfg)
  : comm(mesh.GetComm())
{
  MFEM_VERIFY(mesh.SpaceDimension() == 3,
              "Spherical DtN boundary conditions are only available for 3D simulations!");
  const int bdr_attr_max = mesh.bdr_attributes.Size() ? mesh.bdr_attributes.Max() : 0;
  mesh::AttrToMarker(bdr_attr_max, attr_list, attr_marker, true);

  // Collect the vertices of the absorbing boundary (with repetition, which does not affect
  // the fit for a sphere).
  std::vector<std::array<double, 3>> points;
  mfem::Array<int> verts;
  for (int i = 0; i < mesh.GetNBE(); i++)
  {
    if (!attr_marker[mesh.GetBdrAttribute(i) - 1])
    {
      continue;
    }
    mesh.GetBdrElementVertices(i, verts);
    for (auto v : verts)
    {
      const double *x = mesh.GetVertex(v);
      points.push_back({x[0], x[1], x[2]});
    }
  }

  if (center_cfg.size() == 3)
  {
    center = {center_cfg[0], center_cfg[1], center_cfg[2]};
  }
  else
  {
    // Least-squares sphere fit: |x|² = 2 c·x + d, with d = R² - |c|², linear in (c, d).
    double AtA[16] = {0.0}, Atb[4] = {0.0};
    for (const auto &x : points)
    {
      const double row[4] = {2.0 * x[0], 2.0 * x[1], 2.0 * x[2], 1.0};
      const double rhs = x[0] * x[0] + x[1] * x[1] + x[2] * x[2];
      for (int i = 0; i < 4; i++)
      {
        for (int j = 0; j < 4; j++)
        {
          AtA[4 * i + j] += row[i] * row[j];
        }
        Atb[i] += row[i] * rhs;
      }
    }
    Mpi::GlobalSum(16, AtA, comm);
    Mpi::GlobalSum(4, Atb, comm);
    Eigen::Matrix4d A = Eigen::Map<Eigen::Matrix4d>(AtA);
    Eigen::Vector4d sol = A.ldlt().solve(Eigen::Map<Eigen::Vector4d>(Atb));
    center = {sol(0), sol(1), sol(2)};
  }

  // Radius and sphericity check.
  double stats[3] = {0.0, 0.0, 0.0};  // Sum of distances, count, -min distance
  double dmax = 0.0;
  stats[2] = -std::numeric_limits<double>::max();
  for (const auto &x : points)
  {
    const double d = std::sqrt((x[0] - center[0]) * (x[0] - center[0]) +
                               (x[1] - center[1]) * (x[1] - center[1]) +
                               (x[2] - center[2]) * (x[2] - center[2]));
    stats[0] += d;
    stats[1] += 1.0;
    dmax = std::max(dmax, d);
    stats[2] = std::max(stats[2], -d);
  }
  Mpi::GlobalSum(2, stats, comm);
  Mpi::GlobalMax(1, &stats[2], comm);
  Mpi::GlobalMax(1, &dmax, comm);
  MFEM_VERIFY(stats[1] > 0.0, "No boundary elements found for the spherical DtN boundary!");
  radius = stats[0] / stats[1];
  const double dmin = -stats[2];
  MFEM_VERIFY((dmax - dmin) < 1.0e-3 * radius,
              "Spherical DtN boundary condition requires a spherical absorbing boundary "
              "(vertex distances to the center range from "
                  << dmin << " to " << dmax << ")!");
}

void SphericalDtNData::SetMaxDegree(int n, FiniteElementSpace &nd_fespace)
{
  if (n <= n_max)
  {
    return;
  }
  n_max = n;
  const int n_modes = dtn::NumModes(n_max);
  auto &fespace = nd_fespace.Get();
  const auto &mesh = *fespace.GetParMesh();

  // Local (L-vector) dofs on the boundary, indexed compactly.
  std::vector<int> ldof_row(fespace.GetVSize(), -1);
  std::vector<int> bdr_ldofs;
  mfem::Array<int> vdofs;
  for (int i = 0; i < mesh.GetNBE(); i++)
  {
    if (!attr_marker[mesh.GetBdrAttribute(i) - 1])
    {
      continue;
    }
    fespace.GetBdrElementVDofs(i, vdofs);
    for (auto vdof : vdofs)
    {
      const int ldof = (vdof >= 0) ? vdof : -1 - vdof;
      if (ldof_row[ldof] < 0)
      {
        ldof_row[ldof] = static_cast<int>(bdr_ldofs.size());
        bdr_ldofs.push_back(ldof);
      }
    }
  }

  // Integrate the vector spherical harmonics against the (tangential traces of the)
  // Nédélec basis functions: (w_j)_i = ∮ Ψ_j·φ_i dS.
  Eigen::MatrixXd Wl = Eigen::MatrixXd::Zero(bdr_ldofs.size(), n_modes);
  mfem::DenseMatrix shape, elmat;
  mfem::DofTransformation doftrans;
  std::vector<std::array<double, 2>> grad;
  Eigen::MatrixXd Psi(3, n_modes);
  const double c = 1.0 / radius;
  for (int i = 0; i < mesh.GetNBE(); i++)
  {
    if (!attr_marker[mesh.GetBdrAttribute(i) - 1])
    {
      continue;
    }
    const mfem::FiniteElement &fe = *fespace.GetBE(i);
    fespace.GetBdrElementVDofs(i, vdofs, doftrans);
    auto &T = *const_cast<mfem::ParMesh &>(mesh).GetBdrElementTransformation(i);
    const int ndof = fe.GetDof();
    shape.SetSize(ndof, 3);
    elmat.SetSize(ndof, n_modes);
    elmat = 0.0;
    const int q_order = 2 * fe.GetOrder() + T.OrderW() + 4;
    const auto &ir = mfem::IntRules.Get(fe.GetGeomType(), q_order);
    for (int q = 0; q < ir.GetNPoints(); q++)
    {
      const auto &ip = ir.IntPoint(q);
      T.SetIntPoint(&ip);
      fe.CalcVShape(T, shape);
      const double w = ip.weight * T.Weight();
      double x[3];
      mfem::Vector xv(x, 3);
      T.Transform(ip, xv);
      const double rx = x[0] - center[0], ry = x[1] - center[1], rz = x[2] - center[2];
      const double r = std::sqrt(rx * rx + ry * ry + rz * rz);
      const double theta = std::acos(std::clamp(rz / r, -1.0, 1.0));
      const double phi = std::atan2(ry, rx);
      const double ct = std::cos(theta), st = std::sin(theta), cp = std::cos(phi),
                   sp = std::sin(phi);
      const double th_hat[3] = {ct * cp, ct * sp, -st}, ph_hat[3] = {-sp, cp, 0.0};
      dtn::SphericalHarmonicSurfaceGradients(n_max, theta, phi, grad);
      for (int l = 1; l <= n_max; l++)
      {
        const double scale = c / std::sqrt(l * (l + 1.0));
        for (int m = -l; m <= l; m++)
        {
          const int j = l * l - 1 + m + l;
          const double gt = scale * grad[j][0], gp = scale * grad[j][1];
          for (int d = 0; d < 3; d++)
          {
            // Ψ_TE = r̂ × ∇ₛY / (R √L) = (∂_θ Y φ̂ - (1/sin θ) ∂_φ Y θ̂) / (R √L), and
            // Ψ_TM = ∇ₛY / (R √L).
            Psi(d, 2 * j) = gt * ph_hat[d] - gp * th_hat[d];
            Psi(d, 2 * j + 1) = gt * th_hat[d] + gp * ph_hat[d];
          }
        }
      }
      Eigen::Map<Eigen::MatrixXd> shape_map(shape.Data(), ndof, 3);
      Eigen::Map<Eigen::MatrixXd> elmat_map(elmat.Data(), ndof, n_modes);
      elmat_map.noalias() += w * shape_map * Psi;
    }
    if (!doftrans.IsIdentity())
    {
      for (int j = 0; j < n_modes; j++)
      {
        doftrans.TransformDual(elmat.GetColumn(j));
      }
    }
    for (int k = 0; k < ndof; k++)
    {
      const int vdof = vdofs[k];
      const int row = ldof_row[(vdof >= 0) ? vdof : -1 - vdof];
      const double sgn = (vdof >= 0) ? 1.0 : -1.0;
      for (int j = 0; j < n_modes; j++)
      {
        Wl(row, j) += sgn * elmat(k, j);
      }
    }
  }

  // Map to true dofs (W = Pᵀ Wl) and keep only the rows of the true dofs on the boundary.
  fespace.GetEssentialTrueDofs(attr_marker, rows);
  W.setZero(rows.Size(), n_modes);
  const mfem::Operator &P = *fespace.GetProlongationMatrix();
  Vector lvec(fespace.GetVSize()), tvec(fespace.GetTrueVSize());
  lvec.UseDevice(false);
  tvec.UseDevice(false);
  for (int j = 0; j < n_modes; j++)
  {
    lvec = 0.0;
    for (std::size_t k = 0; k < bdr_ldofs.size(); k++)
    {
      lvec(bdr_ldofs[k]) = Wl(k, j);
    }
    P.MultTranspose(lvec, tvec);
    const double *t = tvec.HostRead();
    for (int k = 0; k < rows.Size(); k++)
    {
      W(k, j) = t[rows[k]];
    }
  }
}

SphericalDtNOperator::SphericalDtNOperator(const SphericalDtNData &data, int n_modes,
                                           std::vector<std::complex<double>> &&g,
                                           int height,
                                           const mfem::Array<int> *dbc_tdof_list)
  : ComplexOperator(height), data(data), g(std::move(g))
{
  MFEM_VERIFY(static_cast<int>(this->g.size()) == n_modes &&
                  n_modes <= data.GetModeVectors().cols(),
              "Invalid number of modes for SphericalDtNOperator!");
  const auto &rows = data.GetRows();
  keep.assign(rows.Size(), true);
  if (dbc_tdof_list && dbc_tdof_list->Size() > 0)
  {
    std::vector<bool> dbc(height, false);
    for (auto i : *dbc_tdof_list)
    {
      dbc[i] = true;
    }
    for (int k = 0; k < rows.Size(); k++)
    {
      keep[k] = !dbc[rows[k]];
    }
  }
  xr.resize(rows.Size());
  xi.resize(rows.Size());
  c.resize(2 * n_modes);
}

void SphericalDtNOperator::Mult(const ComplexVector &x, ComplexVector &y) const
{
  y = 0.0;
  AddMult(x, y, 1.0);
}

void SphericalDtNOperator::AddMult(const ComplexVector &x, ComplexVector &y,
                                   const std::complex<double> a) const
{
  const auto &rows = data.GetRows();
  const int n_modes = static_cast<int>(g.size());
  const auto W = data.GetModeVectors().leftCols(n_modes);
  const double *XR = x.Real().HostRead();
  const double *XI = x.Imag().HostRead();
  for (int k = 0; k < rows.Size(); k++)
  {
    xr(k) = keep[k] ? XR[rows[k]] : 0.0;
    xi(k) = keep[k] ? XI[rows[k]] : 0.0;
  }

  // Mode projections (w_jᵀ x), reduced across processes at once.
  c.head(n_modes).noalias() = W.transpose() * xr;
  c.tail(n_modes).noalias() = W.transpose() * xi;
  Mpi::GlobalSum(2 * n_modes, c.data(), data.GetComm());
  for (int j = 0; j < n_modes; j++)
  {
    const std::complex<double> s = a * g[j] * std::complex<double>(c(j), c(n_modes + j));
    c(j) = s.real();
    c(n_modes + j) = s.imag();
  }
  xr.noalias() = W * c.head(n_modes);
  xi.noalias() = W * c.tail(n_modes);
  double *YR = y.Real().HostReadWrite();
  double *YI = y.Imag().HostReadWrite();
  for (int k = 0; k < rows.Size(); k++)
  {
    if (keep[k])
    {
      YR[rows[k]] += xr(k);
      YI[rows[k]] += xi(k);
    }
  }
}

}  // namespace palace
