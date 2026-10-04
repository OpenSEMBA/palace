// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

// Tests for incident plane wave excitations and the exact spherical Dirichlet-to-Neumann
// (DtN) absorbing boundary condition used for scattering (radar cross section) problems.

#include <array>
#include <cmath>
#include <complex>
#include <fstream>
#include <memory>
#include <vector>
#include <mfem.hpp>
#include <catch2/catch_test_macros.hpp>
#include <nlohmann/json.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "fem/fespace.hpp"
#include "fem/gridfunction.hpp"
#include "fem/integrator.hpp"
#include "fem/mesh.hpp"
#include "linalg/vector.hpp"
#include "models/planewaveoperator.hpp"
#include "models/spaceoperator.hpp"
#include "models/sphericaldtn.hpp"
#include "utils/communication.hpp"
#include "utils/configfile.hpp"
#include "utils/jsonschema.hpp"
#include "utils/labels.hpp"
#include "utils/units.hpp"

namespace palace
{
using json = nlohmann::json;
using namespace Catch::Matchers;
using namespace std::complex_literals;

namespace
{

// Spherical shell between radii 0.5 (boundary attribute 2) and 1.0 (boundary attribute 1),
// with second-order curved elements.
std::unique_ptr<mfem::Mesh> LoadSphereShellMesh()
{
  std::ifstream fi(std::string(PALACE_TEST_DATA_DIR) + "/mesh/sphere-shell-p2.msh");
  REQUIRE(fi.good());
  return std::make_unique<mfem::Mesh>(fi, 1, 1, true);
}

// Unit cube [-0.5, 0.5]³ with one boundary attribute per face (1 to 6), refined so that
// every process owns elements. Elements with centroid in [-0.25, 0.25]³ are given attribute
// 2 if requested.
std::unique_ptr<mfem::Mesh> MakeCubeMesh(MPI_Comm comm, bool inclusion)
{
  auto mesh = std::make_unique<mfem::Mesh>(
      mfem::Mesh::MakeCartesian3D(4, 4, 4, mfem::Element::TETRAHEDRON));
  mesh->Transform(
      [](const mfem::Vector &x, mfem::Vector &p)
      {
        p = x;
        p -= 0.5;
      });
  while (mesh->GetNE() < 8 * Mpi::Size(comm))
  {
    mesh->UniformRefinement();
  }
  if (inclusion)
  {
    mfem::Vector c(3);
    for (int i = 0; i < mesh->GetNE(); i++)
    {
      mesh->GetElementCenter(i, c);
      if (std::abs(c(0)) < 0.25 && std::abs(c(1)) < 0.25 && std::abs(c(2)) < 0.25)
      {
        mesh->SetAttribute(i, 2);
      }
    }
    mesh->SetAttributes();
  }
  return mesh;
}

config::SolverData MakeSolverData(int order)
{
  config::SolverData solver;
  solver.order = order;
  solver.linear.mg_max_levels = 1;
  return solver;
}

// Initialize the global integration order settings for the given solver configuration (as
// done by IoData), restoring the previous settings on destruction.
struct IntegrationOrderScope
{
  int p_trial = fem::DefaultIntegrationOrder::p_trial;
  bool q_order_jac = fem::DefaultIntegrationOrder::q_order_jac;
  int q_order_extra_pk = fem::DefaultIntegrationOrder::q_order_extra_pk;
  int q_order_extra_qk = fem::DefaultIntegrationOrder::q_order_extra_qk;

  IntegrationOrderScope(const config::SolverData &solver)
  {
    fem::DefaultIntegrationOrder::p_trial = solver.order;
    fem::DefaultIntegrationOrder::q_order_jac = solver.q_order_jac;
    fem::DefaultIntegrationOrder::q_order_extra_pk = solver.q_order_extra;
    fem::DefaultIntegrationOrder::q_order_extra_qk = solver.q_order_extra;
  }
  ~IntegrationOrderScope()
  {
    fem::DefaultIntegrationOrder::p_trial = p_trial;
    fem::DefaultIntegrationOrder::q_order_jac = q_order_jac;
    fem::DefaultIntegrationOrder::q_order_extra_pk = q_order_extra_pk;
    fem::DefaultIntegrationOrder::q_order_extra_qk = q_order_extra_qk;
  }
};

config::PlaneWaveData MakePlaneWave()
{
  // Oblique incidence with polarization orthogonal to the propagation direction.
  config::PlaneWaveData pw;
  pw.direction = {1.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0};
  pw.polarization = {2.0 / 3.0, 1.0 / 3.0, -2.0 / 3.0};
  pw.amplitude = 1.0;
  return pw;
}

// Assemble the frequency domain system matrix A(ω) as in the driven solver, and compute
// r = A Π(E_inc) - b for the plane wave excitation with source b.
struct PlaneWaveResidual
{
  ComplexVector r, b, E_inc;
};

PlaneWaveResidual ComputePlaneWaveResidual(SpaceOperator &space_op, double omega)
{
  auto K = space_op.GetStiffnessMatrix<ComplexOperator>(Operator::DIAG_ONE);
  auto C = space_op.GetDampingMatrix<ComplexOperator>(Operator::DIAG_ZERO);
  auto M = space_op.GetMassMatrix<ComplexOperator>(Operator::DIAG_ZERO);
  auto A2 = space_op.GetExtraSystemOperator(omega, Operator::DIAG_ZERO);
  auto A = space_op.GetSystemMatrix(1.0 + 0.0i, 1i * omega, -omega * omega + 0.0i, K.get(),
                                    C.get(), M.get(), A2.get());
  PlaneWaveResidual res;
  REQUIRE(space_op.GetExcitationVector(1, omega, res.b));
  REQUIRE(space_op.GetPlaneWaveIncidentField(1, omega, res.E_inc));
  res.r.SetSize(res.b.Size());
  res.r.UseDevice(true);
  A->Mult(res.E_inc, res.r);
  res.r -= res.b;
  return res;
}

// Marker of the true dofs whose basis functions have support on elements with the given
// attribute (on any process).
std::vector<bool> TrueDofsOnAttribute(FiniteElementSpace &fespace, int attr)
{
  // Mark the local dofs, combine the markers of shared dofs across processes (bitwise OR,
  // independent of the dof orientation signs), and restrict to the true dofs.
  auto &fes = fespace.Get();
  mfem::Array<int> ldof_marker(fes.GetVSize()), tdof_marker(fes.GetTrueVSize());
  ldof_marker = 0;
  mfem::Array<int> vdofs;
  for (int e = 0; e < fes.GetNE(); e++)
  {
    if (fes.GetParMesh()->GetAttribute(e) != attr)
    {
      continue;
    }
    fes.GetElementVDofs(e, vdofs);
    for (auto vdof : vdofs)
    {
      ldof_marker[(vdof >= 0) ? vdof : -1 - vdof] = 1;
    }
  }
  fes.Synchronize(ldof_marker);
  fes.GetRestrictionMatrix()->BooleanMult(ldof_marker, tdof_marker);
  std::vector<bool> marker(tdof_marker.Size());
  for (int i = 0; i < tdof_marker.Size(); i++)
  {
    marker[i] = (tdof_marker[i] != 0);
  }
  return marker;
}

// Riccati-Hankel function ζₙ(x) = x hₙ⁽²⁾(x) and its derivative, from the explicit finite
// series hₙ⁽²⁾(x) = iⁿ⁺¹ e^{-ix} / x Σₘ (n+m)! / (m! (n-m)!) (-i / (2x))ᵐ.
std::pair<std::complex<double>, std::complex<double>> RiccatiHankelSeries(int n, double x)
{
  std::complex<double> z = 0.0, dz = 0.0;
  for (int m = 0; m <= n; m++)
  {
    const double c =
        std::tgamma(n + m + 1.0) / (std::tgamma(m + 1.0) * std::tgamma(n - m + 1.0));
    const std::complex<double> t = c * std::pow(-0.5i, m);
    z += t * std::pow(x, -m);
    dz += t * (-1i * std::pow(x, -m) - double(m) * std::pow(x, -m - 1));
  }
  const std::complex<double> f = std::pow(1i, n + 1) * std::exp(-1i * x);
  return {f * z, f * dz};
}

}  // namespace

TEST_CASE("PlaneWave configuration", "[planewave][config][Serial]")
{
  SECTION("Keyword directions")
  {
    config::PlaneWaveData pw(
        json{{"Index", 1}, {"Direction", "-Z"}, {"Polarization", "+X"}});
    CHECK(pw.direction == std::array<double, 3>{0.0, 0.0, -1.0});
    CHECK(pw.polarization == std::array<double, 3>{1.0, 0.0, 0.0});
    CHECK(pw.amplitude == 1.0);
  }

  SECTION("Array directions are normalized")
  {
    config::PlaneWaveData pw(json{{"Index", 1},
                                  {"Direction", {0.0, 3.0, 4.0}},
                                  {"Polarization", {2.0, 0.0, 0.0}},
                                  {"Amplitude", 2.5}});
    CHECK_THAT(pw.direction[1], WithinRel(0.6));
    CHECK_THAT(pw.direction[2], WithinRel(0.8));
    CHECK_THAT(pw.polarization[0], WithinRel(1.0));
    CHECK(pw.amplitude == 2.5);
  }

  SECTION("Invalid plane waves")
  {
    // Polarization not orthogonal to the propagation direction.
    CHECK_THROWS(config::PlaneWaveData(
        json{{"Index", 1}, {"Direction", "+Z"}, {"Polarization", {1.0, 0.0, 1.0}}}));
    // Zero polarization.
    CHECK_THROWS(config::PlaneWaveData(
        json{{"Index", 1}, {"Direction", "+Z"}, {"Polarization", {0.0, 0.0, 0.0}}}));
    // Nonpositive amplitude.
    CHECK_THROWS(config::PlaneWaveData(json{
        {"Index", 1}, {"Direction", "+Z"}, {"Polarization", "+Y"}, {"Amplitude", 0.0}}));
  }

  SECTION("Domains")
  {
    json domains = {
        {"Materials", json::array({{{"Attributes", {1}}}})},
        {"PlaneWave",
         json::array({{{"Index", 1}, {"Direction", "-Z"}, {"Polarization", "+X"}},
                      {{"Index", 3}, {"Direction", "+X"}, {"Polarization", "+Z"}}})}};
    config::DomainData data(domains);
    REQUIRE(data.plane_wave.size() == 2);
    CHECK(data.plane_wave.count(1) == 1);
    CHECK(data.plane_wave.at(3).direction == std::array<double, 3>{1.0, 0.0, 0.0});
  }

  SECTION("Schema")
  {
    CHECK(ValidateConfig(json{{"Index", 1}, {"Direction", "-Z"}, {"Polarization", "+X"}},
                         "PlaneWave")
              .empty());
    CHECK(ValidateConfig(json{{"Index", 1},
                              {"Direction", {0.0, 0.0, -1.0}},
                              {"Polarization", {0.0, 1.0, 0.0}},
                              {"Amplitude", 2.0}},
                         "PlaneWave")
              .empty());
    CHECK(!ValidateConfig(json{{"Index", 1}, {"Direction", "-Z"}}, "PlaneWave").empty());
    CHECK(!ValidateConfig(json{{"Index", 1},
                               {"Direction", "-Z"},
                               {"Polarization", "+X"},
                               {"Amplitude", -1.0}},
                          "PlaneWave")
               .empty());
  }
}

TEST_CASE("Spherical DtN configuration", "[sphericaldtn][config][Serial]")
{
  config::FarfieldBoundaryData data(json{{"Attributes", {1}},
                                         {"DtN", true},
                                         {"DtNMaxDegree", 8},
                                         {"DtNCenter", {0.0, 0.0, 1.0}}});
  CHECK(data.dtn);
  CHECK(data.dtn_max_degree == 8);
  CHECK(data.dtn_center == std::vector<double>{0.0, 0.0, 1.0});

  config::FarfieldBoundaryData data_default(json{{"Attributes", {1}}});
  CHECK(!data_default.dtn);
  CHECK(data_default.dtn_center.empty());

  // The DtN correction is defined relative to the first-order condition.
  CHECK_THROWS(
      config::FarfieldBoundaryData(json{{"Attributes", {1}}, {"Order", 2}, {"DtN", true}}));

  CHECK(ValidateConfig(json{{"Attributes", {1}}, {"DtN", true}, {"DtNMaxDegree", 10}},
                       "Absorbing")
            .empty());
  CHECK(!ValidateConfig(json{{"Attributes", {1}}, {"DtN", 1}}, "Absorbing").empty());
}

TEST_CASE("PlaneWave incident field", "[planewave][Serial]")
{
  Units units(0.5, 2.0);
  const auto cfg = MakePlaneWave();
  PlaneWaveData pw(cfg, units);
  const double E0 = units.Nondimensionalize<Units::ValueType::FIELD_E>(cfg.amplitude);
  CHECK_THAT(pw.amplitude, WithinRel(E0));

  const double k = 3.7, omega = 1.9;
  const std::vector<std::array<double, 3>> points = {
      {0.0, 0.0, 0.0}, {0.3, -0.2, 0.7}, {-1.1, 0.4, 0.25}, {2.0, 1.5, -0.8}};
  for (const auto &x : points)
  {
    const auto E = pw.E(x.data(), k);
    const double kx =
        k * (cfg.direction[0] * x[0] + cfg.direction[1] * x[1] + cfg.direction[2] * x[2]);
    std::complex<double> k_dot_E = 0.0, p_dot_E = 0.0;
    for (int d = 0; d < 3; d++)
    {
      k_dot_E += cfg.direction[d] * E[d];
      p_dot_E += cfg.polarization[d] * E[d];
    }
    // Transverse, with amplitude E₀ and phase -k k̂·x (exp(+iωt) convention).
    CHECK(std::abs(k_dot_E) < 1.0e-12 * E0);
    CHECK_THAT(std::abs(p_dot_E - E0 * std::exp(-1i * kx)), WithinAbs(0.0, 1.0e-12 * E0));

    // Faraday's law: ∇ × E = -iω B, with the curl approximated by central differences.
    const auto B = pw.B(x.data(), k, omega);
    const double h = 1.0e-5;
    std::array<std::array<std::complex<double>, 3>, 3> dE;  // dE[j][i] = ∂E_i / ∂x_j
    for (int j = 0; j < 3; j++)
    {
      auto xp = x, xm = x;
      xp[j] += h;
      xm[j] -= h;
      const auto Ep = pw.E(xp.data(), k), Em = pw.E(xm.data(), k);
      for (int i = 0; i < 3; i++)
      {
        dE[j][i] = (Ep[i] - Em[i]) / (2.0 * h);
      }
    }
    const std::array<std::complex<double>, 3> curlE = {
        dE[1][2] - dE[2][1], dE[2][0] - dE[0][2], dE[0][1] - dE[1][0]};
    for (int d = 0; d < 3; d++)
    {
      CHECK_THAT(std::abs(curlE[d] + 1i * omega * B[d]), WithinAbs(0.0, 1.0e-7 * k * E0));
    }
  }
}

TEST_CASE("Real spherical harmonics", "[sphericaldtn][Serial]")
{
  // Gauss-Legendre quadrature in cos θ and trapezoidal rule in φ, exact for the products
  // of harmonics of degree up to n_max.
  constexpr int n_max = 6, n_theta = 16, n_phi = 32;
  const auto &ir = mfem::IntRules.Get(mfem::Geometry::SEGMENT, 2 * n_theta - 1);
  const int nY = (n_max + 1) * (n_max + 1), nG = n_max * (n_max + 2);
  std::vector<double> gram_Y(nY * nY, 0.0), gram_G(nG * nG, 0.0);
  std::vector<double> Y;
  std::vector<std::array<double, 2>> G;
  for (int i = 0; i < ir.GetNPoints(); i++)
  {
    const double theta = std::acos(2.0 * ir.IntPoint(i).x - 1.0);
    for (int j = 0; j < n_phi; j++)
    {
      const double phi = 2.0 * M_PI * j / n_phi;
      const double w = 2.0 * ir.IntPoint(i).weight * 2.0 * M_PI / n_phi;
      dtn::SphericalHarmonics(n_max, theta, phi, Y);
      dtn::SphericalHarmonicSurfaceGradients(n_max, theta, phi, G);
      for (int a = 0; a < nY; a++)
      {
        for (int b = 0; b < nY; b++)
        {
          gram_Y[a * nY + b] += w * Y[a] * Y[b];
        }
      }
      for (int a = 0; a < nG; a++)
      {
        for (int b = 0; b < nG; b++)
        {
          gram_G[a * nG + b] += w * (G[a][0] * G[b][0] + G[a][1] * G[b][1]);
        }
      }
    }
  }

  // Orthonormality: ∮ Y_i Y_j dΩ = δ_ij and ∮ ∇ₛY_i·∇ₛY_j dΩ = n(n+1) δ_ij.
  for (int a = 0; a < nY; a++)
  {
    for (int b = 0; b < nY; b++)
    {
      CHECK_THAT(gram_Y[a * nY + b], WithinAbs((a == b) ? 1.0 : 0.0, 1.0e-12));
    }
  }
  for (int n = 1; n <= n_max; n++)
  {
    for (int m = -n; m <= n; m++)
    {
      const int a = n * n - 1 + m + n;
      for (int b = 0; b < nG; b++)
      {
        CHECK_THAT(gram_G[a * nG + b], WithinAbs((a == b) ? n * (n + 1.0) : 0.0, 1.0e-11));
      }
    }
  }

  // Surface gradients against finite differences of the harmonics.
  const double h = 1.0e-6;
  for (const auto &[theta, phi] : std::vector<std::pair<double, double>>{
           {0.3, 1.2}, {1.4, -2.5}, {2.9, 0.4}, {1.0e-3, 0.7}})
  {
    std::vector<double> Ytp, Ytm, Ypp, Ypm;
    dtn::SphericalHarmonics(n_max, theta + h, phi, Ytp);
    dtn::SphericalHarmonics(n_max, theta - h, phi, Ytm);
    dtn::SphericalHarmonics(n_max, theta, phi + h, Ypp);
    dtn::SphericalHarmonics(n_max, theta, phi - h, Ypm);
    dtn::SphericalHarmonicSurfaceGradients(n_max, theta, phi, G);
    for (int n = 1; n <= n_max; n++)
    {
      for (int m = -n; m <= n; m++)
      {
        const int iY = n * n + m + n, iG = n * n - 1 + m + n;
        const double dth = (Ytp[iY] - Ytm[iY]) / (2.0 * h);
        const double dph = (Ypp[iY] - Ypm[iY]) / (2.0 * h) / std::sin(theta);
        CHECK_THAT(G[iG][0], WithinAbs(dth, 1.0e-6 * n * n));
        CHECK_THAT(G[iG][1], WithinAbs(dph, 1.0e-6 * n * n));
      }
    }
  }
}

TEST_CASE("Spherical DtN eigenvalues", "[sphericaldtn][Serial]")
{
  const double R = 1.3;
  for (double x : {0.5, 2.0, 7.3, 25.0})
  {
    const double k = x / R;

    // Degree 1 against the wave impedances of the electric (TM) and magnetic (TE) dipole
    // fields, exp(+iωt): r̂ × ∇ × E = iωμ H_φ θ̂ = λ_TM E_θ θ̂ for the electric dipole, and
    // r̂ × ∇ × E = -iωμ H_θ φ̂ = λ_TE E_φ φ̂ for the magnetic dipole.
    const std::complex<double> u = 1.0 - 1i / x, v = 1.0 - 1i / x - 1.0 / (x * x);
    CHECK_THAT(std::abs(dtn::EigenvalueTM(1, k, R) - 1i * k * u / v),
               WithinAbs(0.0, 1.0e-12 * k));
    CHECK_THAT(std::abs(dtn::EigenvalueTE(1, k, R) - 1i * k * v / u),
               WithinAbs(0.0, 1.0e-12 * k));

    // Higher degrees against the explicit series for the Riccati-Hankel function.
    for (int n = 1; n <= 8; n++)
    {
      const auto [z, dz] = RiccatiHankelSeries(n, x);
      CHECK_THAT(std::abs(dtn::EigenvalueTE(n, k, R) + k * dz / z),
                 WithinAbs(0.0, 1.0e-10 * k * std::abs(dz / z)));
      CHECK_THAT(std::abs(dtn::EigenvalueTM(n, k, R) - k * z / dz),
                 WithinAbs(0.0, 1.0e-10 * k * std::abs(z / dz)));
    }
  }

  // Far from the source, the DtN map reduces to the first-order absorbing condition ik.
  const double k = 2.0, R_far = 1.0e5;
  for (int n = 1; n <= 4; n++)
  {
    CHECK_THAT(std::abs(dtn::EigenvalueTE(n, k, R_far) - 1i * k),
               WithinAbs(0.0, 1.0e-4 * k));
    CHECK_THAT(std::abs(dtn::EigenvalueTM(n, k, R_far) - 1i * k),
               WithinAbs(0.0, 1.0e-4 * k));
  }

  // Outgoing waves carry power away: Re{λ} > 0 for TM modes (capacitive near field) and
  // Im{λ} > 0 for both.
  for (int n = 1; n <= 10; n++)
  {
    CHECK(dtn::EigenvalueTE(n, 1.0, 2.0).imag() > 0.0);
    CHECK(dtn::EigenvalueTM(n, 1.0, 2.0).imag() > 0.0);
  }
}

TEST_CASE("Spherical DtN mode vectors", "[sphericaldtn][Serial][Parallel]")
{
  MPI_Comm comm = Mpi::World();
  auto smesh = LoadSphereShellMesh();
  Mesh mesh(comm, *smesh);
  mfem::ND_FECollection nd_fec(3, 3);
  FiniteElementSpace nd_fespace(mesh, &nd_fec);

  mfem::Array<int> attr_list({1});
  SphericalDtNData data(mesh.Get(), attr_list, {});
  CHECK_THAT(data.GetCenter()[0], WithinAbs(0.0, 1.0e-10));
  CHECK_THAT(data.GetCenter()[1], WithinAbs(0.0, 1.0e-10));
  CHECK_THAT(data.GetCenter()[2], WithinAbs(0.0, 1.0e-10));
  CHECK_THAT(data.GetRadius(), WithinRel(1.0, 1.0e-10));

  // Interpolate each vector spherical harmonic Ψ_j (extended radially) into the Nédélec
  // space: its projections onto the mode vectors are w_kᵀ Π(Ψ_j) = ∮ Ψ_k·Ψ_j dS ≈ δ_jk.
  constexpr int n_max = 3;
  data.SetMaxDegree(n_max, nd_fespace);
  const int n_modes = dtn::NumModes(n_max);
  REQUIRE(data.GetModeVectors().cols() == n_modes);
  const auto &rows = data.GetRows();
  const auto &W = data.GetModeVectors();
  for (int j = 0; j < n_modes; j++)
  {
    mfem::VectorFunctionCoefficient Psi(
        3,
        [j](const mfem::Vector &x, mfem::Vector &f)
        {
          const double r = x.Norml2();
          const double theta = std::acos(x(2) / r), phi = std::atan2(x(1), x(0));
          std::vector<std::array<double, 2>> G;
          dtn::SphericalHarmonicSurfaceGradients(n_max, theta, phi, G);
          const int l = static_cast<int>(std::sqrt(j / 2 + 1.0));
          const double s = 1.0 / std::sqrt(l * (l + 1.0));  // R = 1
          const double gt = s * G[j / 2][0], gp = s * G[j / 2][1];
          const double th[3] = {std::cos(theta) * std::cos(phi),
                                std::cos(theta) * std::sin(phi), -std::sin(theta)};
          const double ph[3] = {-std::sin(phi), std::cos(phi), 0.0};
          for (int d = 0; d < 3; d++)
          {
            f(d) = (j % 2 == 0) ? gt * ph[d] - gp * th[d] : gt * th[d] + gp * ph[d];
          }
        });
    GridFunction E(nd_fespace);
    E.Real().ProjectCoefficient(Psi);
    Vector t(nd_fespace.GetTrueVSize());
    E.Real().ParallelProject(t);
    std::vector<double> c(n_modes, 0.0);
    for (int k = 0; k < n_modes; k++)
    {
      for (int i = 0; i < rows.Size(); i++)
      {
        c[k] += W(i, k) * t(rows[i]);
      }
    }
    Mpi::GlobalSum(n_modes, c.data(), comm);
    for (int k = 0; k < n_modes; k++)
    {
      INFO("Mode " << j << ", projection on mode " << k);
      CHECK_THAT(c[k], WithinAbs((j == k) ? 1.0 : 0.0, 1.0e-2));
    }
  }

  // The operator applies Σ_j g_j w_j (w_jᵀ x), with eliminated rows and columns.
  std::vector<std::complex<double>> g(n_modes);
  for (int j = 0; j < n_modes; j++)
  {
    g[j] = std::complex<double>(1.0 + j, -0.5 * j);
  }
  const int n = nd_fespace.GetTrueVSize();
  ComplexVector x(n), y(n), y_masked(n);
  linalg::SetRandom(comm, x);
  SphericalDtNOperator op(data, n_modes, std::vector<std::complex<double>>(g), n, nullptr);
  op.Mult(x, y);
  mfem::Array<int> dbc;
  for (int i = 0; i < rows.Size(); i += 2)
  {
    dbc.Append(rows[i]);
  }
  SphericalDtNOperator op_masked(data, n_modes, std::vector<std::complex<double>>(g), n,
                                 &dbc);
  op_masked.Mult(x, y_masked);

  // Reference: explicit evaluation of the low-rank sum, with and without masking.
  std::vector<std::complex<double>> c(n_modes, 0.0), c_masked(n_modes, 0.0);
  for (int k = 0; k < n_modes; k++)
  {
    for (int i = 0; i < rows.Size(); i++)
    {
      const std::complex<double> xi(x.Real()(rows[i]), x.Imag()(rows[i]));
      c[k] += W(i, k) * xi;
      c_masked[k] += (i % 2 == 0) ? 0.0 : W(i, k) * xi;
    }
  }
  Mpi::GlobalSum(n_modes, c.data(), comm);
  Mpi::GlobalSum(n_modes, c_masked.data(), comm);
  double err = 0.0, err_masked = 0.0, nrm = 0.0;
  for (int i = 0; i < rows.Size(); i++)
  {
    std::complex<double> yi = 0.0, yi_masked = 0.0;
    for (int k = 0; k < n_modes; k++)
    {
      yi += g[k] * c[k] * W(i, k);
      yi_masked += (i % 2 == 0) ? 0.0 : g[k] * c_masked[k] * W(i, k);
    }
    nrm = std::max(nrm, std::abs(yi));
    err = std::max(
        err, std::abs(yi - std::complex<double>(y.Real()(rows[i]), y.Imag()(rows[i]))));
    err_masked = std::max(
        err_masked, std::abs(yi_masked - std::complex<double>(y_masked.Real()(rows[i]),
                                                              y_masked.Imag()(rows[i]))));
  }
  Mpi::GlobalMax(1, &nrm, comm);
  Mpi::GlobalMax(1, &err, comm);
  Mpi::GlobalMax(1, &err_masked, comm);
  CHECK(nrm > 0.0);
  CHECK(err <= 1.0e-12 * nrm);
  CHECK(err_masked <= 1.0e-12 * nrm);
}

TEST_CASE("Spherical DtN rejects nonspherical boundaries", "[sphericaldtn][Serial]")
{
  // The absorbing boundary consisting of both spheres is not a sphere.
  auto smesh = LoadSphereShellMesh();
  Mesh mesh(Mpi::World(), *smesh);
  mfem::Array<int> attr_both({1, 2});
  CHECK_THROWS(SphericalDtNData(mesh.Get(), attr_both, {}));
}

TEST_CASE("PlaneWave source reproduces the incident field in free space",
          "[planewave][Serial][Parallel]")
{
  // Without scatterers, the system matrix equals the background matrix used to build the
  // plane wave source, so the interpolant of the incident field solves the discrete
  // problem exactly: the scattered field vanishes.
  MPI_Comm comm = Mpi::World();
  const int abc_order = GENERATE(1, 2);
  std::vector<std::unique_ptr<Mesh>> mesh;
  mesh.push_back(std::make_unique<Mesh>(comm, *MakeCubeMesh(comm, false)));

  config::MaterialData material;
  material.attributes = {1};
  config::DomainData domains;
  domains.attributes = {1};
  domains.materials = {material};
  domains.plane_wave[1] = MakePlaneWave();
  config::BoundaryData boundaries;
  boundaries.farfield.attributes = {1, 2, 3, 4, 5, 6};
  boundaries.farfield.order = abc_order;
  Units units(1.0, 1.0);
  const auto solver = MakeSolverData(2);
  IntegrationOrderScope integration_order(solver);
  SpaceOperator space_op(solver, domains, boundaries, ProblemType::DRIVEN, units, mesh);
  REQUIRE(space_op.GetPortExcitations().Size() == 1);

  const double omega = 4.0;
  auto res = ComputePlaneWaveResidual(space_op, omega);
  const double b_norm = linalg::Norml2(comm, res.b);
  CHECK(b_norm > 0.0);
  CHECK(linalg::Norml2(comm, res.r) <= 1.0e-12 * b_norm);

  // The background wavenumber is that of free space.
  CHECK_THAT(space_op.GetPlaneWaveWavenumber(omega), WithinRel(omega, 1.0e-14));
}

TEST_CASE("PlaneWave scattered field source is local to the scatterer",
          "[planewave][Serial][Parallel]")
{
  // With a dielectric inclusion, the scattered field E_s = E - Π(E_inc) is driven by
  // A E_s = -(A - A_bg) Π(E_inc), which is supported only on the dofs of the inclusion.
  MPI_Comm comm = Mpi::World();
  std::vector<std::unique_ptr<Mesh>> mesh;
  mesh.push_back(std::make_unique<Mesh>(comm, *MakeCubeMesh(comm, true)));

  config::MaterialData air, dielectric;
  air.attributes = {1};
  dielectric.attributes = {2};
  dielectric.epsilon_r = 4.0;
  dielectric.mu_r = 2.0;
  config::DomainData domains;
  domains.attributes = {1, 2};
  domains.materials = {air, dielectric};
  domains.plane_wave[1] = MakePlaneWave();
  config::BoundaryData boundaries;
  boundaries.farfield.attributes = {1, 2, 3, 4, 5, 6};
  boundaries.farfield.order = 2;
  Units units(1.0, 1.0);
  const auto solver = MakeSolverData(2);
  IntegrationOrderScope integration_order(solver);
  SpaceOperator space_op(solver, domains, boundaries, ProblemType::DRIVEN, units, mesh);

  const double omega = 4.0;
  auto res = ComputePlaneWaveResidual(space_op, omega);
  const auto on_inclusion = TrueDofsOnAttribute(space_op.GetNDSpace(), 2);
  double r_in = 0.0, r_out = 0.0, b_max = 0.0;
  for (int i = 0; i < res.r.Size(); i++)
  {
    const double ri = std::abs(std::complex<double>(res.r.Real()(i), res.r.Imag()(i)));
    (on_inclusion[i] ? r_in : r_out) = std::max(on_inclusion[i] ? r_in : r_out, ri);
    b_max =
        std::max(b_max, std::abs(std::complex<double>(res.b.Real()(i), res.b.Imag()(i))));
  }
  Mpi::GlobalMax(1, &b_max, comm);
  Mpi::GlobalMax(1, &r_in, comm);
  Mpi::GlobalMax(1, &r_out, comm);
  CHECK(r_in > 1.0e-3 * b_max);
  CHECK(r_out <= 1.0e-12 * b_max);
}

TEST_CASE("PlaneWave with spherical DtN boundary",
          "[planewave][sphericaldtn][Serial][Parallel]")
{
  MPI_Comm comm = Mpi::World();
  const bool pec = GENERATE(false, true);
  std::vector<std::unique_ptr<Mesh>> mesh;
  mesh.push_back(std::make_unique<Mesh>(comm, *LoadSphereShellMesh()));

  config::MaterialData material;
  material.attributes = {1};
  config::DomainData domains;
  domains.attributes = {1};
  domains.materials = {material};
  domains.plane_wave[1] = MakePlaneWave();
  config::BoundaryData boundaries;
  boundaries.farfield.attributes = {1};
  boundaries.farfield.dtn = true;
  if (pec)
  {
    boundaries.pec.attributes = {2};
  }
  Units units(1.0, 1.0);
  const auto solver = MakeSolverData(2);
  IntegrationOrderScope integration_order(solver);
  SpaceOperator space_op(solver, domains, boundaries, ProblemType::DRIVEN, units, mesh);
  REQUIRE(space_op.GetFarfieldOp().HasDtN());

  const double omega = 3.0;
  auto res = ComputePlaneWaveResidual(space_op, omega);
  const double b_norm = linalg::Norml2(comm, res.b);
  CHECK(b_norm > 0.0);
  CHECK(space_op.GetFarfieldOp().GetDtNData()->GetMaxDegree() ==
        space_op.GetFarfieldOp().GetDtNMaxDegree(omega));
  if (!pec)
  {
    // The DtN correction is included consistently in the system and background matrices.
    CHECK(linalg::Norml2(comm, res.r) <= 1.0e-12 * b_norm);
  }
  else
  {
    // The source vanishes on the PEC boundary, where the total field is zero.
    const auto &dbc = space_op.GetNDDbcTDofLists().back();
    double b_pec = 0.0;
    for (auto i : dbc)
    {
      b_pec =
          std::max(b_pec, std::abs(std::complex<double>(res.b.Real()(i), res.b.Imag()(i))));
    }
    Mpi::GlobalMax(1, &b_pec, comm);
    CHECK(b_pec == 0.0);
  }
}

}  // namespace palace
