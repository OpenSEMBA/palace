// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include "strattonchu.hpp"

#include <numbers>

#include "fem/coefficient.hpp"
#include "utils/omp.hpp"

namespace palace
{

// Computes contribution to Stratton-Chu far-field integrands for multiple observation
// directions for the given element T.
//
// The Stratton-Chu transformations to compute far-field electric field rE_∞(r̂)
// at multiple observation directions r₀ (specified by a vector of (θ, ϕ)s) are:
//
// r E_∞(r₀) = (ik/4π) r₀ × ∫_S [n̂ × E - Z r₀ × (n̂ × H)] e^{ikr₀·r'} dS  ≈ r₀ × Σ (Iᵣ +
// iIᵢ).
//
// where:
//   - S is the boundary surface with outward normal n̂.
//   - E, H are the electric and magnetic fields on the surface.
//   - k = ω/c is the wavenumber.
//   - Z is the impedance.
//   - r₀ = (sin θ cos φ, sin θ sin φ, cos θ) are observation directions.
//   - r' are source points on the surface.
//
// This function computes Iᵣ and Iᵢ for all the input observation points.
//
// This equation was obtained starting from Stratton-Chu's transformations, assuming an
// analytic expression for Green's function (G(r, r0) = exp(-ik|r - r₀|) / (4π|r - r₀|)),
// and expanding for r that goes to infinity.
//
// Note:
// - This equation is only valid in three dimensional space.
// - This equation is only valid when all the materials have scalar μ and ε.
// - The implementation assumes S is an external boundary.
//
// Note also:
//
// This function uses std::vector<std::array<double, 3>> instead of Vectors. The
// reason for this is so that we can ensure that memory layout is simple (and
// contiguous), ensuring that we can perform one single MPI reduction for all
// the points in integrand_*.
void AddStrattonChuIntegrandAtElement(const GridFunction &E, const GridFunction &B,
                                      const MaterialOperator &mat_op, double omega_re,
                                      double omega_im,
                                      std::vector<std::array<double, 3>> &r_naughts,
                                      mfem::ElementTransformation &T,
                                      const mfem::IntegrationRule &ir,
                                      std::vector<std::array<double, 3>> &integrand_r,
                                      std::vector<std::array<double, 3>> &integrand_i)
{
  MFEM_VERIFY(E.VectorDim() == 3,
              "Stratton-Chu far-field computation is only available for 3D simulations!");
  MFEM_VERIFY(B.VectorDim() == 3,
              "Stratton-Chu far-field computation is only available for 3D simulations!");

  MFEM_VERIFY(integrand_r.size() == r_naughts.size() &&
                  integrand_i.size() == r_naughts.size(),
              "Mismatch between input points and result vector.")

  MFEM_ASSERT(T.ElementType == mfem::ElementTransformation::BDR_ELEMENT,
              "Unexpected element type in BdrSurfaceFluxCoefficient!");

  StaticVector<3> r_phys;
  StaticVector<3> E_real, E_imag, B_real, B_imag;
  StaticVector<3> ZH_real, ZH_imag;
  StaticVector<3> normal;
  StaticVector<3> n_cross_Er, n_cross_ZHr;
  StaticVector<3> n_cross_Ei, n_cross_ZHi;

  const mfem::ParMesh *mesh = E.Real().ParFESpace()->GetParMesh();
  mfem::FaceElementTransformations FET;
  mfem::IsoparametricTransformation T1, T2;

  for (int j = 0; j < ir.GetNPoints(); j++)
  {
    const mfem::IntegrationPoint &ip = ir.IntPoint(j);
    T.SetIntPoint(&ip);

    MFEM_ASSERT(T.ElementType == mfem::ElementTransformation::BDR_ELEMENT,
                "Unexpected element type in BdrSurfaceFluxCoefficient!");

    bool invert = BdrGridFunctionCoefficient::GetBdrElementNeighborTransformations(
        T.ElementNo, *mesh, FET, T1, T2, &ip);  // NOTE: this updates FET.
    MFEM_VERIFY(!FET.Elem2,
                "FarField computations are only supported on external boundaries.")

    T.Transform(ip, r_phys);

    // Evaluate E and B fields on this element.
    E.Real().GetVectorValue(*FET.Elem1, FET.Elem1->GetIntPoint(), E_real);
    E.Imag().GetVectorValue(*FET.Elem1, FET.Elem1->GetIntPoint(), E_imag);
    B.Real().GetVectorValue(*FET.Elem1, FET.Elem1->GetIntPoint(), B_real);
    B.Imag().GetVectorValue(*FET.Elem1, FET.Elem1->GetIntPoint(), B_imag);

    // We assume that the material is isotropic, so the wave speed is a scalar.
    double wave_speed = mat_op.GetLightSpeedMax(FET.Elem1->Attribute);
    double k_re = omega_re / wave_speed;
    double k_im = omega_im / wave_speed;
    double quadrature_weight = ip.weight * T.Weight();

    // Complex prefactor: (ik/4π) = (i(k_re + ik_im)/4π) = (ik_re - k_im)/4π.
    double prefactor_re = -quadrature_weight * k_im / (4 * std::numbers::pi);
    double prefactor_im = quadrature_weight * k_re / (4 * std::numbers::pi);

    // Z * H = c0 * B.
    mat_op.GetLightSpeed(FET.Elem1->Attribute).Mult(B_real, ZH_real);
    mat_op.GetLightSpeed(FET.Elem1->Attribute).Mult(B_imag, ZH_imag);
    BdrGridFunctionCoefficient::GetNormal(T, normal, invert);

    // n̂ × E.
    linalg::Cross3(normal, E_real, n_cross_Er);
    linalg::Cross3(normal, E_imag, n_cross_Ei);

    // n̂ × ZH.
    linalg::Cross3(normal, ZH_real, n_cross_ZHr);
    linalg::Cross3(normal, ZH_imag, n_cross_ZHi);

    // This is a hot loop. Manually unrolling and avoiding Vectors significantly
    // increases performance.
    PalacePragmaOmp(parallel for schedule(static))
    for (std::size_t i = 0; i < r_naughts.size(); i++)
    {
      const auto &r = r_naughts[i];

      double r0 = r[0], r1 = r[1], r2 = r[2];
      double dot_product = r0 * r_phys(0) + r1 * r_phys(1) + r2 * r_phys(2);

      // Complex phase: exp(i*k*r₀·r') = exp(i*(k_re + ik_im)*dot_product)
      //                               = exp(i*k_re*dot_product - k_im*dot_product)
      //                               = exp(-k_im*dot_product) * exp(i*k_re*dot_product).
      double amplitude = std::exp(-k_im * dot_product);
      double phase_re = k_re * dot_product;
      double cos_phase = std::cos(phase_re);
      double sin_phase = std::sin(phase_re);

      // Complex weight: prefactor * exp(i*k*r₀·r').
      double w_real = amplitude * (prefactor_re * cos_phase - prefactor_im * sin_phase);
      double w_imag = amplitude * (prefactor_re * sin_phase + prefactor_im * cos_phase);

      // r₀ × (n̂ × ZH).
      double cr0 = r1 * n_cross_ZHr(2) - r2 * n_cross_ZHr(1);
      double cr1 = r2 * n_cross_ZHr(0) - r0 * n_cross_ZHr(2);
      double cr2 = r0 * n_cross_ZHr(1) - r1 * n_cross_ZHr(0);

      double ci0 = r1 * n_cross_ZHi(2) - r2 * n_cross_ZHi(1);
      double ci1 = r2 * n_cross_ZHi(0) - r0 * n_cross_ZHi(2);
      double ci2 = r0 * n_cross_ZHi(1) - r1 * n_cross_ZHi(0);

      // Complex multiplication: (A + iB) * (w_real + i*w_imag).
      double A0 = n_cross_Er(0) - cr0, B0 = n_cross_Ei(0) - ci0;
      double A1 = n_cross_Er(1) - cr1, B1 = n_cross_Ei(1) - ci1;
      double A2 = n_cross_Er(2) - cr2, B2 = n_cross_Ei(2) - ci2;

      integrand_r[i][0] += A0 * w_real - B0 * w_imag;
      integrand_r[i][1] += A1 * w_real - B1 * w_imag;
      integrand_r[i][2] += A2 * w_real - B2 * w_imag;

      integrand_i[i][0] += A0 * w_imag + B0 * w_real;
      integrand_i[i][1] += A1 * w_imag + B1 * w_real;
      integrand_i[i][2] += A2 * w_imag + B2 * w_real;
    }
  }
}

void AddStrattonChuIntegrand2DAtElement(const GridFunction &E, const GridFunction &B,
                                        double omega,
                                        const std::vector<std::array<double, 3>> &r_naughts,
                                        mfem::ElementTransformation &T,
                                        const mfem::IntegrationRule &ir,
                                        std::vector<std::array<double, 3>> &integrand_r,
                                        std::vector<std::array<double, 3>> &integrand_i)
{
  // 2D far-field transform from the equivalence principle. With the out-of-plane magnetic
  // field H_z and the in-plane electric field E, the equivalent line currents on the
  // integration contour are J = n x H and M = -n x E, and the far-field amplitude is
  //   E_amp = e^{-i pi/4}/sqrt(8 pi k) * i k \int [ (n x H) + (n x E) x r_hat ]
  //                                          e^{i k r_hat . r'} dl,
  // so that sigma_2D = 2 pi |E_amp|^2 / |E_inc|^2.
  MFEM_VERIFY(E.VectorDim() == 2,
              "2D far-field computation requires a two-component electric field!");
  MFEM_VERIFY(B.VectorDim() == 1,
              "2D far-field computation requires a scalar magnetic field!");
  MFEM_CONTRACT_VAR(B);  // H_z is recovered from the curl of E (see below).
  MFEM_ASSERT(T.ElementType == mfem::ElementTransformation::BDR_ELEMENT,
              "Unexpected element type in AddStrattonChuIntegrand2DAtElement!");
  MFEM_VERIFY(integrand_r.size() == r_naughts.size() &&
                  integrand_i.size() == r_naughts.size(),
              "Mismatch between input points and result vector.")

  const mfem::ParMesh *mesh = E.Real().ParFESpace()->GetParMesh();
  mfem::FaceElementTransformations FET;
  mfem::IsoparametricTransformation T1, T2;
  mfem::Vector r_phys(2), E_real(2), E_imag(2), normal(2);

  for (int j = 0; j < ir.GetNPoints(); j++)
  {
    const mfem::IntegrationPoint &ip = ir.IntPoint(j);
    T.SetIntPoint(&ip);

    const bool invert = BdrGridFunctionCoefficient::GetBdrElementNeighborTransformations(
        T.ElementNo, *mesh, FET, T1, T2, &ip);  // NOTE: this updates FET.
    MFEM_VERIFY(!FET.Elem2,
                "FarField computations are only supported on external boundaries.")

    T.Transform(ip, r_phys);

    // Evaluate E on the adjacent volume element. In 2D the out-of-plane magnetic field is
    // stored in an L2 space using an integral (moment) basis, so instead of evaluating B
    // at the point we recover H_z = B_z directly from the curl of E,
    // B = -1/(iω) ∇×E = (i/ω) ∇×E, as computed by the driven solver.
    E.Real().GetVectorValue(*FET.Elem1, FET.Elem1->GetIntPoint(), E_real);
    E.Imag().GetVectorValue(*FET.Elem1, FET.Elem1->GetIntPoint(), E_imag);
    mfem::Vector curl_real(1), curl_imag(1);
    E.Real().GetCurl(*FET.Elem1, curl_real);
    E.Imag().GetCurl(*FET.Elem1, curl_imag);
    const std::complex<double> Hz(-curl_imag(0) / omega, curl_real(0) / omega);
    BdrGridFunctionCoefficient::GetNormal(T, normal, invert);

    const double quadrature_weight = ip.weight * T.Weight();
    // Prefactor i k exp(-i pi/4) / sqrt(8 pi k) = sqrt(k / (8 pi)) * (1 + i) / sqrt(2),
    // with k = omega in internal units (c0 = 1).
    const double C = quadrature_weight * std::sqrt(omega / (8.0 * M_PI));
    const std::complex<double> pref(C / std::sqrt(2.0), C / std::sqrt(2.0));

    const std::complex<double> Ex(E_real(0), E_imag(0));
    const std::complex<double> Ey(E_real(1), E_imag(1));
    // (n x E)_z.
    const std::complex<double> nxE_z = normal(0) * Ey - normal(1) * Ex;

    PalacePragmaOmp(parallel for schedule(static))
    for (std::size_t i = 0; i < r_naughts.size(); i++)
    {
      const auto &r = r_naughts[i];
      const double dot_product = r[0] * r_phys(0) + r[1] * r_phys(1);
      const std::complex<double> phase(std::cos(omega * dot_product),
                                       std::sin(omega * dot_product));
      const std::complex<double> w = pref * phase;
      // n x H + (n x E) x r_hat.
      const std::complex<double> F0 = normal(1) * Hz + nxE_z * r[1];
      const std::complex<double> F1 = -normal(0) * Hz - nxE_z * r[0];
      const std::complex<double> v0 = w * F0;
      const std::complex<double> v1 = w * F1;
      integrand_r[i][0] += v0.real();
      integrand_r[i][1] += v1.real();
      integrand_i[i][0] += v0.imag();
      integrand_i[i][1] += v1.imag();
    }
  }
}

};  // namespace palace
