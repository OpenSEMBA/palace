// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#ifndef PALACE_MODELS_STRATTONCHU_HPP
#define PALACE_MODELS_STRATTONCHU_HPP

#include <vector>
#include <mfem.hpp>
#include <linalg/vector.hpp>
#include "fem/gridfunction.hpp"
#include "models/materialoperator.hpp"

namespace palace
{
void AddStrattonChuIntegrandAtElement(const GridFunction &E, const GridFunction &B,
                                      const MaterialOperator &mat_op, double omega_re,
                                      double omega_im,
                                      std::vector<std::array<double, 3>> &r_naughts,
                                      mfem::ElementTransformation &T,
                                      const mfem::IntegrationRule &ir,
                                      std::vector<std::array<double, 3>> &integrand_r,
                                      std::vector<std::array<double, 3>> &integrand_i);

// Two-dimensional version using the equivalence principle. Computes the in-plane
// far-field amplitude
//   E_amp(r_hat) = e^{-i pi/4}/sqrt(8 pi k) * i k
//                  * \int_C [ (n x H) + (n x E) x r_hat ] e^{i k r_hat . r'} dl
// from the out-of-plane magnetic field H_z = B (B has a single component in 2D) and the
// in-plane electric field E. The 2D RCS is sigma(r_hat) = 2 pi |E_amp|^2 / |E_inc|^2.
void AddStrattonChuIntegrand2DAtElement(const GridFunction &E, const GridFunction &B,
                                        double omega,
                                        const std::vector<std::array<double, 3>> &r_naughts,
                                        mfem::ElementTransformation &T,
                                        const mfem::IntegrationRule &ir,
                                        std::vector<std::array<double, 3>> &integrand_r,
                                        std::vector<std::array<double, 3>> &integrand_i);

}  // namespace palace

#endif  // PALACE_MODELS_STRATTONCHU_HPP
