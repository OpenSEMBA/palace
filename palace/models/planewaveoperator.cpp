// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include "planewaveoperator.hpp"

#include <cmath>
#include <fmt/ranges.h>
#include "fem/fespace.hpp"
#include "fem/gridfunction.hpp"
#include "utils/communication.hpp"
#include "utils/iodata.hpp"
#include "utils/units.hpp"

namespace palace
{

PlaneWaveData::PlaneWaveData(const config::PlaneWaveData &data, const Units &units)
  : direction(data.direction), polarization(data.polarization),
    amplitude(units.Nondimensionalize<Units::ValueType::FIELD_E>(data.amplitude))
{
  // Configuration parsing guarantees unit length and orthogonality, but be defensive since
  // the data may be constructed programmatically.
  auto Normalize = [](std::array<double, 3> &v)
  {
    const double norm = std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
    MFEM_VERIFY(norm > 0.0, "Plane wave direction and polarization must be nonzero!");
    for (auto &x : v)
    {
      x /= norm;
    }
  };
  Normalize(direction);
  Normalize(polarization);
  const double k_dot_p = direction[0] * polarization[0] + direction[1] * polarization[1] +
                         direction[2] * polarization[2];
  MFEM_VERIFY(std::abs(k_dot_p) < 1.0e-6,
              "Plane wave polarization must be orthogonal to its propagation direction!");
  MFEM_VERIFY(amplitude > 0.0, "Plane wave amplitude must be positive!");
}

std::array<std::complex<double>, 3> PlaneWaveData::E(const double *x, double k) const
{
  const double phase =
      -k * (direction[0] * x[0] + direction[1] * x[1] + direction[2] * x[2]);
  const std::complex<double> f = amplitude * std::polar(1.0, phase);
  return {f * polarization[0], f * polarization[1], f * polarization[2]};
}

std::array<std::complex<double>, 3> PlaneWaveData::B(const double *x, double k,
                                                     double omega) const
{
  // Faraday's law with exp(+iωt): ∇ × E = -iω B. With E = E₀ p̂ exp(-i k k̂·x), we have
  // ∇ × E = -i k k̂ × E, so B = (k / ω) k̂ × E.
  const auto e = E(x, k);
  const double s = k / omega;
  return {s * (direction[1] * e[2] - direction[2] * e[1]),
          s * (direction[2] * e[0] - direction[0] * e[2]),
          s * (direction[0] * e[1] - direction[1] * e[0])};
}

PlaneWaveOperator::PlaneWaveOperator(const IoData &iodata)
  : PlaneWaveOperator(iodata.domains.plane_wave, iodata.problem.type, iodata.units)
{
}

PlaneWaveOperator::PlaneWaveOperator(const std::map<int, config::PlaneWaveData> &plane_wave,
                                     ProblemType problem_type, const Units &units)
{
  MFEM_VERIFY(plane_wave.empty() || problem_type == ProblemType::DRIVEN,
              "Incident plane wave excitations are only available for driven simulations!");
  for (const auto &[idx, data] : plane_wave)
  {
    plane_waves.emplace(idx, PlaneWaveData(data, units));
  }
  PrintPlaneWaveInfo(units);
}

void PlaneWaveOperator::PrintPlaneWaveInfo(const Units &units) const
{
  if (plane_waves.empty())
  {
    return;
  }
  Mpi::Print("\nConfiguring incident plane wave excitations:\n");
  for (const auto &[idx, data] : plane_waves)
  {
    Mpi::Print(" Plane wave {:d}:\n"
               " \tAmplitude = {:.3e} V/m\n"
               " \tDirection = ({:.3e})\n"
               " \tPolarization = ({:.3e})\n",
               idx, units.Dimensionalize<Units::ValueType::FIELD_E>(data.amplitude),
               fmt::join(data.direction, ", "), fmt::join(data.polarization, ", "));
  }
}

const PlaneWaveData &PlaneWaveOperator::GetPlaneWave(int excitation_idx) const
{
  auto it = plane_waves.find(excitation_idx);
  MFEM_VERIFY(it != plane_waves.end(),
              "Unknown plane wave excitation index " << excitation_idx << "!");
  return it->second;
}

bool PlaneWaveOperator::GetIncidentField(int excitation_idx, double k,
                                         FiniteElementSpace &nd_fespace,
                                         ComplexVector &E_inc) const
{
  E_inc.SetSize(nd_fespace.GetTrueVSize());
  E_inc.UseDevice(true);
  E_inc = 0.0;
  auto it = plane_waves.find(excitation_idx);
  if (it == plane_waves.end())
  {
    return false;
  }
  const auto &data = it->second;
  const int dim = nd_fespace.SpaceDimension();
  MFEM_VERIFY(dim == 3, "Incident plane waves are only available for 3D simulations!");

  // Interpolate the incident field into the Nédélec space. The interpolant (rather than,
  // for example, the L² projection) is used so that the discrete incident field is local
  // and cheap to evaluate.
  auto Component = [&data, k](bool imag)
  {
    return mfem::VectorFunctionCoefficient(
        3,
        [&data, k, imag](const mfem::Vector &x, mfem::Vector &e)
        {
          const auto E = data.E(x.GetData(), k);
          for (int d = 0; d < 3; d++)
          {
            e(d) = imag ? E[d].imag() : E[d].real();
          }
        });
  };
  auto Er = Component(false), Ei = Component(true);
  GridFunction E(nd_fespace, true);
  E.Real().ProjectCoefficient(Er);
  E.Imag().ProjectCoefficient(Ei);
  E.Real().ParallelProject(E_inc.Real());
  E.Imag().ParallelProject(E_inc.Imag());
  return true;
}

}  // namespace palace
