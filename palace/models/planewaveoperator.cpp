// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include "planewaveoperator.hpp"

#include <algorithm>
#include "fem/coefficient.hpp"
#include "models/materialoperator.hpp"
#include "utils/communication.hpp"
#include "utils/geodata.hpp"

namespace palace
{

namespace
{

// Vector coefficient evaluating the real or imaginary part of the incident electric field
// (1/Z) E_inc on the absorbing boundary.
class PlaneWaveVectorCoefficient : public mfem::VectorCoefficient
{
private:
  const PlaneWaveData &data;
  double omega;
  bool imag;
  double scale;
  const std::vector<std::complex<double>> *attr_scale;

public:
  PlaneWaveVectorCoefficient(const PlaneWaveData &data, double omega, bool imag,
                             double scale = 1.0,
                             const std::vector<std::complex<double>> *attr_scale = nullptr)
    : mfem::VectorCoefficient(data.dim), data(data), omega(omega), imag(imag), scale(scale),
      attr_scale(attr_scale)
  {
  }

  void Eval(mfem::Vector &V, mfem::ElementTransformation &T,
            const mfem::IntegrationPoint &ip) override;
};

void PlaneWaveVectorCoefficient::Eval(mfem::Vector &V, mfem::ElementTransformation &T,
                                      const mfem::IntegrationPoint &ip)
{
  mfem::Vector x(3);
  T.Transform(ip, x);
  std::complex<double> E[3];
  data.GetElectricField(x, omega, E);
  std::complex<double> s = scale;
  if (attr_scale && T.Attribute > 0 && T.Attribute < static_cast<int>(attr_scale->size()))
  {
    s *= (*attr_scale)[T.Attribute];
  }
  V.SetSize(data.dim);
  for (int d = 0; d < data.dim; d++)
  {
    const std::complex<double> v = s * E[d];
    V(d) = imag ? v.imag() : v.real();
  }
}

// Build a boundary attribute marker array of size max(attr) with the listed attributes
// marked.
mfem::Array<int> AttrListToMarker(const mfem::Array<int> &attr_list)
{
  int max_attr = 0;
  for (int i = 0; i < attr_list.Size(); i++)
  {
    max_attr = std::max(max_attr, attr_list[i]);
  }
  mfem::Array<int> marker;
  if (max_attr == 0)
  {
    return marker;
  }
  marker.SetSize(max_attr);
  marker = 0;
  for (int i = 0; i < attr_list.Size(); i++)
  {
    marker[attr_list[i] - 1] = 1;
  }
  return marker;
}

}  // namespace

PlaneWaveData::PlaneWaveData(const config::PlaneWaveData &data,
                             const MaterialOperator &mat_op, const mfem::ParMesh &mesh)
  : excitation(data.excitation), dim(mesh.SpaceDimension()), direction(dim),
    polarization(dim), amplitude(0.0), origin(dim)
{
  MFEM_VERIFY(dim == 2 || dim == 3,
              "Plane wave excitation is only available for 2D and 3D simulations!");
  for (int d = 0; d < dim; d++)
  {
    direction(d) = data.direction[d];
    polarization(d) = data.polarization[d];
    origin(d) = data.origin[d];
  }
  if (dim == 2)
  {
    MFEM_VERIFY(std::abs(data.direction[2]) < 1.0e-14 &&
                    std::abs(data.polarization[2]) < 1.0e-14,
                "Plane wave direction and polarization must be in-plane for 2D "
                "simulations!");
  }
  // The config amplitude is already nondimensionalized in NondimensionalizeInputs.
  amplitude = data.amplitude * std::exp(std::complex<double>(0.0, data.phase));

  // Collect the valid boundary attributes.
  const int bdr_attr_max = mesh.bdr_attributes.Size() ? mesh.bdr_attributes.Max() : 0;
  mfem::Array<int> bdr_attr_marker;
  if (bdr_attr_max > 0)
  {
    bdr_attr_marker.SetSize(bdr_attr_max);
    bdr_attr_marker = 0;
    for (auto attr : mesh.bdr_attributes)
    {
      bdr_attr_marker[attr - 1] = 1;
    }
  }
  attr_list.Reserve(static_cast<int>(data.attributes.size()));
  for (auto attr : data.attributes)
  {
    if (attr > 0 && attr <= bdr_attr_max && bdr_attr_marker[attr - 1])
    {
      attr_list.Append(attr);
    }
    else
    {
      Mpi::Warning("Unknown plane wave boundary attribute {:d}!\n", attr);
    }
  }
}

void PlaneWaveData::GetElectricField(const mfem::Vector &x, double omega,
                                     std::complex<double> E[3]) const
{
  // Background medium is internal vacuum, c0 = 1. The scattered-field formulation
  // solves for E_s with the free-space incident field, so the material response is
  // carried by E_s and the incident field is independent of the local material.
  const double k = omega;
  double phase = 0.0;
  for (int d = 0; d < dim; d++)
  {
    phase += direction(d) * (x(d) - origin(d));
  }
  // Phasor exp(-i k k_hat . x): under Palace's exp(+iωt) convention the wave travels
  // along +Direction, so the configured Direction is the propagation direction and the
  // monostatic/backscatter direction is -Direction.
  phase *= -k;
  const std::complex<double> e = amplitude * std::exp(std::complex<double>(0.0, phase));
  for (int d = 0; d < dim; d++)
  {
    E[d] = e * polarization(d);
  }
  for (int d = dim; d < 3; d++)
  {
    E[d] = 0.0;
  }
}

PlaneWaveOperator::PlaneWaveOperator(const std::map<int, config::PlaneWaveData> &planewave,
                                     const MaterialOperator &mat_op,
                                     const mfem::ParMesh &mesh)
{
  MFEM_VERIFY(mesh.SpaceDimension() == 2 || mesh.SpaceDimension() == 3,
              "Plane wave excitation is only available for 2D and 3D simulations!");
  SetUpBoundaryProperties(planewave, mat_op, mesh);
  PrintBoundaryInfo(mesh);
}

void PlaneWaveOperator::SetUpBoundaryProperties(
    const std::map<int, config::PlaneWaveData> &planewave, const MaterialOperator &mat_op,
    const mfem::ParMesh &mesh)
{
  for (const auto &[idx, data] : planewave)
  {
    waves.try_emplace(idx, data, mat_op, mesh);
  }
}

void PlaneWaveOperator::PrintBoundaryInfo(const mfem::ParMesh &mesh)
{
  if (waves.empty())
  {
    return;
  }
  Mpi::Print("\nConfiguring incident plane wave excitation at attributes:\n");
  for (const auto &[idx, data] : waves)
  {
    const int attr = (data.attr_list.Size() > 0) ? data.attr_list[0] : 0;
    if (data.dim == 2)
    {
      Mpi::Print(" {:d}: Index = {:d}, k = ({:+.3f}, {:+.3f}), "
                 "E = ({:+.3f}, {:+.3f}), Excitation = {:d}\n",
                 attr, idx, data.direction(0), data.direction(1), data.polarization(0),
                 data.polarization(1), data.excitation);
    }
    else
    {
      Mpi::Print(" {:d}: Index = {:d}, k = ({:+.3f}, {:+.3f}, {:+.3f}), "
                 "E = ({:+.3f}, {:+.3f}, {:+.3f}), Excitation = {:d}\n",
                 attr, idx, data.direction(0), data.direction(1), data.direction(2),
                 data.polarization(0), data.polarization(1), data.polarization(2),
                 data.excitation);
    }
  }
}

const PlaneWaveData &PlaneWaveOperator::GetWave(int idx) const
{
  auto it = waves.find(idx);
  MFEM_VERIFY(it != waves.end(), "Unknown plane wave index requested!");
  return it->second;
}

void PlaneWaveOperator::AddFieldCoefficients(
    int excitation_idx, double omega, SumVectorCoefficient &fr, SumVectorCoefficient &fi,
    double scale, const std::vector<std::complex<double>> *attr_scale) const
{
  for (const auto &[idx, data] : waves)
  {
    if (data.excitation != excitation_idx)
    {
      continue;
    }
    fr.AddCoefficient(std::make_unique<PlaneWaveVectorCoefficient>(data, omega, false,
                                                                   scale, attr_scale));
    fi.AddCoefficient(
        std::make_unique<PlaneWaveVectorCoefficient>(data, omega, true, scale, attr_scale));
  }
}

const PlaneWaveData *PlaneWaveOperator::GetExcitationWave(int excitation_idx) const
{
  for (const auto &[idx, data] : waves)
  {
    if (data.excitation == excitation_idx)
    {
      return &data;
    }
  }
  return nullptr;
}

mfem::Array<int> PlaneWaveOperator::GetExcitationBdrMarker(int excitation_idx) const
{
  int max_attr = 0;
  for (const auto &[idx, data] : waves)
  {
    if (data.excitation != excitation_idx)
    {
      continue;
    }
    for (int i = 0; i < data.attr_list.Size(); i++)
    {
      max_attr = std::max(max_attr, data.attr_list[i]);
    }
  }
  mfem::Array<int> marker;
  if (max_attr == 0)
  {
    return marker;
  }
  marker.SetSize(max_attr);
  marker = 0;
  for (const auto &[idx, data] : waves)
  {
    if (data.excitation != excitation_idx)
    {
      continue;
    }
    for (int i = 0; i < data.attr_list.Size(); i++)
    {
      marker[data.attr_list[i] - 1] = 1;
    }
  }
  return marker;
}

mfem::Array<int> PlaneWaveOperator::GetAttrMarker() const
{
  int max_attr = 0;
  for (const auto &[idx, data] : waves)
  {
    for (int i = 0; i < data.attr_list.Size(); i++)
    {
      max_attr = std::max(max_attr, data.attr_list[i]);
    }
  }
  mfem::Array<int> marker;
  if (max_attr == 0)
  {
    return marker;
  }
  marker.SetSize(max_attr);
  marker = 0;
  for (const auto &[idx, data] : waves)
  {
    for (int i = 0; i < data.attr_list.Size(); i++)
    {
      marker[data.attr_list[i] - 1] = 1;
    }
  }
  return marker;
}

}  // namespace palace
