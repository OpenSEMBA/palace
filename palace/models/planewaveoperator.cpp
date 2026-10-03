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
  bool include_invz;

public:
  PlaneWaveVectorCoefficient(const PlaneWaveData &data, double omega, bool imag,
                             bool include_invz)
    : mfem::VectorCoefficient(3), data(data), omega(omega), imag(imag),
      include_invz(include_invz)
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
  data.GetElectricField(x, omega, T.Attribute, E);
  const double invz = (T.Attribute > 0 && T.Attribute <= data.attr_invz.Size())
                          ? data.attr_invz[T.Attribute - 1]
                          : 1.0;
  V.SetSize(3);
  for (int d = 0; d < 3; d++)
  {
    const std::complex<double> v = (include_invz ? invz : 1.0) * E[d];
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
  : excitation(data.excitation), direction(3), polarization(3), amplitude(0.0), origin(3)
{
  for (int d = 0; d < 3; d++)
  {
    direction(d) = data.direction[d];
    polarization(d) = data.polarization[d];
    origin(d) = data.origin[d];
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

  // Per-boundary-attribute local wave speed and inverse impedance, taken from the adjacent
  // domain element material. The incident plane wave is defined in this local medium.
  attr_c.SetSize(bdr_attr_max);
  attr_c = 1.0;
  attr_invz.SetSize(bdr_attr_max);
  attr_invz = 1.0;
  mfem::Array<int> attr_marker = AttrListToMarker(attr_list);
  for (int i = 0; i < mesh.GetNBE(); i++)
  {
    const int attr = mesh.GetBdrAttribute(i);
    if (attr <= 0 || attr > bdr_attr_max || !attr_marker[attr - 1])
    {
      continue;
    }
    int elem_id, face_id;
    mesh.GetBdrElementAdjacentElement(i, elem_id, face_id);
    if (elem_id < 0)
    {
      continue;
    }
    const int dom_attr = mesh.GetAttribute(elem_id);
    MFEM_VERIFY(mat_op.IsIsotropic(dom_attr),
                "Plane wave absorbing boundaries require isotropic materials, but "
                "attribute "
                    << dom_attr << " is not!");
    attr_c[attr - 1] = mat_op.GetLightSpeedMax(dom_attr);
    attr_invz[attr - 1] = mat_op.GetInvImpedance(dom_attr)(0, 0);
  }
}

void PlaneWaveData::GetElectricField(const mfem::Vector &x, double omega, int attr,
                                     std::complex<double> E[3]) const
{
  const double c =
      (attr > 0 && attr <= attr_c.Size()) ? attr_c[attr - 1] : 1.0;  // Internal c = 1.
  const double k = omega / c;
  double phase = 0.0;
  for (int d = 0; d < 3; d++)
  {
    phase += direction(d) * (x(d) - origin(d));
  }
  // Phasor exp(+i k k_hat . x): with Palace's internal time convention this
  // propagates along +k_hat, so Direction is the propagation direction,
  // consistent with the far-field observation angle convention (the
  // backscatter/monostatic direction is -k_hat).
  phase *= k;
  const std::complex<double> e = amplitude * std::exp(std::complex<double>(0.0, phase));
  for (int d = 0; d < 3; d++)
  {
    E[d] = e * polarization(d);
  }
}

void PlaneWaveData::GetElectricField(const mfem::Vector &x, double omega,
                                     std::complex<double> E[3]) const
{
  GetElectricField(x, omega, -1, E);
}

PlaneWaveOperator::PlaneWaveOperator(
    const std::map<int, config::PlaneWaveData> &planewave, const MaterialOperator &mat_op,
    const mfem::ParMesh &mesh)
{
  MFEM_VERIFY(mesh.SpaceDimension() == 3,
              "Plane wave excitation is only available for 3D simulations!");
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
    Mpi::Print(" {:d}: Index = {:d}, k = ({:+.3f}, {:+.3f}, {:+.3f}), "
               "E = ({:+.3f}, {:+.3f}, {:+.3f}), Excitation = {:d}\n",
               (data.attr_list.Size() > 0) ? data.attr_list[0] : 0, idx, data.direction(0),
               data.direction(1), data.direction(2), data.polarization(0),
               data.polarization(1), data.polarization(2), data.excitation);
  }
}

const PlaneWaveData &PlaneWaveOperator::GetWave(int idx) const
{
  auto it = waves.find(idx);
  MFEM_VERIFY(it != waves.end(), "Unknown plane wave index requested!");
  return it->second;
}

void PlaneWaveOperator::AddExcitationBdrCoefficients(int excitation_idx, double omega,
                                                     SumVectorCoefficient &fbr,
                                                     SumVectorCoefficient &fbi) const
{
  for (const auto &[idx, data] : waves)
  {
    if (data.excitation != excitation_idx)
    {
      continue;
    }
    fbr.AddCoefficient(
        std::make_unique<PlaneWaveVectorCoefficient>(data, omega, false, true));
    fbi.AddCoefficient(
        std::make_unique<PlaneWaveVectorCoefficient>(data, omega, true, true));
  }
}

void PlaneWaveOperator::AddFieldCoefficients(int excitation_idx, double omega,
                                             SumVectorCoefficient &fr,
                                             SumVectorCoefficient &fi) const
{
  for (const auto &[idx, data] : waves)
  {
    if (data.excitation != excitation_idx)
    {
      continue;
    }
    fr.AddCoefficient(
        std::make_unique<PlaneWaveVectorCoefficient>(data, omega, false, false));
    fi.AddCoefficient(
        std::make_unique<PlaneWaveVectorCoefficient>(data, omega, true, false));
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
