// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include "surfacepostoperator.hpp"

#include <complex>
#include <set>
#include <string>
#include "fem/gridfunction.hpp"
#include "fem/integrator.hpp"
#include "linalg/vector.hpp"
#include "models/materialoperator.hpp"
#include "models/planewaveoperator.hpp"
#include "models/strattonchu.hpp"
#include "utils/communication.hpp"
#include "utils/geodata.hpp"
#include "utils/iodata.hpp"
#include "utils/prettyprint.hpp"
#include "utils/timer.hpp"

namespace palace
{

namespace
{

bool IsSupportedSurfaceFunctionalDimension(const mfem::ParMesh &mesh)
{
  return (mesh.Dimension() == 2 && mesh.SpaceDimension() == 2) ||
         (mesh.Dimension() == 3 && mesh.SpaceDimension() == 3);
}

bool IsSupportedSurfaceFluxDimension(const mfem::ParMesh &mesh)
{
  // The libCEED surface-flux kernels currently implement 3D surface integrals. 2D line
  // flux semantics remain on the legacy path explicitly rather than by silent invalid
  // fallback.
  return mesh.Dimension() == 3 && mesh.SpaceDimension() == 3;
}

void RequireCeedSurfaceFunctional(const SurfaceFunctional *func, const std::string &what)
{
  MFEM_VERIFY(func && func->IsValid(), "libCEED postprocessing was expected for " + what +
                                           ", but SurfaceFunctional could not assemble!");
}

// Computes the contribution of the analytic incident plane wave to the Stratton-Chu
// far-field integrands, mirroring AddStrattonChuIntegrandAtElement with E -> E_inc and
// Z H -> k̂ × E_inc.
void AddStrattonChuIncidentIntegrandAtElement(
    const PlaneWaveData &incident, std::complex<double> omega,
    const std::vector<std::array<double, 3>> &r_naughts, const mfem::ParMesh &mesh,
    mfem::ElementTransformation &T, const mfem::IntegrationRule &ir,
    std::vector<std::array<double, 3>> &integrand_r,
    std::vector<std::array<double, 3>> &integrand_i)
{
  const double omega_re = omega.real();
  const double omega_im = omega.imag();
  mfem::Vector r_phys(3), normal(3);
  for (int j = 0; j < ir.GetNPoints(); j++)
  {
    const mfem::IntegrationPoint &ip = ir.IntPoint(j);
    T.SetIntPoint(&ip);

    mfem::FaceElementTransformations FET;
    mfem::IsoparametricTransformation T1, T2;
    const bool invert = BdrGridFunctionCoefficient::GetBdrElementNeighborTransformations(
        T.ElementNo, mesh, FET, T1, T2, &ip);
    MFEM_VERIFY(!FET.Elem2,
                "FarField computations are only supported on external boundaries.")
    T.Transform(ip, r_phys);

    std::complex<double> E[3];
    incident.GetElectricField(r_phys, omega_re, T.Attribute, E);

    // Z H_inc = k̂ × E_inc for a plane wave in a homogeneous medium.
    std::complex<double> ZH[3];
    ZH[0] = incident.direction(1) * E[2] - incident.direction(2) * E[1];
    ZH[1] = incident.direction(2) * E[0] - incident.direction(0) * E[2];
    ZH[2] = incident.direction(0) * E[1] - incident.direction(1) * E[0];

    BdrGridFunctionCoefficient::GetNormal(T, normal, invert);
    std::complex<double> n_cross_E[3], n_cross_ZH[3];
    for (int d = 0; d < 3; d++)
    {
      const int d1 = (d + 1) % 3, d2 = (d + 2) % 3;
      n_cross_E[d] = normal(d1) * E[d2] - normal(d2) * E[d1];
      n_cross_ZH[d] = normal(d1) * ZH[d2] - normal(d2) * ZH[d1];
    }

    const double wave_speed =
        (T.Attribute > 0 && T.Attribute <= incident.attr_c.Size())
            ? incident.attr_c[T.Attribute - 1]
            : 1.0;
    const double k_re = omega_re / wave_speed;
    const double k_im = omega_im / wave_speed;
    const double quadrature_weight = ip.weight * T.Weight();
    const double prefactor_re = -quadrature_weight * k_im / (4 * M_PI);
    const double prefactor_im = quadrature_weight * k_re / (4 * M_PI);

    for (std::size_t i = 0; i < r_naughts.size(); i++)
    {
      const auto &r = r_naughts[i];
      const double dot_product =
          r[0] * r_phys(0) + r[1] * r_phys(1) + r[2] * r_phys(2);
      const double amplitude = std::exp(-k_im * dot_product);
      const double phase_re = k_re * dot_product;
      const double cos_phase = std::cos(phase_re);
      const double sin_phase = std::sin(phase_re);
      const double w_real =
          amplitude * (prefactor_re * cos_phase - prefactor_im * sin_phase);
      const double w_imag =
          amplitude * (prefactor_re * sin_phase + prefactor_im * cos_phase);

      // r₀ × (n̂ × ZH).
      std::complex<double> cr[3];
      cr[0] = r[1] * n_cross_ZH[2] - r[2] * n_cross_ZH[1];
      cr[1] = r[2] * n_cross_ZH[0] - r[0] * n_cross_ZH[2];
      cr[2] = r[0] * n_cross_ZH[1] - r[1] * n_cross_ZH[0];

      for (int d = 0; d < 3; d++)
      {
        const std::complex<double> A = n_cross_E[d] - cr[d];
        integrand_r[i][d] += A.real() * w_real - A.imag() * w_imag;
        integrand_i[i][d] += A.real() * w_imag + A.imag() * w_real;
      }
    }
  }
}

template <typename T>
mfem::Array<int> SetUpBoundaryProperties(const T &data,
                                         const mfem::Array<int> &bdr_attr_marker)
{
  mfem::Array<int> attr_list;
  attr_list.Reserve(static_cast<int>(data.attributes.size()));
  std::set<int> bdr_warn_list;
  for (auto attr : data.attributes)
  {
    // MFEM_VERIFY(attr > 0 && attr <= bdr_attr_max,
    //             "Boundary postprocessing attribute tags must be non-negative and "
    //             "correspond to attributes in the mesh!");
    // MFEM_VERIFY(bdr_attr_marker[attr - 1],
    //             "Unknown boundary postprocessing attribute " << attr << "!");
    if (attr <= 0 || attr > bdr_attr_marker.Size() || !bdr_attr_marker[attr - 1])
    {
      bdr_warn_list.insert(attr);
    }
    else
    {
      attr_list.Append(attr);
    }
  }
  if (!bdr_warn_list.empty())
  {
    Mpi::Print("\n");
    Mpi::Warning(
        "Unknown boundary postprocessing attributes!\nSolver will just ignore them!");
    utils::PrettyPrint(bdr_warn_list, "Boundary attribute list:");
    Mpi::Print("\n");
  }
  return attr_list;
}

}  // namespace

SurfacePostOperator::SurfaceFluxData::SurfaceFluxData(
    const config::SurfaceFluxData &data, const mfem::ParMesh &mesh,
    const mfem::Array<int> &bdr_attr_marker)
{
  // Store boundary attributes for this postprocessing boundary.
  attr_list = SetUpBoundaryProperties(data, bdr_attr_marker);

  // Store the type of flux.
  switch (data.type)
  {
    case SurfaceFlux::ELECTRIC:
      type = SurfaceFlux::ELECTRIC;
      break;
    case SurfaceFlux::MAGNETIC:
      type = SurfaceFlux::MAGNETIC;
      break;
    case SurfaceFlux::POWER:
      type = SurfaceFlux::POWER;
      break;
  }

  // Store information about the global direction for orientation. Note the true boundary
  // normal is used in calculating the flux, this is just used to determine the sign.
  two_sided = data.two_sided;
  if (!two_sided)
  {
    center.SetSize(mesh.SpaceDimension());
    if (data.no_center)
    {
      // Compute the center as the bounding box centroid for all boundary elements making up
      // this postprocessing boundary.
      mfem::Vector bbmin, bbmax;
      mesh::GetAxisAlignedBoundingBox(
          mesh, mesh::AttrToMarker(bdr_attr_marker.Size(), attr_list), true, bbmin, bbmax);
      for (int d = 0; d < mesh.SpaceDimension(); d++)
      {
        center(d) = 0.5 * (bbmin(d) + bbmax(d));
      }
    }
    else
    {
      std::copy(data.center.begin(), data.center.end(), center.begin());
    }
  }
}

std::unique_ptr<mfem::Coefficient>
SurfacePostOperator::SurfaceFluxData::GetCoefficient(const mfem::ParGridFunction *E,
                                                     const mfem::ParGridFunction *B,
                                                     const MaterialOperator &mat_op) const
{
  switch (type)
  {
    case SurfaceFlux::ELECTRIC:
      return std::make_unique<
          RestrictedCoefficient<BdrSurfaceFluxCoefficient<SurfaceFlux::ELECTRIC>>>(
          attr_list, E, nullptr, mat_op, two_sided, center);
    case SurfaceFlux::MAGNETIC:
      return std::make_unique<
          RestrictedCoefficient<BdrSurfaceFluxCoefficient<SurfaceFlux::MAGNETIC>>>(
          attr_list, nullptr, B, mat_op, two_sided, center);
    case SurfaceFlux::POWER:
      return std::make_unique<
          RestrictedCoefficient<BdrSurfaceFluxCoefficient<SurfaceFlux::POWER>>>(
          attr_list, E, B, mat_op, two_sided, center);
  }
  return {};
}

SurfacePostOperator::InterfaceDielectricData::InterfaceDielectricData(
    const config::InterfaceDielectricData &data, const mfem::ParMesh &mesh,
    const mfem::Array<int> &bdr_attr_marker)
{
  // Store boundary attributes for this postprocessing boundary.
  attr_list = SetUpBoundaryProperties(data, bdr_attr_marker);

  // Calculate surface dielectric loss according to the formulas from J. Wenner et al.,
  // Surface loss simulations of superconducting coplanar waveguide resonators, Appl. Phys.
  // Lett. (2011). If only a general layer permittivity is specified and not any special
  // metal-air (MA), metal-substrate (MS), or substrate-air (SA) permittivity, compute the
  // numerator of the participation ratio according to the regular formula
  //                       p * E_elec = 1/2 t Re{∫ (ε E)ᴴ E_m dS} .
  switch (data.type)
  {
    case InterfaceDielectric::DEFAULT:
      type = InterfaceDielectric::DEFAULT;
      break;
    case InterfaceDielectric::MA:
      type = InterfaceDielectric::MA;
      break;
    case InterfaceDielectric::MS:
      type = InterfaceDielectric::MS;
      break;
    case InterfaceDielectric::SA:
      type = InterfaceDielectric::SA;
      break;
  }
  t = data.t;
  epsilon = data.epsilon_r;
  tandelta = data.tandelta;
}

std::unique_ptr<mfem::Coefficient>
SurfacePostOperator::InterfaceDielectricData::GetCoefficient(
    const GridFunction &E, const MaterialOperator &mat_op) const
{
  switch (type)
  {
    case InterfaceDielectric::DEFAULT:
      return std::make_unique<RestrictedCoefficient<
          InterfaceDielectricCoefficient<InterfaceDielectric::DEFAULT>>>(
          attr_list, E, mat_op, t, epsilon);
    case InterfaceDielectric::MA:
      return std::make_unique<
          RestrictedCoefficient<InterfaceDielectricCoefficient<InterfaceDielectric::MA>>>(
          attr_list, E, mat_op, t, epsilon);
    case InterfaceDielectric::MS:
      return std::make_unique<
          RestrictedCoefficient<InterfaceDielectricCoefficient<InterfaceDielectric::MS>>>(
          attr_list, E, mat_op, t, epsilon);
    case InterfaceDielectric::SA:
      return std::make_unique<
          RestrictedCoefficient<InterfaceDielectricCoefficient<InterfaceDielectric::SA>>>(
          attr_list, E, mat_op, t, epsilon);
  }
  return {};  // For compiler warning
}

SurfacePostOperator::FarFieldData::FarFieldData(const config::FarFieldPostData &data,
                                                const mfem::ParMesh &mesh,
                                                const mfem::Array<int> &bdr_attr_marker)
  : thetaphis(data.thetaphis), scattered(data.scattered), monostatic(data.monostatic)
{
  // Store boundary attributes for this postprocessing boundary.
  attr_list = SetUpBoundaryProperties(data, bdr_attr_marker);
}

SurfacePostOperator::SurfacePostOperator(const config::BoundaryPostData &postpro,
                                         ProblemType problem_type,
                                         const MaterialOperator &mat_op,
                                         mfem::ParFiniteElementSpace &h1_fespace,
                                         mfem::ParFiniteElementSpace &nd_fespace)
  : mat_op(mat_op), h1_fespace(h1_fespace), nd_fespace(nd_fespace)
{
  // Check that boundary attributes have been specified correctly.
  const auto &mesh = *h1_fespace.GetParMesh();
  int bdr_attr_max = mesh.bdr_attributes.Size() ? mesh.bdr_attributes.Max() : 0;
  mfem::Array<int> bdr_attr_marker;
  if (!postpro.flux.empty() || !postpro.dielectric.empty() || !postpro.farfield.empty())
  {
    bdr_attr_marker.SetSize(bdr_attr_max);
    bdr_attr_marker = 0;
    for (auto attr : mesh.bdr_attributes)
    {
      bdr_attr_marker[attr - 1] = 1;
    }
  }

  // Surface flux postprocessing.
  for (const auto &[idx, data] : postpro.flux)
  {
    MFEM_VERIFY(problem_type != ProblemType::ELECTROSTATIC ||
                    data.type == SurfaceFlux::ELECTRIC,
                "Magnetic field or power surface flux postprocessing are not available "
                "for electrostatic problems!");
    MFEM_VERIFY(problem_type != ProblemType::MAGNETOSTATIC ||
                    data.type == SurfaceFlux::MAGNETIC,
                "Electric field or power surface flux postprocessing are not available "
                "for magnetostatic problems!");
    flux_surfs.try_emplace(idx, data, *h1_fespace.GetParMesh(), bdr_attr_marker);
  }

  // Interface dielectric postprocessing.
  MFEM_VERIFY(postpro.dielectric.empty() || problem_type != ProblemType::MAGNETOSTATIC,
              "Interface dielectric loss postprocessing is not available for "
              "magnetostatic problems!");
  for (const auto &[idx, data] : postpro.dielectric)
  {
    eps_surfs.try_emplace(idx, data, *h1_fespace.GetParMesh(), bdr_attr_marker);
  }

  // FarField postprocessing.
  MFEM_VERIFY(postpro.farfield.empty() || problem_type == ProblemType::DRIVEN ||
                  problem_type == ProblemType::EIGENMODE,
              "Far-field extraction is only available for driven and eigenmode problems!");

  // Check that we don't have anisotropic materials.
  if (!postpro.farfield.empty())
  {
    const auto &mesh = *nd_fespace.GetParMesh();
    int bdr_attr_max = mesh.bdr_attributes.Size() ? mesh.bdr_attributes.Max() : 0;
    mfem::Array<int> bdr_attr_marker =
        mesh::AttrToMarker(bdr_attr_max, postpro.farfield.attributes);

    std::set<int> domain_attrs;

    for (int i = 0; i < mesh.GetNBE(); i++)
    {
      if (bdr_attr_marker[mesh.GetBdrAttribute(i) - 1])
      {
        int elem_id, _face_id;
        mesh.GetBdrElementAdjacentElement(i, elem_id, _face_id);
        if (elem_id >= 0)
        {
          domain_attrs.insert(mesh.GetAttribute(elem_id));
        }
      }
    }

    for (int attr : domain_attrs)
    {
      MFEM_VERIFY(mat_op.IsIsotropic(attr),
                  "FarField requires isotropic materials, but attribute " +
                      std::to_string(attr) + " is not.");
    }
  }

  farfield = FarFieldData(postpro.farfield, *nd_fespace.GetParMesh(), bdr_attr_marker);
}

SurfacePostOperator::SurfacePostOperator(const IoData &iodata,
                                         const MaterialOperator &mat_op,
                                         mfem::ParFiniteElementSpace &h1_fespace,
                                         mfem::ParFiniteElementSpace &nd_fespace)
  : SurfacePostOperator(iodata.boundaries.postpro, iodata.problem.type, mat_op, h1_fespace,
                        nd_fespace)
{
}

std::complex<double> SurfacePostOperator::GetSurfaceFlux(int idx, const GridFunction *E,
                                                         const GridFunction *B) const
{
  // For complex-valued fields, output the separate real and imaginary parts for the time-
  // harmonic quantity. For power flux (Poynting vector), output only the stationary real
  // part and not the part which has double the frequency.
  auto it = flux_surfs.find(idx);
  MFEM_VERIFY(it != flux_surfs.end(),
              "Unknown surface flux postprocessing index requested!");
  const bool has_imag = (E) ? E->HasImag() : B->HasImag();
  const auto &mesh = *h1_fespace.GetParMesh();
  int bdr_attr_max = mesh.bdr_attributes.Size() ? mesh.bdr_attributes.Max() : 0;
  mfem::Array<int> attr_marker = mesh::AttrToMarker(bdr_attr_max, it->second.attr_list);

  // Use the libCEED surface functional path when supported, avoiding per-call
  // boundary LinearForm assembly in the legacy coefficient path.
  auto &func = flux_funcs[idx];
  if (!func)
  {
    func = std::make_unique<SurfaceFunctional>(
        mat_op.GetMesh(), attr_marker, E ? E->ParFESpace() : nullptr,
        B ? B->ParFESpace() : nullptr, mat_op, it->second.type, it->second.two_sided,
        it->second.center);
  }
  if (func && func->IsValid())
  {
    return func->EvalFlux(E, B);
  }
  if (IsSupportedSurfaceFluxDimension(mesh))
  {
    RequireCeedSurfaceFunctional(func.get(), "3D surface flux");
  }

  auto f =
      it->second.GetCoefficient(E ? &E->Real() : nullptr, B ? &B->Real() : nullptr, mat_op);
  std::complex<double> dot(GetLocalSurfaceIntegral(*f, attr_marker), 0.0);
  if (has_imag)
  {
    f = it->second.GetCoefficient(E ? &E->Imag() : nullptr, B ? &B->Imag() : nullptr,
                                  mat_op);
    double doti = GetLocalSurfaceIntegral(*f, attr_marker);
    if (it->second.type == SurfaceFlux::POWER)
    {
      dot += doti;
    }
    else
    {
      dot.imag(doti);
    }
  }
  Mpi::GlobalSum(1, &dot, (E) ? E->GetComm() : B->GetComm());
  return dot;
}

double SurfacePostOperator::GetInterfaceLossTangent(int idx) const
{
  auto it = eps_surfs.find(idx);
  MFEM_VERIFY(it != eps_surfs.end(),
              "Unknown interface dielectric postprocessing index requested!");
  return it->second.tandelta;
}

double SurfacePostOperator::GetInterfaceElectricFieldEnergy(int idx,
                                                            const GridFunction &E) const
{
  auto it = eps_surfs.find(idx);
  MFEM_VERIFY(it != eps_surfs.end(),
              "Unknown interface dielectric postprocessing index requested!");
  const auto &mesh = *h1_fespace.GetParMesh();
  int bdr_attr_max = mesh.bdr_attributes.Size() ? mesh.bdr_attributes.Max() : 0;
  mfem::Array<int> attr_marker = mesh::AttrToMarker(bdr_attr_max, it->second.attr_list);

  // Use the libCEED surface functional path when supported, avoiding per-call
  // boundary LinearForm assembly in the legacy coefficient path.
  auto &func = eps_funcs[idx];
  if (!func)
  {
    func = std::make_unique<SurfaceFunctional>(mat_op.GetMesh(), attr_marker,
                                               *E.ParFESpace(), mat_op, it->second.type,
                                               it->second.t, it->second.epsilon);
  }
  if (func && func->IsValid())
  {
    return func->Eval(E);
  }
  if (IsSupportedSurfaceFunctionalDimension(mesh))
  {
    RequireCeedSurfaceFunctional(func.get(), "interface dielectric postprocessing");
  }

  auto f = it->second.GetCoefficient(E, mat_op);
  double dot = GetLocalSurfaceIntegral(*f, attr_marker);
  Mpi::GlobalSum(1, &dot, E.GetComm());
  return dot;
}

double
SurfacePostOperator::GetLocalSurfaceIntegral(mfem::Coefficient &f,
                                             const mfem::Array<int> &attr_marker) const
{
  // Integrate the coefficient over the boundary attributes making up this surface index.
  mfem::LinearForm s(&h1_fespace);
  s.AddBoundaryIntegrator(new BoundaryLFIntegrator(f),
                          const_cast<mfem::Array<int> &>(attr_marker));
  s.UseFastAssembly(false);
  s.UseDevice(false);
  s.Assemble();
  s.UseDevice(true);
  return linalg::LocalSum(s);
}

std::vector<std::array<std::complex<double>, 3>> SurfacePostOperator::GetFarFieldrE(
    const std::vector<std::pair<double, double>> &theta_phi_pairs, const GridFunction &E,
    const GridFunction &B, std::complex<double> omega, const PlaneWaveData *incident) const
{
  if (theta_phi_pairs.empty())
    return {};
  MFEM_VERIFY(nd_fespace.GetParMesh()->SpaceDimension() == 3,
              "Far-field computation is only available for 3D simulations!");
  BlockTimer bt0(Timer::POSTPRO_FARFIELD);

  // Compute target unit vectors from the given theta and phis.
  std::vector<std::array<double, 3>> r_naughts;
  r_naughts.reserve(theta_phi_pairs.size());
  for (const auto &[theta, phi] : theta_phi_pairs)
  {
    r_naughts.emplace_back(std::array<double, 3>{
        std::sin(theta) * std::cos(phi), std::sin(theta) * std::sin(phi), std::cos(theta)});
  }

  const auto &mesh = *nd_fespace.GetParMesh();
  int bdr_attr_max = mesh.bdr_attributes.Size() ? mesh.bdr_attributes.Max() : 0;
  mfem::Array<int> attr_marker = mesh::AttrToMarker(bdr_attr_max, farfield.attr_list);

  // Subtract the incident plane wave contribution from the total far field. The
  // Stratton-Chu transformation is linear, so evaluate the analytic incident field
  // integral on the same surface and subtract it to obtain the scattered far field.
  auto SubtractIncident =
      [&](std::vector<std::array<std::complex<double>, 3>> result)
  {
    std::vector<std::array<double, 3>> inc_r(theta_phi_pairs.size());
    std::vector<std::array<double, 3>> inc_i(theta_phi_pairs.size());
    for (int i = 0; i < mesh.GetNBE(); i++)
    {
      if (!attr_marker[mesh.GetBdrAttribute(i) - 1])
      {
        continue;
      }
      auto *T = const_cast<mfem::ParMesh &>(mesh).GetBdrElementTransformation(i);
      const auto *fe = nd_fespace.GetBE(i);
      const auto *ir =
          &mfem::IntRules.Get(fe->GetGeomType(), fem::DefaultIntegrationOrder::Get(*T));
      AddStrattonChuIncidentIntegrandAtElement(*incident, omega, r_naughts, mesh, *T, *ir,
                                               inc_r, inc_i);
    }
    double *data_r_ptr = inc_r.data()->data();
    double *data_i_ptr = inc_i.data()->data();
    std::size_t total_elements = inc_r.size() * 3;
    Mpi::GlobalSum(total_elements, data_i_ptr, E.GetComm());
    Mpi::GlobalSum(total_elements, data_r_ptr, E.GetComm());
    std::array<double, 3> tmp_r, tmp_i;
    for (std::size_t k = 0; k < theta_phi_pairs.size(); k++)
    {
      linalg::Cross3(r_naughts[k], inc_r[k], tmp_r);
      linalg::Cross3(r_naughts[k], inc_i[k], tmp_i);
      for (std::size_t d = 0; d < 3; d++)
      {
        result[k][d] -= std::complex<double>{tmp_r[d], tmp_i[d]};
      }
    }
    return result;
  };

  // Use the libCEED surface functional path when supported, avoiding per-point host
  // coefficient evaluation in the MFEM path. Observation directions are part of the
  // assembled operator and must therefore invalidate the cached functional when changed.
  const bool use_ceed = E.HasImag() && B.HasImag();
  if (use_ceed && (!farfield_func || farfield_func_dirs != r_naughts))
  {
    farfield_func = std::make_unique<SurfaceFunctional>(
        mat_op.GetMesh(), attr_marker, *E.ParFESpace(), *B.ParFESpace(), mat_op, r_naughts);
    farfield_func_dirs = r_naughts;
  }
  if (use_ceed && farfield_func && farfield_func->IsValid())
  {
    auto result = farfield_func->EvalFarField(E, B, omega);
    return incident ? SubtractIncident(std::move(result)) : result;
  }
  if (use_ceed)
  {
    RequireCeedSurfaceFunctional(farfield_func.get(), "3D far-field postprocessing");
  }

  // Integrate. Each MPI process computes its contribution and we will reduce
  // everything at the end. We make them std::vector<std::array<double, 3>>
  // because we want a very simple memory layout so that we can reduce
  // everything with two MPI calls.
  std::vector<std::array<double, 3>> integrals_r(theta_phi_pairs.size());
  std::vector<std::array<double, 3>> integrals_i(theta_phi_pairs.size());

  for (int i = 0; i < mesh.GetNBE(); i++)
  {
    if (!attr_marker[mesh.GetBdrAttribute(i) - 1])
      continue;

    auto *T = const_cast<mfem::ParMesh &>(mesh).GetBdrElementTransformation(i);
    const auto *fe = nd_fespace.GetBE(i);
    const auto *ir =
        &mfem::IntRules.Get(fe->GetGeomType(), fem::DefaultIntegrationOrder::Get(*T));

    AddStrattonChuIntegrandAtElement(E, B, mat_op, omega.real(), omega.imag(), r_naughts,
                                     *T, *ir, integrals_r, integrals_i);
  }

  double *data_r_ptr = integrals_r.data()->data();
  double *data_i_ptr = integrals_i.data()->data();
  std::size_t total_elements = integrals_r.size() * 3;
  Mpi::GlobalSum(total_elements, data_i_ptr, E.GetComm());
  Mpi::GlobalSum(total_elements, data_r_ptr, E.GetComm());

  // Finally, we apply cross product to reduced integrals and package the result
  // in a neatly accessible vector of arrays of complex numbers.
  std::vector<std::array<std::complex<double>, 3>> result(theta_phi_pairs.size());
  StaticVector<3> tmp_r, tmp_i;
  for (std::size_t k = 0; k < theta_phi_pairs.size(); k++)
  {
    linalg::Cross3(r_naughts[k], integrals_r[k], tmp_r);
    linalg::Cross3(r_naughts[k], integrals_i[k], tmp_i);
    for (std::size_t d = 0; d < 3; d++)
    {
      result[k][d] = std::complex<double>{tmp_r[d], tmp_i[d]};
    }
  }
  return incident ? SubtractIncident(std::move(result)) : result;
}

}  // namespace palace
