// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include <vector>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <nlohmann/json.hpp>
#include "utils/configfile.hpp"
#include "utils/iodata.hpp"

using json = nlohmann::json;
using namespace palace;

TEST_CASE("PlaneWaveConfiguration", "[planewave][Serial]")
{
  // Direction and polarization are normalized, attributes sorted, excitation parsed.
  json wave = {{"Index", 1},
               {"Attributes", {2, 1}},
               {"Excitation", true},
               {"Direction", {2.0, 0.0, 0.0}},
               {"Polarization", {0.0, -3.0, 0.0}},
               {"Amplitude", 2.5},
               {"Phase", 0.25},
               {"ReferencePoint", {0.1, 0.2, 0.3}}};
  config::PlaneWaveData data(wave);
  CHECK(data.attributes == std::vector<int>{1, 2});
  CHECK(data.excitation == 1);
  CHECK(data.direction[0] == Catch::Approx(1.0));
  CHECK(data.direction[1] == Catch::Approx(0.0));
  CHECK(data.direction[2] == Catch::Approx(0.0));
  CHECK(data.polarization[0] == Catch::Approx(0.0));
  CHECK(data.polarization[1] == Catch::Approx(-1.0));
  CHECK(data.polarization[2] == Catch::Approx(0.0));
  CHECK(data.amplitude == Catch::Approx(2.5));
  CHECK(data.phase == Catch::Approx(0.25));
  CHECK(data.origin[0] == Catch::Approx(0.1));
  CHECK(data.origin[1] == Catch::Approx(0.2));
  CHECK(data.origin[2] == Catch::Approx(0.3));
  CHECK(!data.empty());

  // Defaults.
  json minimal = {
      {"Index", 2}, {"Attributes", {3}}, {"Direction", {0.0, 0.0, -1.0}}};
  config::PlaneWaveData defaults(minimal);
  CHECK(defaults.excitation == 0);
  CHECK(defaults.amplitude == Catch::Approx(1.0));
  CHECK(defaults.phase == Catch::Approx(0.0));
  CHECK(defaults.polarization[0] == Catch::Approx(1.0));
}

TEST_CASE("PlaneWaveIoData", "[planewave][Serial]")
{
  // The embedded schema must accept the new boundary section and IoData must parse it.
  json config = {
      {"Problem", {{"Type", "Driven"}, {"Output", "planewave_test"}}},
      {"Model", {{"Mesh", "test.msh"}}},
      {"Domains", {{"Materials", {{{"Attributes", {1}}}}}}},
      {"Boundaries",
       {{"Absorbing", {{"Attributes", {1}}, {"Order", 2}}},
        {"PlaneWave",
         {{{"Index", 1},
           {"Attributes", {1}},
           {"Excitation", 1},
           {"Direction", {0.0, 0.0, -1.0}},
           {"Polarization", {1.0, 0.0, 0.0}},
           {"Amplitude", 3.0}}}}}},
      {"Solver", {{"Driven", {{"MinFreq", 1.0}, {"MaxFreq", 2.0}, {"FreqStep", 1.0}}}}}};

  IoData iodata(config, false);
  REQUIRE(iodata.boundaries.planewave.count(1) == 1);
  const auto &pw = iodata.boundaries.planewave.at(1);
  CHECK(pw.excitation == 1);
  CHECK(pw.attributes == std::vector<int>{1});
  CHECK(pw.direction[2] == Catch::Approx(-1.0));
  CHECK(pw.polarization[0] == Catch::Approx(1.0));
}
