#pragma once

// Copyright (c) 2024 Rohit Menon and Maren Bennewitz
// Humanoid Robots Lab, University of Bonn
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// * Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the following disclaimer.
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the following disclaimer in the documentation.
// * Neither the name of the University of Bonn nor the names of its
//   contributors may be used to endorse or promote products derived from this
//   software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// Author: Rohit Menon

#include "kimera_semantics_evidential/semantic_representation_base.h"

namespace kimera {

/**
 * @brief OneHot representation strategy.
 * Assumes the input state is one-hot encoded, with exactly one value as 1 and the rest as 0.
 */
class OneHotRepresentation : public SemanticRepresentationBase {
 public:
  bool validate(const voxblox::SemanticState& measurement) const override {
    int count_ones = (measurement.array() == 1.0f).count();
    int count_non_binary = (measurement.array() != 0.0f && measurement.array() != 1.0f).count();
    
    CHECK(count_ones == 1) << "OneHotRepresentation: State must contain exactly one '1' value.";
    CHECK(count_non_binary == 0) << "OneHotRepresentation: State contains non-binary values.";
    return true;
  }

  voxblox::SemanticState preprocess(const voxblox::SemanticState& measurement) const override {
    return measurement;  // No conversion needed for one-hot encoding
  }

  void postprocess(SemanticVoxel* voxel) const override {
    float sum = voxel->dirichlet_params.sum();
    CHECK(sum > 0) << "OneHotRepresentation: Dirichlet parameters must have a non-zero sum.";
    voxel->semantic_probabilities = voxel->dirichlet_params / sum;
  }

  float calculateMeasurementUncertainty(const voxblox::SemanticState& measurement) const override {
    return 0.0f;  // One-hot encoding has no uncertainty
  }
};

/**
 * @brief Evidential representation strategy.
 * Assumes the input state is represented as Dirichlet alphas (evidence).
 */
class EvidentialRepresentation : public SemanticRepresentationBase {
 public:
  bool validate(const voxblox::SemanticState& measurement) const override {
    CHECK((measurement.array() >= 1.0f).all()) << "EvidentialRepresentation: All alphas must be >= 1.";
    return true;
  }

  voxblox::SemanticState preprocess(const voxblox::SemanticState& measurement) const override {
    return measurement;  // No conversion needed for Dirichlet alphas
  }

  void postprocess(SemanticVoxel* voxel) const override {
    float sum = voxel->dirichlet_params.sum();
    CHECK(sum > 0) << "EvidentialRepresentation: Dirichlet parameters must have a non-zero sum.";
    voxel->semantic_probabilities = voxel->dirichlet_params / sum;
  }

  float calculateMeasurementUncertainty(const voxblox::SemanticState& measurement) const override {
    float alpha_sum = measurement.sum();
    float num_classes = static_cast<float>(measurement.size());
    //LOG(INFO)<<"Measurement Uncertainty: "<<num_classes / alpha_sum;
    return num_classes / alpha_sum;  // Total uncertainty = k / alpha_0
  }
};

/**
 * @brief Probability representation strategy.
 * Assumes the input state is represented as probabilities.
 */
class ProbabilityRepresentation : public SemanticRepresentationBase {
 public:
  bool validate(const voxblox::SemanticState& measurement) const override {
    float sum = measurement.sum();
    CHECK(std::abs(sum - 1.0f) < 1e-5) << "ProbabilityRepresentation: State probabilities must sum to ~1.";
    return true;
  }

  voxblox::SemanticState preprocess(const voxblox::SemanticState& measurement) const override {
    return measurement;  // No conversion needed for probabilities
  }

  void postprocess(SemanticVoxel* voxel) const override {
    float sum = voxel->dirichlet_params.sum();
    CHECK(sum > 0) << "ProbabilityRepresentation: Dirichlet parameters must have a non-zero sum.";
    voxel->semantic_probabilities = voxel->dirichlet_params / sum;
  }

  /**
   * @brief Calculate the measurement uncertainty for a given semantic state.
   *
   * This function computes the uncertainty associated with a given semantic 
   * measurement. The uncertainty is typically derived from the properties of 
   * the measurement and the underlying model used for semantic representation.
   *
   * @param measurement The semantic state measurement for which the uncertainty 
   *        is to be calculated.
   * @return The calculated uncertainty as a float value.
   */
  float calculateMeasurementUncertainty(const voxblox::SemanticState& measurement) const override {
    // Calculate uncertainty as entropy of the probabilities
    Eigen::ArrayXf non_zero_probs = measurement.array().select(measurement.array(), 1.0f).log();
    float entropy = -(measurement.array() * non_zero_probs).sum();
    return entropy/kMaxEntropy;
  }

  private:
  static constexpr float kMaxEntropy = std::log(static_cast<float>(kTotalNumberOfLabels));
};

}  // namespace kimera

