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
//   this list of conditions and the following disclaimer in the documentation
//   and/or other materials provided with the distribution.
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

#pragma once

#include <cmath>
#include <Eigen/Core>
#include <glog/logging.h>
#include "kimera_semantics_evidential/semantic_voxel.h"

namespace kimera{
class SemanticRepresentationBase {
 public:
  virtual ~SemanticRepresentationBase() = default;

  /**
   * @brief Validate the input semantic state.
   * @param state The semantic state to validate.
   * @return True if the state is valid, false otherwise.
   */
  virtual bool validate(const voxblox::SemanticState& measurement) const = 0;

  /**
   * @brief Preprocess the semantic state into a format suitable for fusion.
   * @param state The semantic state to preprocess.
   * @return The processed semantic state.
   */
  virtual voxblox::SemanticState preprocess(const voxblox::SemanticState& measurement) const = 0;

  /**
   * @brief Postprocess the semantic state into probabilities.
   * @param voxel The semantic voxel to postprocess.
   */
  virtual void postprocess(SemanticVoxel* voxel) const = 0;

  /**
   * @brief Calculate the uncertainty of the semantic state.
   * @param state The semantic state to calculate uncertainty for.
   * @return The calculated uncertainty.
   */
  virtual float calculateMeasurementUncertainty(const voxblox::SemanticState& measurement) const = 0;

  voxblox::SemanticState calcMeasurementEvidence(const voxblox::SemanticState& dirichlet_params) const {
    voxblox::SemanticState evidence = dirichlet_params -SemanticState::Ones();
    return evidence;
  }

  voxblox::SemanticState calcProbabilities(const voxblox::SemanticState& input) const {
    voxblox::SemanticState probabilities = input;
    if (input.sum() < 0.98) {
      throw std::runtime_error("Probabilities sum is too low.");
    }
    probabilities /= input.sum();
    return probabilities;
  }
  voxblox::SemanticState probabilitiesToLogOdds(const voxblox::SemanticState& probabilities) const {
    voxblox::SemanticState log_odds = probabilities.array().log();
    return log_odds;
  }

  voxblox::SemanticState logOddsToProbabilities(const voxblox::SemanticState& log_odds) const {
    voxblox::SemanticState probabilities = log_odds.array().exp();
    return calcProbabilities(probabilities);
  }

  kimera::SemanticLabel getSemanticLabel(const voxblox::SemanticState& semantic_state, Uncertainty uncertainty, float unc_threshold= 0.1f) const {
    kimera::SemanticLabel label = kUnknownSemanticLabelId;
    if (uncertainty < unc_threshold) {
      semantic_state.maxCoeff(&label);
    }
    return label;
  }
};
}