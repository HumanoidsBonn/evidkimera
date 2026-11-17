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

#include "kimera_semantics_evidential/semantic_voxel.h"
#include <cmath>

namespace kimera {

/**
 * @brief Base class for voxel uncertainty strategies.
 */
class UncertaintyEstimatorBase {
 public:
  virtual ~UncertaintyEstimatorBase() = default;

  /**
   * @brief Compute the voxel-level uncertainty based on Dirichlet parameters.
   * @param voxel The semantic voxel to evaluate.
   * @return The computed uncertainty value.
   */
  virtual float computeUncertainty(const SemanticState& semantic_state) const = 0;
  
  void incrementUncertaintyWeight( SemanticVoxel* voxel) const {
    //voxel->uncertainty_weight += 1;
  }

protected:
  
};

/**
 * @brief Strategy for computing total uncertainty.
 * Uses the formula: U = k / α₀, where k is the number of classes
 * and α₀ is the sum of Dirichlet parameters (total evidence).
 */
class TotalUncertaintyEstimator : public UncertaintyEstimatorBase {
 public:
  float computeUncertainty(const SemanticState& semantic_state) const override {
    const float alpha_sum = semantic_state.sum();
    const int num_classes = static_cast<int>(semantic_state.size());
    CHECK(alpha_sum > 0.0f) << "TotalUncertainty: Alpha sum must be greater than zero.";
    return static_cast<float>(num_classes) / alpha_sum;  // U = k / α₀
  }
};

/**
 * @brief Strategy for computing variance-based uncertainty.
 * Normalizes variance using the precomputed maximum variance.
 * Uses the formula: Var[αₜc] = αₜc(Sₜ - αₜc) / (Sₜ²(Sₜ + 1)).
 */
class VarianceUncertaintyEstimator : public UncertaintyEstimatorBase {
 public:
  VarianceUncertaintyEstimator() = default;

  float computeUncertainty(const SemanticState& semantic_state) const override {
    const Eigen::VectorXf& alpha = semantic_state;
    const float alpha_sum = alpha.sum();                   // Sₜ
    const float alpha_sum_squared = alpha_sum * alpha_sum; // Sₜ²
    const int num_classes = static_cast<int>(alpha.size()); // k

    CHECK(alpha_sum > 0.0f) << "VarianceUncertainty: Alpha sum must be greater than zero.";

    float total_variance = 0.0f;
    
    for (int c = 0; c < alpha.size(); ++c) {
      float alpha_c = alpha[c];
      float variance_c = (alpha_c * (alpha_sum - alpha_c)) /
                         (alpha_sum_squared * (alpha_sum + 1.0f));
      total_variance += variance_c;
    }

    // Mean variance across all classes
    float mean_variance = total_variance / static_cast<float>(num_classes);
    // Normalize variance using precomputed maximum variance
    return mean_variance / kMaxVariance;
  }

 private:
  // Precompute the maximum variance for the known number of labels
  static constexpr float kMaxVariance =
      static_cast<float>(kTotalNumberOfLabels - 1) /
      (kTotalNumberOfLabels * kTotalNumberOfLabels * (kTotalNumberOfLabels + 1));
};

/**
 * @brief Strategy for computing Dempster-Shafer-based uncertainty.
 * Uses the formula: Ignorance = 1 - α₀ (total evidence).
 */
class DempsterShaferUncertaintyEstimator : public UncertaintyEstimatorBase {
 public:
  float computeUncertainty(const SemanticState& semantic_state) const override {
    const float alpha_sum = semantic_state.sum();  // Total evidence (α₀)
    CHECK(alpha_sum >= 0.0f) << "DempsterShaferUncertainty: Alpha sum must be non-negative.";
    return 1.0f - alpha_sum;  // Ignorance = 1 - total evidence
  }
};

/**
 * @brief Strategy for computing probability-based variance uncertainty.
 * Uses the normalized variance of probabilities for each class.
 */
class ProbabilityVarianceUncertaintyEstimator : public UncertaintyEstimatorBase {
 public:
  float computeUncertainty(const SemanticState& semantic_state) const override {
    const Eigen::VectorXf& alpha = semantic_state;
    const float alpha_sum = alpha.sum();  // Total evidence (α₀)
    CHECK(alpha_sum > 0.0f) << "ProbabilityVarianceUncertainty: Alpha sum must be greater than zero.";

    Eigen::VectorXf probs = alpha / alpha_sum;  // Normalize to probabilities

    float total_variance = 0.0f;
    for (int c = 0; c < probs.size(); ++c) {
      float p_c = probs[c];
      total_variance += p_c * (1.0f - p_c);  // Variance for each class
    }
    return total_variance / static_cast<float>(probs.size());  // Mean variance across classes
  }
};
/**
 * @brief Strategy for computing probability-based entropy uncertainty.
 * Uses the formula: Entropy = -Σ pₜc log(pₜc), where pₜc is the probability of class c.
 */
class ProbabilityEntropyUncertaintyEstimator : public UncertaintyEstimatorBase {
 public:
  float computeUncertainty(const SemanticState& semantic_state) const override {
    const Eigen::VectorXf& alpha = semantic_state;
    const float alpha_sum = alpha.sum();  // Total evidence (α₀)
    CHECK(alpha_sum > 0.0f) << "ProbabilityEntropyUncertainty: Alpha sum must be greater than zero.";

    Eigen::VectorXf probs = alpha / alpha_sum;  // Normalize to probabilities

    float entropy = 0.0f;
    for (int c = 0; c < probs.size(); ++c) {
      float p_c = probs[c];
      if (p_c > 0.0f) {
        entropy -= p_c * std::log(p_c);  // Entropy for each class
      }
    }
    return entropy;  // Total entropy
  }
};

}  // namespace kimera
