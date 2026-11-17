/**
 * @file   semantic_voxel.h
 * @brief  Definition of what a semantic voxel is
 * @author Antoni Rosinol
 */

#pragma once

#include "kimera_semantics_evidential/color.h"
#include "kimera_semantics_evidential/common.h"

using namespace voxblox;
namespace kimera {

struct SemanticVoxel {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  // Initialize voxel to unknown label.
  SemanticLabel semantic_label = kUnknownSemanticLabelId;
  // Initialize voxel to uniform probability.
  // Use log odds! So uniform ditribution of 1/kTotalNumberOfLabels,
  // should be std::log(1/kTotalNumberOfLabels)
  voxblox::SemanticState dirichlet_params =
      // SemanticProbabilities::Constant(std::log(1 / kTotalNumberOfLabels));
      voxblox::SemanticState::Constant(1);

  voxblox::SemanticState semantic_probabilities =
      // SemanticProbabilities::Constant(std::log(1 / kTotalNumberOfLabels));
      voxblox::SemanticState::Constant(-1.36172783602);

  SemanticProbabilities semantic_priors = 
  SemanticProbabilities::Constant(-1.36172783602); //SemanticProbabilities::Constant(std::log(1 / kTotalNumberOfLabels));

  Weight num_measurements = 0;

  Uncertainty uncertainty = 0.0f; 

  VoxelUpdateState update_state = VoxelUpdateState::UnInitialized;

  // Initialize voxel with gray color
  // Make sure that all color maps agree on semantic label 0u -> gray
  HashableColor color = HashableColor::Gray();
};

}  // namespace kimera
