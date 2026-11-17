// NOTE: Most code is derived from voxblox: github.com/ethz-asl/voxblox
// Copyright (c) 2016, ETHZ ASL
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// * Redistributions of source code must retain the above copyright notice, this
//   list of conditions and the following disclaimer.
//
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the following disclaimer in the documentation
//   and/or other materials provided with the distribution.
//
// * Neither the name of voxblox nor the names of its
//   contributors may be used to endorse or promote products derived from
//   this software without specific prior written permission.
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

/**
 * @file   semantic_tsdf_integrator.h
 * @brief  Integrator of semantic and geometric information
 * @author Antoni Rosinol
 */

#pragma once

#include <memory>
#include <utility>

#include <Eigen/Core>

#include <voxblox/integrator/tsdf_integrator.h>
#include <voxblox/utils/timing.h>

#include "kimera_semantics_evidential/color.h"
#include "kimera_semantics_evidential/common.h"
#include "kimera_semantics_evidential/semantic_voxel.h"
#include "kimera_semantics_evidential/semantic_representation.h"
#include "kimera_semantics_evidential/voxel_level_fusion.h"
#include "kimera_semantics_evidential/uncertainty_estimator.h"
#include "kimera_semantics_evidential/viewpoint_metrics.h"
#include <atomic>

namespace kimera {

// enum class ColorMode : int {
//   kColor = 0,
//   kSemantic = 1,
//   kSemanticProbability = 2,
// };

class SemanticIntegratorBase {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  typedef std::shared_ptr<SemanticIntegratorBase> Ptr;
  typedef voxblox::
      ApproxHashArray<12, std::mutex, voxblox::GlobalIndex, voxblox::LongIndexHash>
          Mutexes;

  struct SemanticConfig {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  // Likelihood probability of observing a measurement given the prior label.
  SemanticProbability semantic_measurement_probability_ = 0.9f;

  // How to color the semantic mesh.
  ColorMode color_mode = ColorMode::kSemantic;

  std::shared_ptr<SemanticLabel2Color> semantic_label_to_color_ = nullptr;

  SemanticRepresentationType measurement_representation_strategy = SemanticRepresentationType::kEvidential;
  SemanticRepresentationType voxel_representation_strategy = SemanticRepresentationType::kEvidential;
  VoxelLevelFusionType voxel_fusion_strategy = VoxelLevelFusionType::kSimple;
  // Uncertainty strategy for voxel-level updates (e.g., Entropy, Variance).
  UncertaintyEstimatorType uncertainty_strategy = UncertaintyEstimatorType::kTotalUncertainty;

  // Additional placeholders for strategy-specific parameters.
  voxblox::FloatingPoint uncertainty_threshold = 0.3f;
  voxblox::FloatingPoint label_assignment_threshold = 0.8f;

  bool use_viewpoint_similarity = false;
  bool use_viewpoint_obliqueness = false;
  
  bool replace_on_first_measurement = false;
  int16_t width = 640;
  int16_t height = 480;
  float bbox_padding_in_voxels = 0.05f;
  // Whether to use the kernel in measurement mode.
  bool kernel_in_measurement_mode = false;
  // Whether to correct the position of training points
  bool enable_position_correction = false;
  KernelConfig kernel_params;
  // Whther to use allocate new voxels
  bool allocate_new_voxels = false;
  //Whether to add freespace points to training data
  bool add_freespace_pts_to_trng_data = false;
};


  SemanticIntegratorBase(const SemanticConfig& semantic_config,
                         voxblox::Layer<SemanticVoxel>* semantic_layer);


  
  // SHOULD BE THREAD SAFE. Updates semantic_voxel probabilities given
  // semantic_label measurement and confidence.

  /**
   * @brief updateSemanticVoxel
   * Probabilistically updates semantic voxels according to a measurement model.
   * TODO(Toni): parametrize the measurement model.
   *
   * @param global_voxel_idx
   * @param measurement_frequencies
   * @param mutexes: we pass the list of mutexes bcs it would be too wasteful to
   * have to lock two mutexes (one for the tsdf_voxel, another for the
   * semantic_voxel, instead we expect the user to send the mutexes for
   * tsdf_voxels, and we will always lock them when updating semantic_voxels.
   * @param tsdf_voxel
   * @param semantic_voxel
   */
  void updateSemanticVoxel(const vxb::GlobalIndex& global_voxel_idx,
    const SemanticProbabilities& measurement_frequencies,
    Mutexes* mutexes,
    vxb::TsdfVoxel* tsdf_voxel,
    SemanticVoxel* semantic_voxel);

  void updateSemanticVoxel(const voxblox::GlobalIndex& global_voxel_idx,
                           const voxblox::SemanticState& semantic_measurements,
                           const voxblox::FloatingPoint& measurement_weight,
                           const vxb::DepthUncertainty& depth_uncertainty,
                           Mutexes* mutexes,
                           voxblox::TsdfVoxel* tsdf_voxel,
                           const voxblox::TsdfVoxel& voxel_measured_state,
                           SemanticVoxel* semantic_voxel
                          );
  void updateSemanticVoxelColor(const SemanticLabel& semantic_label,
                          HashableColor* semantic_voxel_color) const;
  void updateSemanticVoxelColor(
                          const SemanticLabel& semantic_label,
                          const SemanticState& semantic_state,
                          const ColorMode& color_mode,
                          HashableColor* semantic_voxel_color, Uncertainty uncertainty = 0.0f) const;
  
  SemanticProbability computeMeasurementProbability(
    vxb::FloatingPoint ray_distance);

  void updateSemanticVoxelProbabilities(
    const SemanticProbabilities& measurement_frequencies,
    SemanticProbabilities* semantic_prior_probability) const;

  // THREAD SAFE
  void normalizeProbabilities(SemanticProbabilities* unnormalized_probs) const;

  // THREAD SAFE
  inline void calculateMaximumLikelihoodLabel(
    const SemanticProbabilities& semantic_posterior,
    SemanticLabel* semantic_label) const;                        
  // Will return a pointer to a voxel located at global_voxel_idx in the label
  // layer. Thread safe.
  // Takes in the last_block_idx and last_block to prevent unneeded map
  // lookups. If the block this voxel would be in has not been allocated, a
  // block in temp_label_block_map_ is created/accessed and a voxel from this
  // map is returned instead. Unlike the layer, accessing
  // temp_label_block_map_ is controlled via a mutex allowing it to grow
  // during integration. These temporary blocks can be merged into the layer
  // later by calling updateLayerWithStoredBlocks()
  SemanticVoxel* allocateStorageAndGetSemanticVoxelPtr(
      const voxblox::GlobalIndex& global_voxel_idx,
      voxblox::Block<SemanticVoxel>::Ptr* last_semantic_block,
      voxblox::BlockIndex* last_block_idx);

  // NOT THREAD SAFE
  void updateSemanticLayerWithStoredBlocks();

  inline bool isSemanticLabelValid(const SemanticLabel& semantic_label)
  {
    return true;
  }

  std::unique_ptr<SemanticRepresentationBase> createMeasurementRepresentation(
    SemanticRepresentationType measurement_representation_type) const;
  std::unique_ptr<VoxelLevelFusionBase> createVoxelLevelFuser(
    VoxelLevelFusionType uncertainty_estimator_type) const;
  std::unique_ptr<UncertaintyEstimatorBase> createUncertaintyEstimator(
    UncertaintyEstimatorType uncertainty_estimator_type) const;
  std::unique_ptr<SemanticRepresentationBase> createVoxelRepresentation(
    SemanticRepresentationType voxel_representation_type_) const;

 protected:
  /// Thread safe.
  // inline bool isSemanticLabelValid(const SemanticLabel& semantic_label) const {
  //   // We discard any point in the dynamic semantic labels.
  //   return std::find(semantic_config_.dynamic_labels_.begin(),
  //                    semantic_config_.dynamic_labels_.end(),
  //                    semantic_label) == semantic_config_.dynamic_labels_.end();
  // }

  std::unique_ptr<VoxelLevelFusionBase> voxel_fuser_;
  std::unique_ptr<UncertaintyEstimatorBase> uncertainty_estimator_;
  std::unique_ptr<SemanticRepresentationBase> measurement_representation_;
  std::unique_ptr<SemanticRepresentationBase> voxel_representation_;
  std::unique_ptr<ViewpointSimilarityEstimator> viewpoint_similarity_estimator_;
  std::unique_ptr<ViewpointObliquenessEstimator> viewpoint_obliqueness_estimator_;


  float computeOccupancyBelief(const vxb::TsdfVoxel& voxel_measured_state) const;
  // Handles free space updates when sdf > semantic_voxel_size_
  void handleFreeSpace(SemanticState& processed_measurement, float occupancy_belief, float weight);
  // Handles unknown space updates when weight < kEpsilon
  void handleUnknownSpace(SemanticState& processed_measurement, float weight);
  // Handles occupied space updates when sdf < 0 and weight is valid
  void handleOccupiedSpace(SemanticState& processed_measurement);
  // Applies the fusion strategy to integrate the measurement into the voxel
  void applyFusionStrategy(SemanticVoxel* semantic_voxel, 
                           const SemanticState& processed_measurement, 
                           float measurement_weight);
  // Finalizes voxel updates, including uncertainty, label assignment, and color
  void finalizeSemanticVoxelUpdate(SemanticVoxel* semantic_voxel, voxblox::TsdfVoxel* tsdf_voxel);
  
  std::atomic<int> frame_idx;

 private:
  /**
   * @brief setSemanticLayer
   * Helper function to store members of the class related to the semantic layer
   * as well as the semantic layer itself from the provided pointer.
   * @param semantic_layer pointer to an already created semantic_layer
   */
  void setSemanticLayer(voxblox::Layer<SemanticVoxel>* semantic_layer);

  void setSemanticProbabilities();

 public:
  /// Configuration
  const SemanticConfig semantic_config_;

  /// Layer with semantic information.
  voxblox::Layer<SemanticVoxel>* semantic_layer_;

  /**
   * Temporary block storage, used to hold blocks that need to be created while
   * integrating a new pointcloud
   */
  mutable std::mutex temp_semantic_block_mutex_;
  voxblox::Layer<SemanticVoxel>::BlockHashMap temp_semantic_block_map_;

  // Log probabilities of matching measurement and prior label,
  // and non-matching.
  SemanticProbability log_match_probability_;
  SemanticProbability log_non_match_probability_;
  // A `#Labels X #Labels` Eigen matrix where each `j` column represents the
  // probability of observing label `j` when current label is `i`, where `i`
  // is the row index of the matrix.
  SemanticLikelihoodFunction semantic_log_likelihood_;

  /// We keep these in case someone wants to experiment with different
  /// semantic/tsdf voxel sizes.
  // Cached map config.
  voxblox::FloatingPoint semantic_voxel_size_;
  size_t semantic_voxels_per_side_;
  voxblox::FloatingPoint semantic_block_size_;

  // Derived types.
  voxblox::FloatingPoint semantic_voxel_size_inv_;
  voxblox::FloatingPoint semantic_voxels_per_side_inv_;
  voxblox::FloatingPoint semantic_block_size_inv_;

  inline void incrementFrameIdx() { frame_idx++; }
  inline int getFrameIdx() { int idx = frame_idx.load(std::memory_order_relaxed); return  idx;}
};

}  // Namespace kimera
