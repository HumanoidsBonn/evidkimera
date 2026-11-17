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

/**
 * @file   semantic_tsdf_integrator.cpp
 * @brief  Integrator of semantic and geometric information
 * @author Antoni Rosinol
 */

#include "kimera_semantics_evidential/semantic_integrator_base.h"

#include <functional>  // For placeholders for Semantics
#include <list>
#include <memory>
#include <numeric>  // For accumulate for Semantics
#include <string>
#include <utility>

#include <Eigen/Core>

#include <voxblox/integrator/tsdf_integrator.h>
#include <voxblox/utils/timing.h>

#include "kimera_semantics_evidential/color.h"
#include "kimera_semantics_evidential/common.h"
#include "kimera_semantics_evidential/semantic_voxel.h"
#include "kimera_semantics_evidential/semantic_integrator_base.h"

namespace kimera {

SemanticIntegratorBase::SemanticIntegratorBase(
    const SemanticConfig& semantic_config,
    vxb::Layer<SemanticVoxel>* semantic_layer)
    : semantic_config_(semantic_config),
      semantic_layer_(nullptr),
      temp_semantic_block_mutex_(),
      temp_semantic_block_map_(),
      log_match_probability_(),
      log_non_match_probability_(),
      semantic_log_likelihood_(),
      semantic_voxel_size_(),
      semantic_voxels_per_side_(),
      semantic_block_size_(),
      semantic_voxel_size_inv_(),
      semantic_voxels_per_side_inv_(),
      semantic_block_size_inv_() {
  setSemanticLayer(semantic_layer);
  CHECK_NOTNULL(semantic_layer_);
  
  //setSemanticProbabilities();

  voxel_fuser_ = createVoxelLevelFuser(semantic_config.voxel_fusion_strategy);
  measurement_representation_ = createMeasurementRepresentation(semantic_config.measurement_representation_strategy);
  uncertainty_estimator_ = createUncertaintyEstimator(semantic_config.uncertainty_strategy);
  voxel_representation_ = createVoxelRepresentation(semantic_config.voxel_representation_strategy);

  viewpoint_similarity_estimator_ = std::make_unique<ViewpointSimilarityEstimator>();
  viewpoint_obliqueness_estimator_ = std::make_unique<ViewpointObliquenessEstimator>();
  frame_idx.store(0);
  // Initialize strategies based on SemanticConfig.
}

std::unique_ptr<SemanticRepresentationBase> SemanticIntegratorBase::createMeasurementRepresentation(
    SemanticRepresentationType representation_type) const {
  LOG(INFO) << "Creating measurement representation of type: " << static_cast<int>(representation_type);
  switch (representation_type) {
    case SemanticRepresentationType::kOneHot:
      return std::make_unique<OneHotRepresentation>();
    case SemanticRepresentationType::kProbability:
      return std::make_unique<ProbabilityRepresentation>();
    case SemanticRepresentationType::kEvidential:
      return std::make_unique<EvidentialRepresentation>();
    default:
      LOG(FATAL) << "Unknown representation strategy type!";
      return nullptr;
  }
}

std::unique_ptr<SemanticRepresentationBase> SemanticIntegratorBase::createVoxelRepresentation(
    SemanticRepresentationType representation_type) const {
    LOG(INFO) << "Creating voxel representation of type: " << static_cast<int>(representation_type);
  switch (representation_type) {
    case SemanticRepresentationType::kProbability:
      return std::make_unique<ProbabilityRepresentation>();
    case SemanticRepresentationType::kEvidential:
      return std::make_unique<EvidentialRepresentation>();
    default:
      LOG(FATAL) << "Unknown representation strategy type!";
      return nullptr;
  }
}

std::unique_ptr<VoxelLevelFusionBase> SemanticIntegratorBase::createVoxelLevelFuser(
    VoxelLevelFusionType strategy_type) const {
  LOG(INFO) << "Creating voxel fuser of type: " << static_cast<int>(strategy_type);
  switch (strategy_type) {
    case VoxelLevelFusionType::kNone:
      return std::make_unique<NoneFusion>();
    case VoxelLevelFusionType::kSimple:
      return std::make_unique<SimpleFusion>();
    case VoxelLevelFusionType::kMeasurementUncertainty:
      return std::make_unique<MeasurementUncertaintyFusion>();
    case VoxelLevelFusionType::kTsdfWeightedUncertainty:
      return std::make_unique<TsdfWeightedUncertaintyFusion>();
    case VoxelLevelFusionType::kVoxelMeasurementUncertainty:
      return std::make_unique<VoxelMeasurementUncertaintyFusion>();
    default:
      LOG(FATAL) << "Unknown voxel fusion strategy type!";
      return nullptr;
  }
}

std::unique_ptr<UncertaintyEstimatorBase> SemanticIntegratorBase::createUncertaintyEstimator(
    UncertaintyEstimatorType uncertainty_estimator_type) const {
  LOG(INFO) << "Creating uncertainty estimator of type: " << static_cast<int>(uncertainty_estimator_type);
  switch (uncertainty_estimator_type) {
    case UncertaintyEstimatorType::kNone:
      return nullptr;  // No uncertainty calculation
    case UncertaintyEstimatorType::kTotalUncertainty:
      return std::make_unique<TotalUncertaintyEstimator>();
    case UncertaintyEstimatorType::kVariance:
      return std::make_unique<VarianceUncertaintyEstimator>();
    case UncertaintyEstimatorType::kDempsterShafer:
      return std::make_unique<DempsterShaferUncertaintyEstimator>();
    case UncertaintyEstimatorType::kProbabilityVariance:
      return std::make_unique<ProbabilityVarianceUncertaintyEstimator>();
    case UncertaintyEstimatorType::kProbabilityEntropy:
      return std::make_unique<ProbabilityEntropyUncertaintyEstimator>();
    default:
      LOG(FATAL) << "Unknown voxel uncertainty strategy type!";
      return nullptr;
  }
}


void SemanticIntegratorBase::setSemanticLayer(
    vxb::Layer<SemanticVoxel>* semantic_layer) {
  CHECK_NOTNULL(semantic_layer);

  semantic_layer_ = semantic_layer;

  semantic_voxel_size_ = semantic_layer_->voxel_size();
  semantic_block_size_ = semantic_layer_->block_size();
  semantic_voxels_per_side_ = semantic_layer_->voxels_per_side();

  semantic_voxel_size_inv_ = 1.0 / semantic_voxel_size_;
  semantic_block_size_inv_ = 1.0 / semantic_block_size_;
  semantic_voxels_per_side_inv_ = 1.0 / semantic_voxels_per_side_;
}


void SemanticIntegratorBase::updateSemanticVoxel(
  const vxb::GlobalIndex& global_voxel_idx,
  const SemanticState& semantic_measurements,
  const  vxb::FloatingPoint& measurement_weight,
  const vxb::DepthUncertainty& depth_uncertainty,
  Mutexes* mutexes,
  voxblox::TsdfVoxel* tsdf_voxel,
  const voxblox::TsdfVoxel& voxel_measured_state,
  SemanticVoxel* semantic_voxel) {
  DCHECK(mutexes != nullptr);
  DCHECK(tsdf_voxel != nullptr);
  DCHECK(semantic_voxel != nullptr);

  // Ensure all required strategies are set
  CHECK(voxel_fuser_) << "Voxel Fuser is not set.";
  CHECK(uncertainty_estimator_) << "Uncertainty estimator is not set.";
  CHECK(measurement_representation_) << "Representation strategy is not set.";
  CHECK(voxel_representation_) << "Voxel representation strategy is not set.";

  std::lock_guard<std::mutex> lock(mutexes->get(global_voxel_idx));

  // Validate the measurement using the representation strategy
  if (!measurement_representation_->validate(semantic_measurements)) {
    LOG(WARNING) << "Invalid semantic state. Skipping voxel update.";
    return;
  }

  // Compute occupancy belief using the final TSDF weight
  // Compute occupancy belief and use it as λ (weighting factor)
  float occupancy_belief = computeOccupancyBelief(voxel_measured_state);
  //LOG(INFO)<<"Occupancy Belief: "<<occupancy_belief;
  float lambda = std::max(occupancy_belief, 0.01f);  // Prevents instability with too-small updates
  float sdf = voxel_measured_state.distance;
  float depth_measurement_weight =voxel_measured_state.weight; // Already adjusted by weight dropoff

  SemanticState processed_measurement = measurement_representation_->preprocess(semantic_measurements);
  

  if (sdf > 2*semantic_voxel_size_) {
     handleFreeSpace(processed_measurement, lambda, depth_measurement_weight);
  } 
  else if (depth_measurement_weight < kEpsilon) {
     handleUnknownSpace(processed_measurement, depth_measurement_weight);
  } 
  else
  {
    handleOccupiedSpace(processed_measurement);
  }
  depth_measurement_weight = measurement_weight;
  applyFusionStrategy(semantic_voxel, processed_measurement, depth_measurement_weight);
  finalizeSemanticVoxelUpdate(semantic_voxel, tsdf_voxel);
}

void SemanticIntegratorBase::setSemanticProbabilities() {
  SemanticProbability match_probability =
      semantic_config_.semantic_measurement_probability_;
  SemanticProbability non_match_probability =
      1.0f - semantic_config_.semantic_measurement_probability_;
  CHECK_GT(match_probability, 0.0);
  CHECK_GT(non_match_probability, 0.0);
  CHECK_LT(match_probability, 1.0);
  CHECK_LT(non_match_probability, 1.0);
  log_match_probability_ = std::log(match_probability);
  log_non_match_probability_ = std::log(non_match_probability);
  CHECK_GT(log_match_probability_, log_non_match_probability_)
      << "Your probabilities do not make sense... The likelihood of a "
         "label, knowing that we have measured that label, should not be"
         "smaller than the likelihood of seeing another label!";
  semantic_log_likelihood_ =
      semantic_log_likelihood_.Constant(log_non_match_probability_);
  semantic_log_likelihood_.diagonal() =
      semantic_log_likelihood_.diagonal().Constant(log_match_probability_);
  // TODO(Toni): sanity checks, set as DCHECK_EQ.
  CHECK_NEAR(semantic_log_likelihood_.diagonal().sum(),
  kTotalNumberOf2DLabels * log_match_probability_,
             100 * vxb::kFloatEpsilon);
  CHECK_NEAR(
      semantic_log_likelihood_.sum(),
      kTotalNumberOf2DLabels * log_match_probability_ +
          std::pow(kTotalNumberOf2DLabels, 2) * log_non_match_probability_ -
          kTotalNumberOf2DLabels * log_non_match_probability_,
      1000 * vxb::kFloatEpsilon);
  // Set the likelihood row for unknown label to 0, so to avoid integrating
  // our knowledge that the voxel is unknown.
  // Log(0) is -inf.
  // Well, now, rather, we give uniform weight, saying that we don't know what
  // is what.
  semantic_log_likelihood_.col(kUnknownSemanticLabelId).setZero();
}
SemanticProbability SemanticIntegratorBase::computeMeasurementProbability(
  vxb::FloatingPoint ray_distance) {
return 1.0;
}

void SemanticIntegratorBase::normalizeProbabilities(
  SemanticProbabilities* unnormalized_probs) const {
CHECK_NOTNULL(unnormalized_probs);
CHECK_GT(unnormalized_probs->size(), 0u);
CHECK_LT((*unnormalized_probs)[0], 0.0) << "Are you sure you are using"
                                           "log odds?";

// Find norm
// Make it a double so you make sure there is no overflow??
SemanticProbability normalization_factor = unnormalized_probs->norm();
// Eigen's normalize :
// "If the input evidentialvector is too small (i.e., this->norm()==0),
// then this function returns a copy of the input."
CHECK_GE(normalization_factor, 0.0);
if (normalization_factor != 0.0) {
  unnormalized_probs->normalize();
} else {
  CHECK_EQ(unnormalized_probs->size(), kTotalNumberOf2DLabels);
  static const SemanticProbability kUniformLogProbability =
      std::log(1 / kTotalNumberOf2DLabels);
  LOG(WARNING) << "Normalization Factor is " << normalization_factor
               << ", all values are 0. Normalizing to log(1/n) = "
               << kUniformLogProbability;
  unnormalized_probs->setConstant(kUniformLogProbability);
}

// TODO(Toni): REMOVE
// Check that normalization really happened.
static constexpr bool kDebug = true;
if (kDebug) {
  CHECK_NEAR(unnormalized_probs->norm(), 1.0f, vxb::kFloatEpsilon);
}
}


void SemanticIntegratorBase::updateSemanticVoxel(
  const vxb::GlobalIndex& global_voxel_idx,
  const SemanticProbabilities& measurement_frequencies,
  Mutexes* mutexes,
  vxb::TsdfVoxel* tsdf_voxel,
  SemanticVoxel* semantic_voxel) {
DCHECK(mutexes != nullptr);
DCHECK(tsdf_voxel != nullptr);
DCHECK(semantic_voxel != nullptr);
//LOG(INFO)<<"measurement frequencies: "<<measurement_frequencies.transpose();
// TODO(Toni): ideally, only lock once in updateTsdfVoxel, but we need to
// modify Voxblox for that.
// Lookup the mutevidentialex that is responsible for this voxel and lock it
std::lock_guard<std::mutex> lock(mutexes->get(global_voxel_idx));

// TODO(Toni): ideally, return new_sdf from updateTsdfVoxel, but we need to
// modify Voxblox for that.
// Similar to color blending in Voxblox tsdf_integrator.cc,
// Quoting Voxblox:
// color blending is expensive only do it close to the surface
//  if (std::abs(sdf) < config_.default_truncation_distance) {
// Do the same for semantic updates:
// Don't do it bcevidentials of copyrights... :'(
////// Here is the actual logic of Kimera-Semantics:   /////////////////////
// Calculate new probabilities given the measurement frequencies.
updateSemanticVoxelProbabilities(measurement_frequencies,
                                 &semantic_voxel->semantic_priors);

// Get MLE semantic label.
calculateMaximumLikelihoodLabel(semantic_voxel->semantic_priors,
                                &semantic_voxel->semantic_label);
//LOG(INFO)<<"Semantic Label: "<<static_cast<int>(semantic_voxel->semantic_label);
// Colorize according to current MLE semantic label.
updateSemanticVoxelColor(semantic_voxel->semantic_label,
                         &semantic_voxel->color);

// Actually change the color of the TSDF voxel so we do not need to change a
// single line for the meshing with respect to Voxblox.
tsdf_voxel->label = semantic_voxel->semantic_label;

switch (semantic_config_.color_mode) {
  case ColorMode::kColor:
    // Nothing, base class colors the tsdf voxel for us.
    break;
  case ColorMode::kSemantic:
    tsdf_voxel->color = semantic_voxel->color;
    break;
  case ColorMode::kSemanticProbability:
    // TODO(Toni): Might be a bit expensive to calc all these exponentials...
    tsdf_voxel->color = vxb::rainbowColorMap(std::exp(
        semantic_voxel->semantic_priors[semantic_voxel->semantic_label]));
    break;
  default:
    LOG(FATAL) << "Unknown semantic color mode: "
               << static_cast<std::underlying_type<ColorMode>::type>(
                      semantic_config_.color_mode);
    break;
}
////////////////////////////////////////////////////////////////////////////
//}
}

void SemanticIntegratorBase::updateSemanticVoxelProbabilities(
  const SemanticProbabilities& measurement_frequencies,
  SemanticProbabilities* semantic_prior_probability) const {
DCHECK(semantic_prior_probability != nullptr);
DCHECK_EQ(semantic_prior_probability->size(), kTotalNumberOf2DLabels);
DCHECK_LE((*semantic_prior_probability)[0], 0.0);
DCHECK(std::isfinite((*semantic_prior_probability)[0]));
DCHECK(!semantic_prior_probability->hasNaN());
DCHECK_EQ(measurement_frequencies.size(), kTotalNumberOf2DLabels);
DCHECK_GE(measurement_frequencies.sum(), 1.0)
    << "We should at least have one measurement when calling this "
       "function.";

// Check prior is normalized!
// static constexpr bool kDebug = true;
// if (kDebug) {
//  CHECK_NEAR(semantic_prior_probability->sum(), 1.0, vxb::kFloatEpsilon);
//}

// A. Pre-multiply each column of
// likelihood matrix with corresponding measurement frequency.
// B. Compute posterior probabilities:
// Post_i = (likelihood * meas freqs)_i + prior_i;
*semantic_prior_probability +=
    semantic_log_likelihood_ * measurement_frequencies;
//LOG(INFO)<<"Semantic Prior Probability: "<<semantic_prior_probability->transpose();
DCHECK(!semantic_prior_probability->hasNaN());

// Normalize posterior probability.
// TODO(Toni): no need to normalize all the time unless someone asks
// for meaningful probabilities?
// normalizeProbabilities(semantic_prior_probability);
}

void SemanticIntegratorBase::calculateMaximumLikelihoodLabel(
  const SemanticProbabilities& semantic_posterior,
  SemanticLabel* semantic_label) const {
CHECK_NOTNULL(semantic_label);
// Return semantic label with current max probability.
// TODO(Toni): what if there is a draw?
// Warning:
//    the result is undefined if semantic_posterior contains NaN.
CHECK(!semantic_posterior.hasNaN())
    << "Eigen's maxCoeff has undefined behaviour with NaNs, fix your "
       "posteriors.";
// TODO(Toni): if the semantic_label is still unknown, then, if the
// kUnknownSemanticLabel is set to something != 0, the 0 class will be
// unfairly shown as the class for this voxel, while in reality we don't know.
semantic_posterior.maxCoeff(semantic_label);
}

// Will return a pointer to a voxel located at global_voxel_idx in the tsdf
// layer. Thread safe.
// Takes in the last_block_idx and last_block to prevent unneeded map lookups.
// If the block this voxel would be in has not been allocated, a block in
// temp_block_map_ is created/accessed and a voxel from this map is returned
// instead. Unlike the layer, accessing temp_block_map_ is controlled via a
// mutex allowing it to grow during integration.
// These temporary blocks can be merged into the layer later by calling
// updateLayerWithStoredBlocks()
SemanticVoxel* SemanticIntegratorBase::allocateStorageAndGetSemanticVoxelPtr(
    const vxb::GlobalIndex& global_voxel_idx,
    vxb::Block<SemanticVoxel>::Ptr* last_block,
    vxb::BlockIndex* last_block_idx) {
  DCHECK(last_block != nullptr);
  DCHECK(last_block_idx != nullptr);

  const vxb::BlockIndex& block_idx = vxb::getBlockIndexFromGlobalVoxelIndex(
      global_voxel_idx, semantic_voxels_per_side_inv_);

  if ((block_idx != *last_block_idx) || (*last_block == nullptr)) {
    *last_block = semantic_layer_->getBlockPtrByIndex(block_idx);
    *last_block_idx = block_idx;
  }

  // If no block at this location currently exists, we allocate a temporary
  // voxel that will be merged into the map later
  if (*last_block == nullptr) {
    // To allow temp_label_block_map_ to grow we can only let
    // one thread in at once
    std::lock_guard<std::mutex> lock(temp_semantic_block_mutex_);

    typename vxb::Layer<SemanticVoxel>::BlockHashMap::iterator it =
        temp_semantic_block_map_.find(block_idx);
    if (it != temp_semantic_block_map_.end()) {
      *last_block = it->second;
    } else {
      auto insert_status = temp_semantic_block_map_.emplace(
          block_idx,
          std::make_shared<vxb::Block<SemanticVoxel>>(
              semantic_voxels_per_side_,
              semantic_voxel_size_,
              vxb::getOriginPointFromGridIndex(block_idx,
                                               semantic_block_size_)));

      DCHECK(insert_status.second) << "Block already exists when allocating at "
                                   << block_idx.transpose();

      *last_block = insert_status.first->second;
    }
  }

  // Only used if someone calls the getAllUpdatedBlocks I believe.
  (*last_block)->updated() = true;

  const vxb::VoxelIndex local_voxel_idx = vxb::getLocalFromGlobalVoxelIndex(
      global_voxel_idx, semantic_voxels_per_side_);

  return &((*last_block)->getVoxelByVoxelIndex(local_voxel_idx));
}

// NOT THREAD SAFE
void SemanticIntegratorBase::updateSemanticLayerWithStoredBlocks() {
  vxb::BlockIndex last_block_idx;
  vxb::Block<SemanticVoxel>::Ptr block = nullptr;
  for (const std::pair<const vxb::BlockIndex, vxb::Block<SemanticVoxel>::Ptr>&
           tmp_label_block_pair : temp_semantic_block_map_) {
    semantic_layer_->insertBlock(tmp_label_block_pair);
  }
  temp_semantic_block_map_.clear();
}

float SemanticIntegratorBase::computeOccupancyBelief(const vxb::TsdfVoxel& voxel_measured_state) const {
  float sdf = voxel_measured_state.distance;
  float weight = voxel_measured_state.weight;  // Already adjusted by weight dropoff
  // If the weight is too low, return neutral belief (uncertain state)
  if (weight < kEpsilon) {
    return 0.5f;  // High uncertainty, treating it as unknown
  }
  // Define a decay factor to control confidence behavior
  constexpr float beta = 1.0f;  
  float occupancy_belief;
  if (sdf > 0) {  // Free space (before the surface)
    occupancy_belief = weight * (1.0f - std::exp(-beta * sdf));  // Confidence increases with distance
  } 
  else {  // Occupied space (behind the surface)
    occupancy_belief = weight * std::exp(beta * sdf);  // Confidence peaks at the surface, decays inside objects
  }
  return occupancy_belief;
}

void SemanticIntegratorBase::updateSemanticVoxelColor(
  const SemanticLabel& semantic_label,
  HashableColor* semantic_voxel_color) const {
CHECK_NOTNULL(semantic_voxel_color);
// Do Not modify semantic_label_color_map_ here bcs we need to remain
// thread-safe, and adding mutexes for coloring seems silly.
// Precompute, for all possible SemanticLabels (255) a color.
*semantic_voxel_color =
    semantic_config_.semantic_label_to_color_->getColorFromSemanticLabel(
        semantic_label);
    if(static_cast<int>(semantic_label) == 0){
      // Assign white color to unknown label
    *semantic_voxel_color = HashableColor(128,128,128,255);
    //LOG(INFO)<<"[SemanticIntegratorBase] Semantic Label: "<<static_cast<int>(semantic_label)<<", Color: "<<static_cast<int>(semantic_voxel_color->r)<<", "<<static_cast<int>(semantic_voxel_color->g)<<", "<<static_cast<int>(semantic_voxel_color->b);
  }
}

void SemanticIntegratorBase::updateSemanticVoxelColor(
  const kimera::SemanticLabel& semantic_label,
  const vxb::SemanticState& semantic_state,
  const ColorMode& color_mode,
  HashableColor* semantic_voxel_color, Uncertainty uncertainty) const {
    //LOG(INFO)<<"[SemanticIntegratorBase] updateSemanticVoxelColor";
CHECK_NOTNULL(semantic_voxel_color);
// Do Not modify semantic_label_color_map_ here bcs we need to remain
// thread-safe, and adding mutexes for coloring seems silly.
// Precompute, for all possible SemanticLabels (255) a color.
//LOG(INFO)<<"Semantic State size"<<semantic_state.size();
//LOG(INFO)<<"Semantic State"<<semantic_state;
//LOG(INFO)<<"Semantic Label"<<semantic_label;
   float value = semantic_state(semantic_label);
*semantic_voxel_color = semantic_config_.semantic_label_to_color_->getColorFromSemanticState(
      semantic_label, value, ColorMode::kEvidential);
  if(static_cast<int>(semantic_label) == 0){
      // Assign white color to unknown label
    *semantic_voxel_color = HashableColor(128,128,128,255);
    //LOG(INFO)<<"[SemanticIntegratorBase] Semantic Label: "<<static_cast<int>(semantic_label)<<", Color: "<<static_cast<int>(semantic_voxel_color->r)<<", "<<static_cast<int>(semantic_voxel_color->g)<<", "<<static_cast<int>(semantic_voxel_color->b);
  }
  else if (static_cast<int>(semantic_label) == kUnknownSemanticLabelId)
  {
    float unc_alpha = uncertainty/1.0f;
    *semantic_voxel_color = HashableColor::Color(255, 0, 0, static_cast<uint8_t>(255*unc_alpha));
  }
  
  //LOG(INFO)<<"[SemanticIntegratorBase] CHECK_NOTNULL(semantic_voxel_color)";

}

void SemanticIntegratorBase::handleFreeSpace(
  SemanticState& processed_measurement, float occupancy_belief, float weight) {
  
  if (semantic_config_.voxel_representation_strategy == SemanticRepresentationType::kProbability) {
    float log_odds_free_space = std::log(occupancy_belief / (1.0f - occupancy_belief));
    processed_measurement = measurement_representation_->probabilitiesToLogOdds(processed_measurement);
    processed_measurement[kFreeSpaceLabelId] = log_odds_free_space;
  } else {
    processed_measurement.setOnes();
    processed_measurement[kFreeSpaceLabelId] += occupancy_belief * weight;
  }
}

void SemanticIntegratorBase::handleUnknownSpace(SemanticState& processed_measurement, float weight) {
  float lambda = weight * 0.05f;  // Small uncertainty update
  processed_measurement = (1+lambda) * Eigen::VectorXf::Ones(processed_measurement.size());
}

void SemanticIntegratorBase::handleOccupiedSpace(SemanticState& processed_measurement) {
  if (semantic_config_.voxel_representation_strategy == SemanticRepresentationType::kProbability) {
    processed_measurement = measurement_representation_->probabilitiesToLogOdds(processed_measurement);
  }
}

void SemanticIntegratorBase::applyFusionStrategy(
  SemanticVoxel* semantic_voxel, const SemanticState& processed_measurement, float measurement_weight) {
  
  float measurement_uncertainty = 1.0f;
  if (voxel_fuser_->requiresMeasurementUncertainty()) {
    measurement_uncertainty = measurement_representation_->calculateMeasurementUncertainty(processed_measurement);
    //LOG(INFO)<<"Measurement Uncertainty: "<<measurement_uncertainty;
    CHECK(measurement_uncertainty >= 0.0f && measurement_uncertainty <= 1.0f)
        << "Measurement uncertainty must be within [0, 1].";
  }
  auto measurement = processed_measurement;
  if (semantic_config_.measurement_representation_strategy == SemanticRepresentationType::kEvidential) {
    measurement = processed_measurement - SemanticState::Ones();
  }

  voxel_fuser_->fuse(semantic_voxel, measurement, measurement_weight, measurement_uncertainty);
}

void SemanticIntegratorBase::finalizeSemanticVoxelUpdate(SemanticVoxel* semantic_voxel, voxblox::TsdfVoxel* tsdf_voxel) {
  semantic_voxel->uncertainty = uncertainty_estimator_->computeUncertainty(semantic_voxel->dirichlet_params);
  semantic_voxel->num_measurements += 1;
  
  voxel_representation_->postprocess(semantic_voxel);
  semantic_voxel->semantic_label = voxel_representation_->getSemanticLabel(
      semantic_voxel->dirichlet_params, semantic_voxel->uncertainty, 0.5f);
  
  int max_index = 0;
  auto max_value = semantic_voxel->dirichlet_params.maxCoeff(&max_index);
  //LOG(INFO) << "Semantic posterior max value: " << max_value << " at index: " << max_index;
  //LOG(INFO) << "Semantic voxel uncertainty: " << semantic_voxel->uncertainty;
  
  updateSemanticVoxelColor(semantic_voxel->semantic_label, semantic_voxel->dirichlet_params, 
                           semantic_config_.color_mode, &semantic_voxel->color, semantic_voxel->uncertainty);
  
  //LOG(INFO)<<"Semantic Label: "<<static_cast<int>(semantic_voxel->semantic_label)<<", Uncertainty: "<<semantic_voxel->uncertainty;
  
  switch (semantic_config_.color_mode) {
    case ColorMode::kColor:
      break;
    case ColorMode::kSemantic:
      tsdf_voxel->color = semantic_voxel->color;
      break;
    case ColorMode::kSemanticProbability:
      tsdf_voxel->color = semantic_voxel->color;
      break;
    case ColorMode::kEvidential:
      tsdf_voxel->color = semantic_voxel->color;
      break;
    default:
      LOG(FATAL) << "Unknown semantic color mode: "
                 << static_cast<std::underlying_type<ColorMode>::type>(semantic_config_.color_mode);
      break;
  }
  tsdf_voxel->label = semantic_voxel->semantic_label;
  tsdf_voxel->uncertainty = semantic_voxel->uncertainty;
}





}  // Namespace kimera
