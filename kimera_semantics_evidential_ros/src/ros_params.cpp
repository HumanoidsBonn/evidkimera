/**
 * @file   ros_params.cpp
 * @brief  Helper functions to parse ROS params
 * @author Antoni Rosinol
 */

#include "kimera_semantics_evidential_ros/ros_params.h"

#include <string>

#include <ros/ros.h>

#include <glog/logging.h>

#include "kimera_semantics_evidential/common.h"
#include "kimera_semantics_evidential/semantic_integrator_base.h"

namespace kimera {

std::string
getSemanticTsdfIntegratorTypeFromRosParam(const ros::NodeHandle& nh_private) {
  // Get semantic tsdf integrator type, by default using "fast"
  // (could be "merged").
  std::string semantic_tsdf_integrator_type = "fast";
  nh_private.param("method",
                   semantic_tsdf_integrator_type,
                   semantic_tsdf_integrator_type);
  return semantic_tsdf_integrator_type;
}

std::string getSemanticLabelToColorCsvFilepathFromRosParam(
    const ros::NodeHandle& nh) {
  std::string path = "semantics2labels.csv";
  nh.param("semantic_label_2_color_csv_filepath", path, path);
  return path;
}

SemanticIntegratorBase::SemanticConfig
getSemanticTsdfIntegratorConfigFromRosParam(const ros::NodeHandle& nh_private) {
  SemanticIntegratorBase::SemanticConfig semantic_config;

  // Get semantic measurement probability
  double semantic_measurement_probability =
      semantic_config.semantic_measurement_probability_;
  nh_private.param("semantic_measurement_probability",
                   semantic_measurement_probability,
                   semantic_measurement_probability);
  semantic_config.semantic_measurement_probability_ =
      static_cast<SemanticProbability>(semantic_measurement_probability);

  // Get semantic color mode
  std::string color_mode = "semantic_evidential";
  nh_private.param("semantic_color_mode", color_mode, color_mode);
  if (color_mode == "color") {
    semantic_config.color_mode = ColorMode::kColor;
  } else if (color_mode == "semantic") {
    semantic_config.color_mode = ColorMode::kSemantic;
  } else if (color_mode == "semantic_probability") {
    semantic_config.color_mode = ColorMode::kSemanticProbability;
  } else if (color_mode == "semantic_evidential") {
    semantic_config.color_mode = ColorMode::kEvidential;
  } else {
    LOG(FATAL) << "Unknown semantic color mode: " << color_mode;
  }

  // Get semantic map
  semantic_config.semantic_label_to_color_ =
      kimera::make_unique<SemanticLabel2Color>(
          getSemanticLabelToColorCsvFilepathFromRosParam(nh_private));

  // Get measurement representation strategy
  std::string measurement_representation = "evidential";
  nh_private.param("measurement_representation_strategy",
                   measurement_representation,
                   measurement_representation);
  if (measurement_representation == "one_hot") {
    semantic_config.measurement_representation_strategy = 
        SemanticRepresentationType::kOneHot;
  } else if (measurement_representation == "probability") {
    semantic_config.measurement_representation_strategy = 
        SemanticRepresentationType::kProbability;
  } else if (measurement_representation == "evidential") {
    semantic_config.measurement_representation_strategy = 
        SemanticRepresentationType::kEvidential;
  } else {
    LOG(FATAL) << "Unknown measurement representation strategy: " 
               << measurement_representation;
  }

  // Get voxel representation strategy
  std::string voxel_representation = "evidential";
  nh_private.param("voxel_representation_strategy",
                   voxel_representation,
                   voxel_representation);
  if (voxel_representation == "probability") {
    semantic_config.voxel_representation_strategy = 
        SemanticRepresentationType::kProbability;
  } else if (voxel_representation == "evidential") {
    semantic_config.voxel_representation_strategy = 
        SemanticRepresentationType::kEvidential;
  } else {
    LOG(FATAL) << "Unknown voxel representation strategy: " 
               << voxel_representation;
  }

  // Get voxel fusion strategy
  std::string voxel_fusion = "simple";
  nh_private.param("voxel_fusion_strategy",
                   voxel_fusion,
                   voxel_fusion);
  if (voxel_fusion == "none") {
    semantic_config.voxel_fusion_strategy = VoxelLevelFusionType::kNone;
  } else if (voxel_fusion == "simple") {
    semantic_config.voxel_fusion_strategy = VoxelLevelFusionType::kSimple;
  } else if (voxel_fusion == "measurement_uncertainty") {
    semantic_config.voxel_fusion_strategy = VoxelLevelFusionType::kMeasurementUncertainty;
  } else if (voxel_fusion == "tsdf_weighted_uncertainty") {
    semantic_config.voxel_fusion_strategy = VoxelLevelFusionType::kTsdfWeightedUncertainty;
  } else if (voxel_fusion == "voxel_measurement_uncertainty") {
    semantic_config.voxel_fusion_strategy = VoxelLevelFusionType::kVoxelMeasurementUncertainty;
  } else if (voxel_fusion == "divergence_weighted") {
    semantic_config.voxel_fusion_strategy = VoxelLevelFusionType::kDivergenceWeighted;
  } else {
    LOG(FATAL) << "Unknown voxel fusion strategy: " << voxel_fusion;
  }

  // Get uncertainty strategy
  std::string uncertainty_strategy = "total_uncertainty";
  nh_private.param("uncertainty_strategy",
                   uncertainty_strategy,
                   uncertainty_strategy);
  if (uncertainty_strategy == "none") {
    semantic_config.uncertainty_strategy = UncertaintyEstimatorType::kNone;
  } else if (uncertainty_strategy == "total_uncertainty") {
    semantic_config.uncertainty_strategy = UncertaintyEstimatorType::kTotalUncertainty;
  } else if (uncertainty_strategy == "variance") {
    semantic_config.uncertainty_strategy = UncertaintyEstimatorType::kVariance;
  } else if (uncertainty_strategy == "dempster_shafer") {
    semantic_config.uncertainty_strategy = UncertaintyEstimatorType::kDempsterShafer;
  } else if (uncertainty_strategy == "probability_variance") {
    semantic_config.uncertainty_strategy = UncertaintyEstimatorType::kProbabilityVariance;
  } else if (uncertainty_strategy == "probability_entropy") {
    semantic_config.uncertainty_strategy = UncertaintyEstimatorType::kProbabilityEntropy;
  } else {
    LOG(FATAL) << "Unknown uncertainty strategy: " << uncertainty_strategy;
  }
  bool use_viewpoint_similarity = semantic_config.use_viewpoint_similarity;
  nh_private.param("use_viewpoint_similarity",use_viewpoint_similarity,use_viewpoint_similarity);
  semantic_config.use_viewpoint_similarity = use_viewpoint_similarity;
  // Get viewpoint obliqueness usage flag
  bool use_viewpoint_obliqueness = semantic_config.use_viewpoint_obliqueness;
  nh_private.param("use_viewpoint_obliqueness", use_viewpoint_obliqueness, use_viewpoint_obliqueness);
  semantic_config.use_viewpoint_obliqueness = use_viewpoint_obliqueness;
  
  double uncertainty_threshold = semantic_config.uncertainty_threshold;
  
  nh_private.param("uncertainty_threshold",uncertainty_threshold,uncertainty_threshold);
  semantic_config.uncertainty_threshold = static_cast<voxblox::FloatingPoint>(uncertainty_threshold);

  // Get label assignment threshold
  double label_assignment_threshold = semantic_config.label_assignment_threshold;
  nh_private.param("label_assignment_threshold", label_assignment_threshold, label_assignment_threshold);
  semantic_config.label_assignment_threshold = static_cast<voxblox::FloatingPoint>(label_assignment_threshold);

  return semantic_config;
}


}  // namespace kimera
