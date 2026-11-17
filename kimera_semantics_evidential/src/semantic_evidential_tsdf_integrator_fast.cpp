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
 * @file   semantic_evidential_tsdf_integrator_fast.cpp
 * @brief  Integrator of semantic and geometric information
 * @author Antoni Rosinol
 */

#include "kimera_semantics_evidential/semantic_evidential_tsdf_integrator_fast.h"

#include <list>
#include <memory>
#include <utility>

#include <voxblox/utils/timing.h>

#include "kimera_semantics_evidential/color.h"

namespace kimera {

FastSemanticEvidentialTsdfIntegrator::FastSemanticEvidentialTsdfIntegrator(
    const Config& config,
    const SemanticConfig& semantic_config,
    vxb::Layer<vxb::TsdfVoxel>* tsdf_layer,
    vxb::Layer<SemanticVoxel>* semantic_layer)
    : TsdfIntegratorBase(config, tsdf_layer),
      SemanticIntegratorBase(semantic_config, semantic_layer) {
  viewpoint_similarity_estimator_ = std::make_unique<ViewpointSimilarityEstimator>();
      }

void FastSemanticEvidentialTsdfIntegrator::integrateSemanticFunction(
    const vxb::Transformation& T_G_C,
    const vxb::Pointcloud& points_C,
    const vxb::Colors& colors,
    const vxb::Semantics& semantic_measurements,
    const vxb::DepthUncertainties& depth_uncertainties,
    const float& viewpoint_similarity_weight,
    const bool freespace_points,
    vxb::ThreadSafeIndex* index_getter) {
  DCHECK(index_getter != nullptr);

  size_t point_idx;
  while (index_getter->getNextIndex(&point_idx) &&
         (std::chrono::duration_cast<std::chrono::microseconds>(
              std::chrono::steady_clock::now() - integration_start_time_)
              .count() < config_.max_integration_time_s * 1000000)) {
    const vxb::Point& point_C = points_C[point_idx];
    const vxb::Color& color = colors[point_idx];
    const SemanticState& semantic_measurement = semantic_measurements[point_idx];
    //LOG(INFO)<<"Semantic Measurement: "<<semantic_measurement.transpose()<<std::endl;
    const DepthUncertainty& depth_uncertainty = depth_uncertainties[point_idx];
    bool is_clearing;
    if (!isPointValid(point_C, freespace_points, &is_clearing)) {
      continue;
    }
    
    const vxb::Point origin = T_G_C.getPosition();
    const vxb::Point point_G = T_G_C * point_C;
    // Checks to see if another ray in this scan has already started 'close'
    // to this location. If it has then we skip ray casting this point. We
    // measure if a start location is 'close' to another points by inserting
    // the point into a set of voxels. This voxel set has a resolution
    // start_voxel_subsampling_factor times higher then the voxel size.
    vxb::GlobalIndex global_voxel_idx;
    global_voxel_idx = vxb::getGridIndexFromPoint<vxb::GlobalIndex>(
        point_G, config_.start_voxel_subsampling_factor * voxel_size_inv_);
    if (!start_voxel_approx_set_.replaceHash(global_voxel_idx)) {
      continue;
    }

    static constexpr bool cast_from_origin = false;
    vxb::RayCaster ray_caster(origin,
                              point_G,
                              is_clearing,
                              config_.voxel_carving_enabled,
                              config_.max_ray_length_m,
                              voxel_size_inv_,
                              config_.default_truncation_distance,
                              cast_from_origin);

    int64_t consecutive_ray_collisions = 0;

    vxb::Block<vxb::TsdfVoxel>::Ptr block = nullptr;
    vxb::BlockIndex block_idx;
    vxb::Block<SemanticVoxel>::Ptr semantic_block = nullptr;
    vxb::BlockIndex semantic_block_idx;


    while (ray_caster.nextRayIndex(&global_voxel_idx)) {
      // Check if the current voxel has been seen by any ray cast this scan.
      // If it has increment the consecutive_ray_collisions counter, otherwise
      // reset it. If the counter reaches a threshold we stop casting as the
      // ray is deemed to be contributing too little new information.
      if (!voxel_observed_approx_set_.replaceHash(global_voxel_idx)) {
        ++consecutive_ray_collisions;
      } else {
        consecutive_ray_collisions = 0;
      }
      if (consecutive_ray_collisions > config_.max_consecutive_ray_collisions) {
        break;
      }
      // Update the bounding box with the new voxel
      // min_bounds = min_bounds.cwiseMin(point_G);
      // max_bounds = max_bounds.cwiseMax(point_G);
      vxb::TsdfVoxel* voxel =
          allocateStorageAndGetVoxelPtr(global_voxel_idx, &block, &block_idx);
      const float weight = getVoxelWeight(point_C, depth_uncertainty);

      vxb::TsdfVoxel voxel_measured_state;
      //float viewpoint_similarity_weight = 1.0f;

      updateTsdfVoxel(origin, point_G, global_voxel_idx, color, weight, depth_uncertainty, 1.0f, voxel_measured_state, voxel);
      //viewpoint_weighted_weight = viewpoint_similarity_weight*weight;
      //updateTsdfVoxel(origin, point_G, global_voxel_idx, color, weight, voxel);
      if (getFrameIdx() % 5 == 0) 
      {
        SemanticVoxel* semantic_voxel = allocateStorageAndGetSemanticVoxelPtr(
          global_voxel_idx, &semantic_block, &semantic_block_idx);
        updateSemanticVoxel(global_voxel_idx, semantic_measurement, viewpoint_similarity_weight, depth_uncertainty, &mutexes_, voxel, voxel_measured_state, semantic_voxel);
      }
    }
  }
}

void FastSemanticEvidentialTsdfIntegrator::integratePointCloud(
    const vxb::Transformation& T_G_C,
    const vxb::Pointcloud& points_C,
    const vxb::Colors& colors,
    const bool freespace_points) {
  LOG(WARNING)<<"integratePointCloud: This method not implemented"<<std::endl;
}

void FastSemanticEvidentialTsdfIntegrator::integratePointCloud(
  const vxb::Transformation& T_G_C,
  const vxb::Pointcloud& points_C,
  const vxb::Colors& colors,
  const vxb::Semantics& semantic_measurements,
  const bool freespace_points) {
  
  vxb::DepthUncertainties depth_uncertainties;
  depth_uncertainties.resize(points_C.size());
  for (auto& depth_uncertainty : depth_uncertainties) {
    depth_uncertainty.setZero();
  }
  integratePointCloud(T_G_C, points_C, colors, semantic_measurements, depth_uncertainties, freespace_points);
}
void FastSemanticEvidentialTsdfIntegrator::integratePointCloud(
  const vxb::Transformation& T_G_C,
  const vxb::Pointcloud& points_C,
  const vxb::Colors& colors,
  const vxb::Semantics& semantic_measurements,
  const vxb::DepthUncertainties& depth_uncertainties,
  const bool freespace_points) {
  //LOG(WARNING)<<"integratePointCloud With Semantics and Depth Uncertainties"<<std::endl;
  CHECK_EQ(points_C.size(), colors.size());
  CHECK_EQ(points_C.size(), semantic_measurements.size());
  CHECK_EQ(points_C.size(), depth_uncertainties.size());
  //LOG(INFO)<<"T_G_C: "<<T_G_C.getPosition().transpose()<<std::endl;
  //LOG(INFO)<<"T_G_C: "<<T_G_C.getRotation().coeffs().transpose()<<std::endl;
  
  float viewpoint_similarity = viewpoint_similarity_estimator_->computeSimilarity(T_G_C);
  
  
  vxb::timing::Timer integrate_timer("integrate/fast");
  CHECK_EQ(points_C.size(), colors.size());

  integration_start_time_ = std::chrono::steady_clock::now();

  static int64_t reset_counter = 0;
  if ((++reset_counter) >= config_.clear_checks_every_n_frames) {
    reset_counter = 0;
    start_voxel_approx_set_.resetApproxSet();
    voxel_observed_approx_set_.resetApproxSet();
  }

  std::unique_ptr<vxb::ThreadSafeIndex> index_getter(
      vxb::ThreadSafeIndexFactory::get(config_.integration_order_mode,
                                       points_C));

  std::list<std::thread> integration_threads;
  for (size_t i = 0; i < config_.integrator_threads; ++i) {
    integration_threads.emplace_back(
        &FastSemanticEvidentialTsdfIntegrator::integrateSemanticFunction,
        this,
        T_G_C,
        points_C,
        colors,
        semantic_measurements,
        depth_uncertainties,
        viewpoint_similarity,
        freespace_points,
        index_getter.get());
  }

  for (std::thread& thread : integration_threads) {
    thread.join();
  }
  viewpoint_similarity_estimator_->addViewpoint(T_G_C);
  integrate_timer.Stop();

  vxb::timing::Timer insertion_timer("inserting_missed_blocks");
  updateLayerWithStoredBlocks();
  if (getFrameIdx() % 5 == 0)
  {
    updateSemanticLayerWithStoredBlocks();
  }
  incrementFrameIdx();
  insertion_timer.Stop();
}

}  // Namespace kimera
