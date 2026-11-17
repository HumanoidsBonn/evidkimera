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
 * @file   semantic_tsdf_integrator_fast.cpp
 * @brief  Integrator of semantic and geometric information
 * @author Antoni Rosinol
 */

#include "kimera_semantics_evidential/semantic_kernel_integrator_fast.h"

#include <list>
#include <memory>
#include <utility>

#include <voxblox/utils/timing.h>

#include "kimera_semantics_evidential/color.h"
#include "kimera_semantics_evidential/conversions.h"

namespace kimera {

FastSemanticKernelTsdfIntegrator::FastSemanticKernelTsdfIntegrator(
    const Config& config,
    const SemanticConfig& semantic_config,
    vxb::Layer<vxb::TsdfVoxel>* tsdf_layer,
    vxb::Layer<SemanticVoxel>* semantic_layer)
    : TsdfIntegratorBase(config, tsdf_layer),
      SemanticIntegratorBase(semantic_config, semantic_layer) {
  if(semantic_config_.kernel_in_measurement_mode){
    LOG(INFO)<<"Kernel in measurement mode"<<std::endl;
    global_training_data_.reserve(semantic_config_.width*semantic_config_.height);
  }
  global_min_bounds_ = Eigen::Vector3f::Constant(std::numeric_limits<float>::max());
  global_max_bounds_ = Eigen::Vector3f::Constant(std::numeric_limits<float>::lowest());
}

void FastSemanticKernelTsdfIntegrator::integrateSemanticFunction(
    const vxb::Transformation& T_G_C,
    const vxb::Pointcloud& points_C,
    const vxb::Colors& colors,
    const vxb::Semantics& semantic_measurements,
    const vxb::DepthUncertainties& depth_uncertainties,
    const float& viewpoint_similarity_weight,
    const bool freespace_points,
    vxb::ThreadSafeIndex* index_getter) {
  DCHECK(index_getter != nullptr);
  
  VectorSemanticPointData local_training_data;  // Thread-local storage for efficiency

  // Thread-local bounds
  Eigen::Vector3f local_min_bounds = Eigen::Vector3f::Constant(std::numeric_limits<float>::max());
  Eigen::Vector3f local_max_bounds = Eigen::Vector3f::Constant(std::numeric_limits<float>::lowest());

  size_t point_idx;
  while (index_getter->getNextIndex(&point_idx) &&
         (std::chrono::duration_cast<std::chrono::microseconds>(
              std::chrono::steady_clock::now() - integration_start_time_)
              .count() < config_.max_integration_time_s * 1000000)) {
    const vxb::Point& point_C = points_C[point_idx];
    const vxb::Color& color = colors[point_idx];
    const SemanticState& semantic_measurement = semantic_measurements[point_idx];
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

    local_min_bounds = local_min_bounds.cwiseMin(point_G);
    local_max_bounds = local_max_bounds.cwiseMax(point_G);

    if(semantic_config_.kernel_in_measurement_mode)
    {
      local_training_data.global_voxel_index_vector.push_back(global_voxel_idx);
      local_training_data.semantic_state_vector.push_back(semantic_measurement);
      local_training_data.point_G_vector.push_back(point_G);
      local_training_data.weight_vector.push_back(getVoxelWeight(point_C));
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
      const float weight = getVoxelWeight(point_C);

      updateTsdfVoxel(origin, point_G, global_voxel_idx, color, weight, voxel);

      SemanticVoxel* semantic_voxel = allocateStorageAndGetSemanticVoxelPtr(
          global_voxel_idx, &semantic_block, &semantic_block_idx);
      //vxb::SemanticState semantic_label_frequencies =
      //    vxb::SemanticState::Zero();
      //CHECK_LT(semantic_label, semantic_label_frequencies.size());
      //semantic_label_frequencies[semantic_label] += 1.0f;
      //updateSemanticVoxel(global_voxel_idx, semantic_measurement, weight, &mutexes_, voxel, semantic_voxel);
    }
  }
  // Append local training data to the global container
  {
    std::lock_guard<std::mutex> lock(global_training_data_mutex_);
    appendVector(global_training_data_.global_voxel_index_vector, local_training_data.global_voxel_index_vector);
    appendVector(global_training_data_.point_G_vector, local_training_data.point_G_vector);
    appendVector(global_training_data_.semantic_state_vector, local_training_data.semantic_state_vector);
    appendVector(global_training_data_.weight_vector, local_training_data.weight_vector);
  }
  // Update global bounds in a thread-safe manner
  {
    std::lock_guard<std::mutex> lock(bounds_mutex_);
    global_min_bounds_ = global_min_bounds_.cwiseMin(local_min_bounds);
    global_max_bounds_ = global_max_bounds_.cwiseMax(local_max_bounds);
  }
}

void FastSemanticKernelTsdfIntegrator::integratePointCloud(
    const vxb::Transformation& T_G_C,
    const vxb::Pointcloud& points_C,
    const vxb::Colors& colors,
    const bool freespace_points) {
  LOG(WARNING)<<"integratePointCloud: This method not implemented"<<std::endl;
}

void FastSemanticKernelTsdfIntegrator::integratePointCloud(
    const vxb::Transformation& T_G_C,
    const vxb::Pointcloud& points_C,
    const vxb::Colors& colors,
    const vxb::Semantics& semantic_measurements,
    const bool freespace_points) {
  LOG(WARNING)<<"integratePointCloud"<<std::endl;
  
  vxb::timing::Timer integrate_timer("integrate/fast");
  CHECK_EQ(points_C.size(), colors.size());
  CHECK_EQ(points_C.size(), semantic_measurements.size());

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
        &FastSemanticKernelTsdfIntegrator::integrateSemanticFunction,
        this,
        T_G_C,
        points_C,
        colors,
        semantic_measurements,
        freespace_points,
        index_getter.get());
  }

  for (std::thread& thread : integration_threads) {
    thread.join();
  }

  integrate_timer.Stop();



  vxb::timing::Timer insertion_timer("inserting_missed_tsdf_blocks");
  updateLayerWithStoredBlocks();
  insertion_timer.Stop();
  vxb::timing::Timer fusion_timer("kernel_fusion");
  BoundingBox global_bbox(global_min_bounds_, global_max_bounds_);
  float bbox_padding_in_m = semantic_config_.bbox_padding_in_voxels*voxel_size_;
  global_bbox.expand(bbox_padding_in_m);
  performKernelFusion(global_bbox, layer_, semantic_layer_, global_training_data_, 4*voxel_size_);
  updateSemanticLayerWithStoredBlocks();
  
}

BoundingBox FastSemanticKernelTsdfIntegrator::computeBoundingBox(
    const std::unordered_set<vxb::GlobalIndex>& voxel_observed_set) const {
  Eigen::Vector3f min_bounds = Eigen::Vector3f::Constant(std::numeric_limits<float>::max());
  Eigen::Vector3f max_bounds = Eigen::Vector3f::Constant(std::numeric_limits<float>::lowest());

  for (const auto& voxel_idx : voxel_observed_set) {
    Eigen::Vector3f voxel_center; //to be implemented
    min_bounds = min_bounds.cwiseMin(voxel_center);
    max_bounds = max_bounds.cwiseMax(voxel_center);
  }

  return {min_bounds, max_bounds};
}

void FastSemanticKernelTsdfIntegrator::performKernelFusion(
  const BoundingBox& global_bbox,
  const vxb::Layer<vxb::TsdfVoxel>* tsdf_layer,
  const vxb::Layer<SemanticVoxel>* semantic_layer,
  const VectorSemanticPointData& training_data,
  float radius) {
  
  // Generate block indices within the bounding box
  auto block_indices = generateBlockIndicesInBoundingBox(global_bbox, block_size_);

  // Initialize the SurfaceEvaluator
  SurfaceEvaluator surface_evaluator(tsdf_layer, semantic_layer,  tsdf_layer->voxel_size());

  auto worker_func = [&](const vxb::BlockIndex& block_idx) {
    auto block = tsdf_layer->getBlockPtrByIndex(block_idx);
    if (block) {
      processBlockWithKernel(block, training_data, surface_evaluator, radius);
    }
  };

  // Use processBlocksInParallel to process block indices and execute the worker function
  processBlocksInParallel(block_indices, worker_func, 8);
}

std::vector<vxb::BlockIndex> FastSemanticKernelTsdfIntegrator::generateBlockIndicesInBoundingBox(
  const BoundingBox& bbox, float block_size) const {
  
  std::vector<vxb::BlockIndex> block_indices;

  for (float x = bbox.min.x(); x <= bbox.max.x(); x += block_size) {
    for (float y = bbox.min.y(); y <= bbox.max.y(); y += block_size) {
      for (float z = bbox.min.z(); z <= bbox.max.z(); z += block_size) {
        block_indices.push_back(vxb::getGridIndexFromPoint<vxb::BlockIndex>(
            Eigen::Vector3f(x, y, z), block_size));
      }
    }
  }
  return block_indices;
}

void FastSemanticKernelTsdfIntegrator::processBlockWithKernel(
  const std::shared_ptr<const vxb::Block<vxb::TsdfVoxel>>& block,
  const VectorSemanticPointData& training_data_vector,
  const SurfaceEvaluator& surface_evaluator,
  const float& radius) {
  if (!block) return;

  // Step 1: Collect query points in the block
  VectorSemanticPointData query_data_vector = collectQueryPointsInBlock(block, surface_evaluator);
  auto training_points_indices = collectExtendedTrainingPointIndicesForBlock(query_data_vector, training_data_vector, radius);

  if (semantic_config_.kernel_params.use_cuda) {
    // Torch backend
    torch::Tensor query_matrix = convertToTorchTensor(&(query_data_vector.point_G_vector), {}, DataType::Points);
    torch::Tensor training_matrix = convertToTorchTensor(&(training_data_vector.point_G_vector), training_points_indices, DataType::Points);
    torch::Tensor training_point_semantics = convertToTorchTensor(&(training_data_vector.semantic_state_vector), training_points_indices, DataType::SemanticState);
    torch::Tensor tsdf_weights;
    if(semantic_config_.kernel_params.decorator_type == KernelDecoratorType::kTSDF)
    {
      tsdf_weights = convertToTorchTensor(&(training_data_vector.weight_vector), training_points_indices, DataType::Weight);
    }
    std::shared_ptr<KernelDecorator<torch::Tensor>> kernel = createDecoratedKernel<torch::Tensor>(semantic_config_.kernel_params, training_data_vector, training_points_indices, tsdf_weights.to(torch::kCUDA)); 

    // Step 4: Compute the kernel matrix
    torch::Tensor kernel_matrix = kernel->computeKernel(query_matrix.to(torch::kCUDA), training_matrix.to(torch::kCUDA));

    // Step 5: Perform fusion
    torch::Tensor query_point_semantics = (kernel_matrix.matmul(training_point_semantics.to(kernel_matrix.device()))).detach().to(torch::kCPU);
    //applyFusionTorch(block, query_point_semantics);
  } 
  else {
    // Eigen backend
    Eigen::MatrixXf query_matrix = convertToEigen(&(query_data_vector).point_G_vector, {}, DataType::Points);
    Eigen::MatrixXf training_matrix = convertToEigen(&(training_data_vector.point_G_vector), training_points_indices, DataType::Points);
    Eigen::MatrixXf training_point_semantics = convertToEigen(&(training_data_vector.semantic_state_vector), training_points_indices, DataType::SemanticState);
    Eigen::MatrixXf tsdf_weights;
    if(semantic_config_.kernel_params.decorator_type == KernelDecoratorType::kTSDF)
    {
      tsdf_weights = convertToEigen(&(training_data_vector.weight_vector), training_points_indices, DataType::Weight);
    }
    std::shared_ptr<KernelDecorator<Eigen::MatrixXf>> kernel = createDecoratedKernel<Eigen::MatrixXf>(semantic_config_.kernel_params, training_data_vector, training_points_indices, tsdf_weights); 

    // Step 4: Compute the kernel matrix
    Eigen::MatrixXf kernel_matrix = kernel->computeKernel(query_matrix, training_matrix);

    // Step 5: Perform fusion
    Eigen::MatrixXf query_point_semantics = kernel_matrix * training_point_semantics;
    //applyFusionEigen(block, query_point_semantics);
  }
}
  // Step 5: Update semantic voxels based on kernel results
  // for (size_t i = 0; i < query_point_semantics.rows(); ++i) {
    
  //   // Allocate dynamic voxels using the SurfaceEvaluator
  //   if (semantic_config_.allocate_new_voxels)
  //     allocateDynamicVoxels(query_points[i], kernel_output, config_.evidence_threshold, surface_evaluator);
  // }

template <typename WorkerFunc> 
void FastSemanticKernelTsdfIntegrator::processBlocksInParallel(std::vector<vxb::BlockIndex>& block_indices_vector, WorkerFunc worker_func, size_t num_threads) {
  // Create a thread-safe queue
  std::queue<vxb::BlockIndex> block_indices_queue;
  for (const auto& block : block_indices_vector) {
    block_indices_queue.push(block);
  }

  // Mutex for thread-safe access to the queue
  std::mutex queue_mutex;

  // Thread worker function
  auto worker = [&]() {
    while (true) {
      vxb::BlockIndex block_idx;

      // Retrieve a block from the queue
      {
        std::lock_guard<std::mutex> lock(queue_mutex);
        if (block_indices_queue.empty()) return;  // Exit if the queue is empty
        block_idx = block_indices_queue.front();
        block_indices_queue.pop();
      }

      // Process the block using the worker function
      worker_func(block_idx);
    }
  };

  // Launch threads
  std::vector<std::thread> threads;
  for (size_t i = 0; i < num_threads; ++i) {
    threads.emplace_back(worker);
  }

  // Join threads
  for (auto& thread : threads) {
    thread.join();
  }
}

VectorSemanticPointData FastSemanticKernelTsdfIntegrator::collectQueryPointsInBlock(
    const std::shared_ptr<const vxb::Block<vxb::TsdfVoxel>>& block, 
    const SurfaceEvaluator& surface_evaluator) const {
  
  if (!block) return {};

  const int vps = block->voxels_per_side();
  VectorSemanticPointData query_data_vector(block->num_voxels());

  for (size_t linear_idx = 0; linear_idx < block->num_voxels(); ++linear_idx) 
  {
    SemanticPointData query_data_point;
    query_data_vector.global_voxel_index_vector.emplace_back(voxblox::getGlobalVoxelIndexFromBlockAndVoxelIndex(block->block_index(), block->computeVoxelIndexFromLinearIndex(linear_idx), vps));
    query_data_vector.weight_vector.emplace_back(block->getVoxelByLinearIndex(linear_idx).weight);
    if (semantic_config_.enable_position_correction) 
    {
      query_data_vector.point_G_vector.emplace_back(surface_evaluator.computeCorrectedPosition(block, linear_idx));
    }
    else
    {
      query_data_vector.point_G_vector.emplace_back(block->computeCoordinatesFromLinearIndex(linear_idx));
    }
    query_data_vector.semantic_state_vector.emplace_back(SemanticState());
  }
  return query_data_vector;
}

std::vector<size_t> FastSemanticKernelTsdfIntegrator::collectExtendedTrainingPointIndicesForBlock(
    const VectorSemanticPointData& query_data,
    const VectorSemanticPointData& training_data,
    float radius) const {

  std::vector<size_t> collected_indices;
  collected_indices.reserve(training_data.point_G_vector.size());

  for (size_t i = 0; i < training_data.size(); ++i) {
    const auto& training_point_G = training_data.point_G_vector[i];
    for (const auto& query_data_point_G : query_data.point_G_vector) {
      if ((query_data_point_G - training_point_G).norm() <= radius) {
        collected_indices.push_back(i);
        break;  // Avoid duplicate additions for the same training point
      }
    }
  }
  return collected_indices;
}

vxb::Block<SemanticVoxel>::Ptr FastSemanticKernelTsdfIntegrator::getSemanticBlock(
    const vxb::BlockIndex& block_idx) {
  // Check the temporary block map first
  {
    std::lock_guard<std::mutex> lock(temp_semantic_block_mutex_);
    auto it = temp_semantic_block_map_.find(block_idx);
    if (it != temp_semantic_block_map_.end()) {
      return it->second;
    }
  }

  // Fallback to the main semantic layer
  vxb::Block<SemanticVoxel>::Ptr block = semantic_layer_->getBlockPtrByIndex(block_idx);
  if (block) {
    return block;
  }

  // Block not found
  LOG(WARNING) << "Block not found in temp map or layer for index: " << block_idx.transpose();
  return nullptr;
}

// void FastSemanticKernelTsdfIntegrator::allocateDynamicVoxels(
//   const vxb::Block<vxb::TsdfVoxel>::Ptr& block,
//   const Eigen::Vector3f& query_point,
//   const Eigen::VectorXf& kernel_output,
//   float evidence_threshold,
//   const SurfaceEvaluator& surface_evaluator) {
//   // Compute semantic evidence
//   float evidence = kernel_output.mean();

//   // Check if evidence is strong enough
//   if(evidence <= evidence_threshold) return;

//   // Retrieve the global voxel index
//   vxb::GlobalIndex voxel_idx = vxb::getGridIndexFromPoint<vxb::GlobalIndex>(
//       query_point, tsdf_layer_->voxel_size());

//   // Validate surface crossing
//   if (!surface_evaluator.hasSurfaceCrossing(voxel_idx)) return;

//   // Interpolate TSDF distance and weight
//   float interpolated_distance = 0.0f;
//   float interpolated_weight = 0.0f;
//   if (!surface_evaluator.interpolateTsdfDistanceAndWeight(
//           voxel_idx, interpolated_distance, interpolated_weight)) {
//     return;  // Interpolation failed
//   }

//   // Allocate TSDF voxel
//   vxb::TsdfVoxel* tsdf_voxel = allocateStorageAndGetVoxelPtr(
//       voxel_idx, &last_tsdf_block_, &last_tsdf_block_idx_);

//   if (tsdf_voxel) {
//     tsdf_voxel->distance = interpolated_distance;
//     tsdf_voxel->weight = interpolated_weight;
//   }

//   // Allocate semantic voxel
//   SemanticVoxel* semantic_voxel = allocateStorageAndGetSemanticVoxelPtr(
//       voxel_idx, &last_semantic_block_, &last_semantic_block_idx_);

//   if (semantic_voxel) {
//     semantic_voxel->dirichlet_params += kernel_output;  // Update semantic evidence
//   }
// }

// void FastSemanticKernelTsdfIntegrator::updateVoxelStatesAfterKernelFusion(
//   const std::unordered_set<vxb::GlobalIndex>& voxel_observed_set) {
  
//   for (const auto& voxel_idx : voxel_observed_set) {
//     SemanticVoxel* voxel = voxel_grid_[voxel_idx];
//     if (!voxel) continue;

//     voxel->uncertainty = uncertainty_estimator_->computeVoxelUncertainty(voxel);
//     int most_likely_class = 0;
//     voxel->dirichlet_params.maxCoeff(&most_likely_class);
//     voxel->semantics = most_likely_class;
//     voxel->state = (most_likely_class == 0) ? SemanticVoxel::State::FREE : SemanticVoxel::State::OCCUPIED;
//   }
// }
//Not used but for reference

std::vector<vxb::Block<vxb::TsdfVoxel>::Ptr> FastSemanticKernelTsdfIntegrator:: getTsdfBlocksInBoundingBox(
    const BoundingBox& bbox,
    const vxb::Layer<vxb::TsdfVoxel>* tsdf_layer) {
  std::vector<vxb::Block<vxb::TsdfVoxel>::Ptr> tsdf_blocks;

  // Compute block size from the TSDF layer^
  const float block_size = tsdf_layer->block_size();

  // Compute block indices that intersect the bounding box
  const Eigen::Vector3f bbox_min = bbox.min;
  const Eigen::Vector3f bbox_max = bbox.max;

  // Iterate over potential blocks in the bounding box
  for (float x = bbox_min.x(); x <= bbox_max.x(); x += block_size) {
    for (float y = bbox_min.y(); y <= bbox_max.y(); y += block_size) {
      for (float z = bbox_min.z(); z <= bbox_max.z(); z += block_size) {
        // Get the block index for the current point
        vxb::BlockIndex block_idx = vxb::getGridIndexFromPoint<vxb::BlockIndex>(
            Eigen::Vector3f(x, y, z), block_size);

        // Retrieve the block from the TSDF layer
        vxb::Block<vxb::TsdfVoxel>::Ptr block; // = tsdf_layer->getBlockPtrByIndex(block_idx);
        if (block) {
          tsdf_blocks.push_back(block);
        }
      }
    }
  }

  return tsdf_blocks;
}



}  // Namespace kimera
