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

#include "kimera_semantics_evidential/common.h"
#include "kimera_semantics_evidential/semantic_voxel.h"
#include <voxblox/core/layer.h>
#include <voxblox/core/voxel.h>

namespace kimera{

class SurfaceEvaluator {
public:
SurfaceEvaluator( const vxb::Layer<vxb::TsdfVoxel>* tsdf_layer, const vxb::Layer<SemanticVoxel>* semantic_layer, const float& voxel_size) : tsdf_layer_(tsdf_layer), semantic_layer_(semantic_layer), voxel_size_(voxel_size)
  {}

/**
 * @brief Checks if a surface crossing exists in the given voxel.
 * @param voxel_idx The global index of the voxel to check.
 * @return True if a surface crossing is detected, false otherwise.
 */
bool hasSurfaceCrossing(
  const vxb::GlobalIndex& voxel_idx) const;

/**
 * @brief Interpolates the TSDF distance and weight based on valid neighbors.
 * @param voxel_idx The global index of the voxel to interpolate.
 * @param distance[out] Interpolated TSDF distance.
 * @param weight[out] Interpolated TSDF weight.
 * @return True if interpolation is successful, false otherwise.
 */
bool interpolateTsdfDistanceAndWeight(
  const vxb::GlobalIndex& voxel_idx,
  float& distance,
  float& weight) const;

/**
 * @brief Checks if the neighbors of a voxel are valid (not free or unknown).
 * @param voxel_idx The global index of the voxel to check.
 * @return True if neighbors are valid, false otherwise.
 */
bool areNeighborsValid(
  const vxb::GlobalIndex& voxel_idx) const;

Eigen::Vector3f computeCorrectedPosition(
  const std::shared_ptr<const vxb::Block<vxb::TsdfVoxel>>& block,
  const size_t & local_index) const;

private:
const vxb::Layer<vxb::TsdfVoxel>* tsdf_layer_;
const vxb::Layer<SemanticVoxel>* semantic_layer_;
const float voxel_size_;

std::vector<vxb::GlobalIndex> neighbor_offsets_ = 
{{1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, -1, 0}, {0, 0, 1}, {0, 0, -1}};

protected:
/**
 * @brief Computes the gradient of the TSDF distance at a given voxel.
 * @param block The block containing the voxel.
 * @param local_index The local index of the voxel in the block.
 * @return The gradient of the TSDF distance.
 */
Eigen::Vector3f computeGradient(
  const vxb::Block<vxb::TsdfVoxel>::Ptr& block,
  const Eigen::Vector3i& local_index) const; 

};


}