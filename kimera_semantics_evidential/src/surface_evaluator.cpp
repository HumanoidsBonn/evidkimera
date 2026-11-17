
#include "kimera_semantics_evidential/surface_evaluator.h"

using namespace kimera;

bool SurfaceEvaluator::hasSurfaceCrossing(const vxb::Layer<vxb::TsdfVoxel>* tsdf_layer, const vxb::GlobalIndex& voxel_idx) const {
  const vxb::TsdfVoxel* center_voxel = tsdf_layer->getVoxelPtrByGlobalIndex(voxel_idx);
  if (!center_voxel || center_voxel->weight <= 0) return false;

  float center_tsdf = center_voxel->distance;

  for (const auto& offset : neighbor_offsets_) {
    vxb::GlobalIndex neighbor_idx = voxel_idx + offset;

    const vxb::TsdfVoxel* neighbor_voxel = tsdf_layer->getVoxelPtrByGlobalIndex(neighbor_idx);
    if (!neighbor_voxel || neighbor_voxel->weight <= 0) continue;

    float neighbor_tsdf = neighbor_voxel->distance;

    // Check for sign change (surface crossing)
    if ((center_tsdf * neighbor_tsdf) < 0) {
      return true;  // Surface crossing detected
    }
  }
  return false;  // No surface crossing detected
}

bool SurfaceEvaluator::interpolateTsdfDistanceAndWeight(const vxb::Layer<vxb::TsdfVoxel>* tsdf_layer,
  const vxb::GlobalIndex& voxel_idx, float& distance, float& weight) const {
  
  float total_distance = 0.0f;
  float total_weight = 0.0f;

  for (const auto& offset : neighbor_offsets_) {
    vxb::GlobalIndex neighbor_idx = voxel_idx + offset;

    const vxb::TsdfVoxel* neighbor_voxel = tsdf_layer->getVoxelPtrByGlobalIndex(neighbor_idx);
    if (neighbor_voxel && neighbor_voxel->weight > 0) {
      total_distance += neighbor_voxel->distance * neighbor_voxel->weight;
      total_weight += neighbor_voxel->weight;
    }
  }

  if (total_weight > 0) {
    distance = total_distance / total_weight;  // Weighted average
    weight = total_weight;
    return true;
  }

  // No valid neighbors
  distance = 0.0f;
  weight = 0.0f;
  return false;
}

bool SurfaceEvaluator::areNeighborsValid(
  const vxb::Layer<vxb::TsdfVoxel>* tsdf_layer, 
  const vxb::GlobalIndex& voxel_idx) const {
  
  for (const auto& offset : neighbor_offsets_) {
    vxb::GlobalIndex neighbor_idx = voxel_idx + offset;

    const vxb::TsdfVoxel* tsdf_voxel = tsdf_layer->getVoxelPtrByGlobalIndex(neighbor_idx);
    if (!tsdf_voxel || tsdf_voxel->weight <= 0) {
      return false;  // Invalid neighbor
    }
  }
  return true;  // All neighbors are valid
}

Eigen::Vector3f SurfaceEvaluator::computeCorrectedPosition(
  cconst std::shared_ptr<const vxb::Block<vxb::TsdfVoxel>>& block,
  const size_t & linear_idx) const {

  const vxb::TsdfVoxel& voxel = block->getVoxelByLinearIndex(linear_idx);
  const Eigen::Vector3f voxel_center = block->computeCoordinatesFromLinearIndex(linear_idx);

  if(voxel.weight <= 0) return voxel_center;

  const Eigen::Vector3i local_index = block->computeVoxelIndexFromLinearIndex(linear_idx);
  const size_t voxels_per_side = block->voxels_per_side();
  const float voxel_size = block->voxel_size();

  Eigen::Vector3f gradient = computeGradient(block, local_index);

  return voxel_center + voxel.distance * gradient;
}

Eigen::Vector3f SurfaceEvaluator::computeGradient(
  const vxb::Block<vxb::TsdfVoxel>::Ptr& block,
  const Eigen::Vector3i& local_index) const {
  
  auto getVoxelDistance = [&](int x, int y, int z) -> float {
      const size_t voxels_per_side = block->voxels_per_side();
      if (x >= 0 && x < static_cast<int>(voxels_per_side) &&
          y >= 0 && y < static_cast<int>(voxels_per_side) &&
          z >= 0 && z < static_cast<int>(voxels_per_side)) {
      const vxb::TsdfVoxel& neighbor_voxel =
          block->getVoxelByVoxelIndex(Eigen::Vector3i(x, y, z));
      return neighbor_voxel.distance;
      }
      return std::numeric_limits<float>::max();  // Default for out-of-bounds neighbors
  };

  float dx = (getVoxelDistance(local_index.x() + 1, local_index.y(), local_index.z()) -
              getVoxelDistance(local_index.x() - 1, local_index.y(), local_index.z())) /
              (2 * voxel_size_);
  float dy = (getVoxelDistance(local_index.x(), local_index.y() + 1, local_index.z()) -
              getVoxelDistance(local_index.x(), local_index.y() - 1, local_index.z())) /
              (2 * voxel_size_);
  float dz = (getVoxelDistance(local_index.x(), local_index.y(), local_index.z() + 1) -
              getVoxelDistance(local_index.x(), local_index.y(), local_index.z() - 1)) /
              (2 * voxel_size_);

  Eigen::Vector3f gradient(dx, dy, dz);
  float norm = gradient.norm();
  if (norm > 1e-6f) {
      gradient /= norm;
  } else {
      gradient.setZero();
  }

  return gradient;
}


