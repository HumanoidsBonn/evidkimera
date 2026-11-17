/**
 * @file   common.h
 * @brief  Common definitions
 * @author Antoni Rosinol
 * @co
 */

#pragma once

#include <voxblox/core/block_hash.h>
#include <voxblox/core/common.h>
#include <voxblox/utils/approx_hash_array.h>
// #include <torch/torch.h>
// #include <torch/script.h>

namespace kimera {

namespace vxb = voxblox;

typedef uint8_t SemanticLabel;
typedef int32_t Weight;
typedef vxb::AlignedVector<SemanticLabel> SemanticLabels;
// Consider id 0 to be the `unknown' label, for which we don't update the
// log-likelihood for that measurement.
// static constexpr size_t kLabelsInDataset = 23;
// static constexpr size_t kTotalNumberOfLabels = kLabelsInDataset + 1;
// static constexpr uint8_t kUnknownSemanticLabelId = 0u;
// static constexpr uint8_t kFreeSpaceSemanticLabelId = kTotalNumberOfLabels - 1u;
// The size of this array determines how many semantic labels SemanticVoxblox
// supports.
// TODO(Toni): parametrize this, although that means it becomes unknown at
// compile time...

typedef vxb::FloatingPoint SemanticProbability;
typedef vxb::FloatingPoint Evidence;
typedef vxb::FloatingPoint ConcentrationParameter;
typedef vxb::FloatingPoint Uncertainty;

// A `#Labels X #Labels` Eigen matrix where each `j` column represents the
// probability of observing label `j` when current label is `i`, where `i`
// is the row index of the matrix.
typedef Eigen::Matrix<SemanticProbability, kTotalNumberOf2DLabels, kTotalNumberOf2DLabels>
        SemanticLikelihoodFunction;

typedef Eigen::Matrix<SemanticProbability, kTotalNumberOf2DLabels, 1>
        SemanticProbabilities;

typedef vxb::LongIndexHashMapType<vxb::AlignedVector<size_t>>::type VoxelMap;
typedef VoxelMap::value_type VoxelMapElement;

// Add compatibility for c++11's lack of make_unique.
template <typename T, typename... Args>
std::unique_ptr<T> make_unique(Args&&... args) {
  return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
}

enum class SemanticRepresentationType {
  kOneHot,
  kProbability,
  kEvidential
};

enum class VoxelLevelFusionType {
  kNone,
  kSimple,
  kMeasurementUncertainty,
  kTsdfWeightedUncertainty,
  kVoxelMeasurementUncertainty,
  kDivergenceWeighted
};

enum class UncertaintyEstimatorType {
  kNone,
  kTotalUncertainty,
  kVariance,
  kDempsterShafer,
  kProbabilityVariance,
  kProbabilityEntropy
};

enum class VoxelUpdateState{
  UnInitialized,
  Measured,
  Updated
};

enum class KernelType {
  KSparse,
  kNeighbor,
  kRGBF
};

enum class KernelDecoratorType {
  kNone,
  kTSDF,
  kUncertainty,
  kTSDFAndUncertainty,
};

struct KernelConfig {
  bool use_cuda;
  float ell = 1.0;
  float sf2 = 1.0;
  float gamma = 1.0;
  KernelType kernel_type;
  KernelDecoratorType decorator_type;
};

struct BoundingBox {
  Eigen::Vector3f min;  ///< Minimum corner of the bounding box
  Eigen::Vector3f max;  ///< Maximum corner of the bounding box

  BoundingBox() 
      : min(Eigen::Vector3f::Constant(std::numeric_limits<float>::max())),
        max(Eigen::Vector3f::Constant(std::numeric_limits<float>::lowest())) {}

  BoundingBox(const Eigen::Vector3f& min, const Eigen::Vector3f& max)
      : min(min), max(max) {}

  /**
   * @brief Expand the bounding box by a specified margin.
   * @param margin The margin to expand in all directions.
   */
  void expand(float margin) {
    min -= Eigen::Vector3f::Constant(margin);
    max += Eigen::Vector3f::Constant(margin);
  }

  /**
   * @brief Check if a point is inside the bounding box.
   * @param point The point to check.
   * @return True if the point is inside, false otherwise.
   */
  bool contains(const Eigen::Vector3f& point) const {
    return (point.array() >= min.array()).all() &&
           (point.array() <= max.array()).all();
  }
};


struct SemanticPointData {
  vxb::GlobalIndex global_voxel_index;               // Voxel index
  Eigen::Vector3f point_G;              // Global position of the point
  vxb::SemanticState semantic_state; // Semantic measurement (e.g., Dirichlet parameters)
  float weight;                       // Weight of the point
};

enum class DataType {
  Index,
  Points,
  SemanticState,
  Weight
};

typedef std::vector<SemanticPointData> SemanticPointDataVector;

struct VectorSemanticPointData {
  VectorSemanticPointData() = default;
  VectorSemanticPointData(const size_t& vec_size): global_voxel_index_vector(vec_size), weight_vector(vec_size) {}
  std::vector<vxb::GlobalIndex> global_voxel_index_vector;
  vxb::AlignedVector<vxb::Point> point_G_vector;
  vxb::AlignedVector<vxb::SemanticState> semantic_state_vector;
  std::vector<float> weight_vector;

  size_t size() const { return global_voxel_index_vector.size(); }
  void reserve(const size_t& size) {
    global_voxel_index_vector.reserve(size);
    point_G_vector.reserve(size);
    semantic_state_vector.reserve(size);
    weight_vector.reserve(size);
  }
};

}  // namespace kimera
