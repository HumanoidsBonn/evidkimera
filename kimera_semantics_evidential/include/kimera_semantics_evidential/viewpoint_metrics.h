#include <Eigen/Dense>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <limits>
#include <voxblox/core/common.h>

namespace kimera
{
  class ViewpointObliquenessEstimator {
  public:
    ViewpointObliquenessEstimator(double angle_thresh=40.0*M_PI/180.0, double epsilon=0.707)
        : angle_thresh_(angle_thresh), epsilon_(epsilon) {}

    // double computeWeightingFactor(const Eigen::Vector3f& viewpoint_dir, const Eigen::Vector3f& sensor_dir) const {
    //   // Normalize input directions
    //   Eigen::Vector3f vp_dir_norm = viewpoint_dir.normalized();
    //   Eigen::Vector3f sensor_dir_norm = sensor_dir.normalized();

    //   // Compute the angle between viewpoint and sensor direction
    //   double cos_theta = std::clamp(vp_dir_norm.dot(sensor_dir_norm), -1.0, 1.0);
    //   double theta = std::acos(cos_theta);  // Angle in radians

    //   // Compute weighting factor: lower for oblique viewpoints
    //   double weight = std::max(epsilon_, 1.0 - (theta / angle_thresh_));

    //   return weight;
    // }

  private:
    double angle_thresh_;  // Maximum allowable obliqueness (radians)
    double epsilon_;       // Minimum weight value to prevent complete elimination
  };
 


class ViewpointSimilarityEstimator {
 public:
 ViewpointSimilarityEstimator(size_t buffer_size=2000, float distance_thresh=0.1, float angle_thresh=15*M_PI/180.0, float epsilon=0.707)
      : buffer_size_(buffer_size),
        distance_thresh_(distance_thresh),
        angle_thresh_(angle_thresh),
        epsilon_(epsilon),
        latest_idx_(0),
        filled_(false) {
    origins_ = Eigen::MatrixXf::Zero(buffer_size_, 3);
    directions_ = Eigen::MatrixXf::Zero(buffer_size_, 3);
  }

  void addViewpoint(const voxblox::Transformation& T_G_C) {
    vxb::Point origin = T_G_C.getPosition();  // Extract sensor origin
    vxb::Ray direction = T_G_C.getRotationMatrix()* Ray(0, 0, 1);  // Z-axis in global frame
    Eigen::Vector3f new_origin_f(origin(0), origin(1), origin(2));
    Eigen::Vector3f new_dir_f(direction(0), direction(1), direction(2));

    origins_.row(latest_idx_) = new_origin_f.transpose();  // Store as 1x3 row
    directions_.row(latest_idx_) = new_dir_f.normalized().transpose();  // Store as 1x3 row

    latest_idx_ = (latest_idx_ + 1) % buffer_size_;  // Circular buffer update
    if (latest_idx_ == 0) filled_ = true;  // Buffer is now full
  }

  float computeSimilarity(const Transformation& T_G_C) const {
    if (!filled_ && latest_idx_ == 0) return 1.0;  // No stored viewpoints yet

    vxb::Point new_origin = T_G_C.getPosition();  // Sensor origin in global frame
    vxb::Ray new_dir = T_G_C.getRotationMatrix() * vxb::Ray(0, 0, 1);  // Sensor direction in global frame
    new_dir.normalize();

    Eigen::Vector3f new_origin_f(new_origin(0), new_origin(1), new_origin(2));
    Eigen::Vector3f new_dir_f(new_dir(0), new_dir(1), new_dir(2));

    size_t count = filled_ ? buffer_size_ : latest_idx_;

    // Extract valid rows
    
    Eigen::MatrixXf origins_valid = origins_.topRows(count);
    Eigen::MatrixXf directions_valid = directions_.topRows(count);


    // Compute Euclidean distance (vectorized)
    Eigen::VectorXf dists = (origins_valid.rowwise() - new_origin_f.transpose()).rowwise().norm();

    // Compute angle differences (vectorized dot product)
    Eigen::VectorXf cos_thetas = directions_valid * new_dir_f;
    cos_thetas = cos_thetas.array().cwiseMax(-1.0).cwiseMin(1.0);  // Clamp to avoid NaN
    Eigen::VectorXf angles = cos_thetas.array().acos();  // Angle in radians

    // // Threshold check
    // for (int i = 0; i < count; ++i) {
    //   if (dists[i] > distance_thresh_ || angles[i] > angle_thresh_) {
    //     return 1.0;  // Highly dissimilar viewpoint
    //   }
    // }

    // Compute viewpoint dissimilarity index (VpD)
    
    Eigen::VectorXf vpds = (1.0 - cos_thetas.array()) * (dists.array() / distance_thresh_);
    //LOG(INFO) << "vpds: " << vpds;
    return std::max(vpds.minCoeff(), epsilon_);
  }

 private:
  size_t buffer_size_;
  float distance_thresh_;
  float angle_thresh_;
  float epsilon_;
  size_t latest_idx_;
  bool filled_;
  
  Eigen::MatrixXf origins_;   // Nx3 matrix storing viewpoint origins
  Eigen::MatrixXf directions_;  // Nx3 matrix storing viewpoint directions
};


} // namespace kimera