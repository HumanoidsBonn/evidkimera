#pragma once

#include <Eigen/Core>
#include <cmath>
#include <memory>
#include <torch/torch.h>
#include <iostream>

namespace kimera {

/**
 * @brief  Templated base interface for kernel computations
 */
template <typename InputType>
class Kernel {
 public:
  virtual ~Kernel() = default;

  /**
   * @brief Computes the kernel matrix for the given query points.
   *
   * This function computes the kernel matrix using the provided query points.
   *
   * @param query_points A matrix where each row represents a query point in the feature space.
   * @param training_points A vector representing the training points in the feature space.
   * @return A matrix representing the computed kernel.
   */
  virtual InputType computeKernel(const InputType& query_points, const InputType& training_points) const = 0;
};

/**
 * @brief Sparse kernel with compact support.
 */
template <typename InputType>
class SparseKernel : public Kernel<InputType>{
 private:
  float ell_;  // Length scale parameter.
  float sf2_;  // Signal variance parameter.

 public:
  /**
   * @brief Constructor for SparseKernel.
   * @param ell Length scale parameter.
   * @param sf2 Signal variance parameter.
   */
  SparseKernel(float ell, float sf2) : ell_(ell), sf2_(sf2) {}

  /**
   * @brief Compute the kernel values between query points and a single training point.
   * @param query_points Matrix of query points (each row is a point).
   * @param training_points A single training point.
   * @return The kernel values as a column vector.
   */
  InputType computeKernel(const InputType& query_points, const InputType& training_points) const override {
    if constexpr (std::is_same_v<InputType, Eigen::MatrixXf>) {
      // Eigen implementation
      Eigen::MatrixXf dist_mat = (query_points.rowwise().replicate(training_points.rows()) -
                                  training_points.colwise().replicate(query_points.rows()))
                                     .rowwise()
                                     .norm();

      Eigen::MatrixXf kernel_values = ((2.0f + (dist_mat.array() * 2.0f * M_PI).cos()) *
                                           (1.0f - dist_mat.array()) / 3.0f +
                                       (dist_mat.array() * 2.0f * M_PI).sin() / (2.0f * M_PI))
                                          .matrix();
      return kernel_values;
    } else if constexpr (std::is_same_v<InputType, torch::Tensor>) {
      // Torch implementation
      torch::Tensor dist_mat = torch::cdist(query_points, training_points);
      torch::Tensor kernel_values = (2.0 + torch::cos(2.0 * M_PI * dist_mat)) * (1.0 - dist_mat) / 3.0 +
                                    torch::sin(2.0 * M_PI * dist_mat) / (2.0 * M_PI);
      return kernel_values;
    }
  }

  protected:
  /**
   * @brief Compute the distance matrix between query points and training points.
   * @param query_points Matrix of query points (each row is a point).
   * @param training_points Matrix of training points (each row is a point).
   * @return The distance matrix.
   */
  Eigen::MatrixXf computeDistanceMatrix(const Eigen::MatrixXf& query_points,
                                  const Eigen::MatrixXf& training_points) const {
    Eigen::MatrixXf dist_mat = Eigen::MatrixXf::Zero(query_points.rows(), training_points.rows());
    for (int i = 0; i < query_points.rows(); ++i) {
      for(int j = 0; j < training_points.rows(); ++j) {
        dist_mat(i,j) = (query_points.row(i) - training_points.row(j)).norm() / ell_;
      }
    }
    return dist_mat;
  }
};

}  // namespace kimera
