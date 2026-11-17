#pragma once

#include <memory>
#include <Eigen/Core>
#include "kimera_semantics_evidential/kernels/kernels.h"  // Include base kernel definitions

namespace kimera {

/**
 * @brief Base class for kernel decorators that extend kernel functionality.
 */
template <typename InputType>
class KernelDecorator : public Kernel<InputType> {
 protected:
  std::shared_ptr<Kernel<InputType>> base_kernel_;  ///< Wrapped base kernel

 public:
  explicit KernelDecorator(std::shared_ptr<Kernel<InputType>> base_kernel)
      : base_kernel_(std::move(base_kernel)) {}

  InputType computeKernel(const InputType& query_points, const InputType& training_points) const override {
    return base_kernel_->computeKernel(query_points, training_points);
  }
};


/**
 * @brief TSDF-weighted kernel decorator for handling query and training points as Eigen::MatrixXf.
 */
template <typename InputType>
class TSDFWeightedKernelDecorator : public KernelDecorator<InputType> {
 private:
  InputType tsdf_weights_;  ///< Element-wise weights

 public:
  TSDFWeightedKernelDecorator(std::shared_ptr<Kernel<InputType>> base_kernel, const InputType& tsdf_weights)
      : KernelDecorator<InputType>(std::move(base_kernel)), tsdf_weights_(tsdf_weights) {}

  InputType computeKernel(const InputType& query_points, const InputType& training_points) const override {
    InputType base_kernel_values = this->base_kernel_->computeKernel(query_points, training_points);

    if constexpr (std::is_same_v<InputType, Eigen::MatrixXf>) {
      // Element-wise scaling for Eigen
      return base_kernel_values.array() * tsdf_weights_.array();
    } else if constexpr (std::is_same_v<InputType, torch::Tensor>) {
      // Element-wise scaling for Torch
      return base_kernel_values * tsdf_weights_;
    }
  }
};

/**
 * @brief Uncertainty-weighted kernel decorator for handling query and training points as Eigen::MatrixXf.
 */
template <typename InputType>
class UncertaintyWeightedKernelDecorator : public KernelDecorator<InputType> {
 private:
  InputType uncertainty_weights_;  ///< Precomputed uncertainty weights

 public:
  UncertaintyWeightedKernelDecorator(std::shared_ptr<Kernel<InputType>> base_kernel, const InputType& uncertainty_weights)
      : KernelDecorator<InputType>(std::move(base_kernel)), uncertainty_weights_(uncertainty_weights) {}

  UncertaintyWeightedKernelDecorator(std::shared_ptr<Kernel<InputType>> base_kernel)
      : KernelDecorator<InputType>(std::move(base_kernel)){}

  InputType computeKernel(const InputType& query_points, const InputType& training_points) const override {
    InputType base_kernel_values = this->base_kernel_->computeKernel(query_points, training_points);

    if constexpr (std::is_same_v<InputType, Eigen::MatrixXf>) {
      Eigen::MatrixXf uncertainty_factors = computeUncertaintyFactors();
      return base_kernel_values.array() * uncertainty_factors.array();
    } else if constexpr (std::is_same_v<InputType, torch::Tensor>) {
      torch::Tensor uncertainty_factors = computeUncertaintyFactors();
      return base_kernel_values * uncertainty_factors;
    }
  }

 private:
  InputType computeUncertaintyFactors() const {
    const float a = 1.0f;
    const float b = -3.0f;

    if constexpr (std::is_same_v<InputType, Eigen::MatrixXf>) {
      return (a + b * uncertainty_weights_.array()).exp().matrix();
    } else if constexpr (std::is_same_v<InputType, torch::Tensor>) {
      return torch::exp(a + b * uncertainty_weights_);
    }
  }
};

/**
 * @brief Combined TSDF and uncertainty-weighted kernel decorator using composition.
 */
template <typename InputType>
class TSDFAndUncertaintyWeightedKernelDecorator : public KernelDecorator<InputType> {
 public:
  /**
   * @brief Constructor for the combined decorator.
   * @param base_kernel The base kernel to decorate.
   * @param tsdf_values Vector of TSDF weights for the training points.
   * @param uncertainties Vector of uncertainty weights for the training points.
   */
  TSDFAndUncertaintyWeightedKernelDecorator(
    std::shared_ptr<Kernel<InputType>> base_kernel,
    const InputType& tsdf_weights,
    const InputType& uncertainty_weights)
  : KernelDecorator<InputType>(std::move(base_kernel))
  {
    tsdf_weighted_decorator_ = std::make_shared<TSDFWeightedKernelDecorator<InputType>>(this->base_kernel_, tsdf_weights);
    uncertainty_weighted_decorator_ = std::make_shared<UncertaintyWeightedKernelDecorator<InputType>>(tsdf_weighted_decorator_, uncertainty_weights);
  }

  TSDFAndUncertaintyWeightedKernelDecorator(
    std::shared_ptr<Kernel<InputType>> base_kernel,
    const InputType& tsdf_weights)
  : KernelDecorator<InputType>(std::move(base_kernel))
  {
    tsdf_weighted_decorator_ = std::make_shared<TSDFWeightedKernelDecorator<InputType>>(this->base_kernel_, tsdf_weights);
    uncertainty_weighted_decorator_ = std::make_shared<UncertaintyWeightedKernelDecorator<InputType>>(tsdf_weighted_decorator_);
  }

  /**
   * @brief Compute the TSDF and uncertainty-weighted kernel values.
   * @param query_points Matrix of query points (each row is a point).
   * @param training_points Matrix of training points (each row is a point).
   * @return A matrix of kernel values with combined TSDF and uncertainty weighting applied.
   */
  InputType computeKernel(const InputType& query_points, const InputType& training_points) const override {
    InputType tsdf_weighted = tsdf_weighted_decorator_->computeKernel(query_points, training_points);
    return uncertainty_weighted_decorator_->computeKernel(tsdf_weighted, training_points);
  }

 private:
  std::shared_ptr<Kernel<InputType>> tsdf_weighted_decorator_;
  std::shared_ptr<Kernel<InputType>> uncertainty_weighted_decorator_;
};
}  // namespace kimera
