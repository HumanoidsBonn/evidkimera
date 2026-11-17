#pragma once

#include "kimera_semantics_evidential/kernels/kernel_factory.h"

namespace kimera {
class SparseKernelWrapper {
 private:
  std::shared_ptr<void> kernel_;  // Holds Eigen or Torch kernel
  bool use_cuda_;

 public:
  explicit SparseKernelWrapper(const KernelConfig& config) : use_cuda_(config.use_cuda) {
    if (use_cuda_) {
      kernel_ = createDecoratedKernel<torch::Tensor>(config);
    } else {
      kernel_ = createDecoratedKernel<Eigen::MatrixXf>(config);
    }
  }

  void computeKernel(const Eigen::MatrixXf& query_points, const Eigen::MatrixXf& training_points) const {
    if (!use_cuda_) {
      auto kernel = std::static_pointer_cast<Kernel<Eigen::MatrixXf>>(kernel_);
      kernel->computeKernel(query_points, training_points);
    } else {
      throw std::logic_error("Eigen computation requested, but CUDA is enabled.");
    }
  }

  void computeKernel(const torch::Tensor& query_points, const torch::Tensor& training_points) const {
    if (use_cuda_) {
      auto kernel = std::static_pointer_cast<Kernel<torch::Tensor>>(kernel_);
      kernel->computeKernel(query_points, training_points);
    } else {
      throw std::logic_error("Torch computation requested, but CUDA is disabled.");
    }
  }
};

}  // namespace kimera
