#pragma once

#include <memory>
#include <string>
#include "kimera_semantics_evidential/common.h"
#include "kimera_semantics_evidential/conversions.h"
#include "kimera_semantics_evidential/kernels/kernels.h"
#include "kimera_semantics_evidential/kernels/kernel_decorators.h"

// Kernel configuration structure
namespace kimera {

template <typename InputType>
std::shared_ptr<KernelDecorator<InputType>> createDecoratedKernel(
  const KernelConfig& config,
  const VectorSemanticPointData& training_data_vector,
  const std::vector<size_t>& training_points_indices,
  const InputType& tsdf_weights) {
  // Extract the kernel type and decorator type from the configuration  
  //using InputType = std::conditional_t<std::is_same_v<InputType, torch::Tensor>, torch::Tensor, Eigen::MatrixXf>;

  // Create the base kernel
  std::shared_ptr<Kernel<InputType>> base_kernel;
  switch (config.kernel_type) {
    case KernelType::KSparse:
      base_kernel = std::make_shared<SparseKernel<InputType>>(config.ell, config.sf2);  // Example parameters
      break;
    default:
      LOG(FATAL) << "Unknown kernel type!";
      return nullptr;
  }

  // Prepare weights and uncertainties based on the decorator type and execution mode
  std::shared_ptr<KernelDecorator<InputType>> decorated_kernel;
  switch (config.decorator_type) {
    case KernelDecoratorType::kNone:
      decorated_kernel = std::make_shared<KernelDecorator<InputType>>(base_kernel);
      break;
    case KernelDecoratorType::kTSDF: {
      decorated_kernel = std::make_shared<TSDFWeightedKernelDecorator<InputType>>(base_kernel, tsdf_weights);
      break;
    }
    case KernelDecoratorType::kUncertainty: {
      decorated_kernel = std::make_shared<UncertaintyWeightedKernelDecorator<InputType>>(base_kernel);
      break;
    }
    case KernelDecoratorType::kTSDFAndUncertainty: {
      decorated_kernel = std::make_shared<TSDFAndUncertaintyWeightedKernelDecorator<InputType>>(base_kernel, tsdf_weights);
      break;
    }
    default:
      LOG(FATAL) << "Unknown decorator type!";
      return nullptr;
  }
  return decorated_kernel;
}

}  // namespace kimera