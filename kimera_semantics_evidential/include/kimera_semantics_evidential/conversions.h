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

/**
 * @file   conversions.h
 * @brief  Different inline conversion functions
 * @author Rohit Menon
 */
#include <Eigen/Dense>
#include <vector>
#include "kimera_semantics_evidential/common.h"
#include <Eigen/Core>
#include <vector>
#include <stdexcept>
namespace kimera
{
  // Convert std::vector<Eigen::Vector3f> to Eigen::MatrixXf
inline Eigen::MatrixXf fromVectorEigenToMatrixEigen(const std::vector<Eigen::Vector3f>& input) {
  size_t num_points = input.size();
  Eigen::MatrixXf output(num_points, 3);  // 3 columns for x, y, z
  for (size_t i = 0; i < num_points; ++i) {
    output.row(i) = input[i].transpose();  // Copy each Vector3f as a row
  }
  return output;
}

// inline Eigen::MatrixXf convertSemanticPointDataVectorToMatrix(
//   const SemanticPointDataVector& training_data,
//   const std::vector<size_t>& indices,
//   DataType output_type) {

//   std::vector<size_t> effective_indices;
//   if (indices.empty()) {
//     // Use all indices if the vector is empty
//     effective_indices.resize(data_vector.size());
//     std::iota(effective_indices.begin(), effective_indices.end(), 0);
//   } else {
//     effective_indices = indices; // Use the provided indices
//   }
  
//   size_t num_rows = effective_indices.size();
//   size_t num_columns = 0;
  
//   // Determine the number of columns based on the output type
//   if(output_type == DataType::Points)
//   {
//     num_columns = 3;
//   }
//   else if(output_type == DataType::SemanticState)
//   {
//     num_columns = data_vector[0].semantic_state.size();
//   }
//   else if(output_type == DataType::Weight)
//   {
//     num_columns = 1;
//   }
//   else
//   {
//     throw std::runtime_error("Invalid output type");
//   }

//   // Create the matrix
//   Eigen::MatrixXf matrix(num_rows, num_columns);

//   for (size_t i = 0; i < effective_indices.size(); ++i) {
//     const auto& data = training_data[effective_indices[i]];

//     if (output_type == DataType::Points) {
//       matrix.row(i) = data.point_G;  // Assign point
//     } else if (output_type == DataType::SemanticState) {
//       matrix.row(i) = data.semantic_state;  // Assign semantic measurement
//     }
//     else if (output_type == DataType::Weight) {
//       matrix.row(i) = data.weight;  // Assign weight
//     }
//   }

//   return matrix;
// }

// torch::Tensor convertSemanticPointDataVectorToTensor(
//   const SemanticPointDataVector& data_vector,
//   const std::vector<size_t>& indices,
//   DataType output_type) {
  
//   std::vector<size_t> effective_indices;
//   if (indices.empty()) {
//     // Use all indices if the vector is empty
//     effective_indices.resize(data_vector.size());
//     std::iota(effective_indices.begin(), effective_indices.end(), 0);
//   } else {
//     effective_indices = indices; // Use the provided indices
//   }

//   size_t num_points = effective_indices.size();
//   size_t num_columns = 0;
  
//   // Determine the number of columns based on the output type
//   if(output_type == DataType::Points) {
//     num_columns = 3;
//   }
//   else if(output_type == DataType::SemanticState) {
//     num_columns = data_vector[0].semantic_state.size();
//   }
//   else if(output_type == DataType::Weight)  {
//     num_columns = 1;
//   }
//   else {
//     throw std::runtime_error("Invalid output type");
//   }

//   // Allocate a tensor
//   torch::Tensor tensor = torch::empty({static_cast<long>(num_points), static_cast<long>(dim)});

//   for (size_t i = 0; i < num_points; ++i) {
//     size_t idx = effective_indices[i];
//     if (output_type == DataType::Points) {
//       auto row = tensor[i];
//       for (int j = 0; j < 3; ++j) {
//         row[j] = data_vector[idx].point_G[j];
//       }
//     } else if (output_type == DataType::SemanticState) {
//       auto row = tensor[i];
//       for (int j = 0; j < dim; ++j) {
//         row[j] = data_vector[idx].semantic_state[j];
//       }
//     }
//     else if (output_type == DataType::Weight) {
//       tensor[i] = data_vector[idx].weight;
//     }
//   }
//   return tensor.clone();
// }

template <typename T>
inline void appendVector(std::vector<T>& destination, std::vector<T>& source) {
  destination.insert(destination.end(), source.begin(), source.end());
}

template <typename T>
inline void appendVector(vxb::AlignedVector<T>& destination, vxb::AlignedVector<T>& source) {
  destination.insert(destination.end(), source.begin(), source.end());
}

inline std::vector<size_t> getEffectiveIndices(const size_t & data_size, const std::vector<size_t>& indices) {
  std::vector<size_t> effective_indices;
  if (indices.empty()) {
    // Use all indices if the vector is empty
    effective_indices.resize(data_size);
    std::iota(effective_indices.begin(), effective_indices.end(), 0);
  } else {
    effective_indices = indices; // Use the provided indices
  }
  return effective_indices;
}

inline Eigen::MatrixXf convertToEigen(const void* inputData, const std::vector<size_t>& indices, DataType type) {
  switch (type) {
    case DataType::Weight: {
      // Convert std::vector<float> to Eigen::MatrixXf
      const auto& data = *static_cast<const std::vector<float>*>(inputData);
      auto effective_indices = getEffectiveIndices(data.size(), indices);
      Eigen::MatrixXf matrix(effective_indices.size(), 1);
      for (size_t i = 0; i < effective_indices.size(); ++i) {
        matrix(i, 0) = data[i];
      }
      return matrix;
    }
    case DataType::Points: {
      // Convert AlignedVector<Point> to Eigen::MatrixXf
      const auto& data = *static_cast<const AlignedVector<vxb::Point>*>(inputData); 
      auto effective_indices = getEffectiveIndices(data.size(), indices);
      Eigen::MatrixXf matrix(effective_indices.size(), 1);
      for (size_t i = 0; i < effective_indices.size(); ++i) {
        matrix.row(i) = data[effective_indices[i]].transpose();
      }
      return matrix;
    }
    case DataType::SemanticState: {
      // Convert AlignedVector<SemanticState> to Eigen::MatrixXf
      const auto& data = *static_cast<const AlignedVector<SemanticState>*>(inputData);
      auto effective_indices = getEffectiveIndices(data.size(), indices);
      Eigen::MatrixXf matrix(effective_indices.size(), kTotalNumberOfLabels);
      for (size_t i = 0; i < effective_indices.size(); ++i) {
        matrix.row(i) = data[effective_indices[i]].transpose();
      }
      return matrix;
    }
    default:
      throw std::invalid_argument("Unsupported DataType");
  }
}

// Conversion function with indices to torch::Tensor
inline torch::Tensor convertToTorchTensor(const void* inputData, const std::vector<size_t>& indices, DataType type) {
  switch (type) {
    case DataType::Weight: {
      const auto& data = *static_cast<const std::vector<float>*>(inputData);
      auto effective_indices = getEffectiveIndices(data.size(), indices);

      std::vector<float> selected_data;
      selected_data.reserve(effective_indices.size());
      for (size_t idx : effective_indices) {
        selected_data.push_back(data[idx]);
      }

      return torch::from_blob(selected_data.data(), {static_cast<int64_t>(effective_indices.size()), 1}, torch::TensorOptions().dtype(torch::kFloat32)).clone();
    }
    case DataType::Points: {
      const auto& data = *static_cast<const AlignedVector<vxb::Point>*>(inputData);
      auto effective_indices = getEffectiveIndices(data.size(), indices);

      std::vector<float> selected_data;
      selected_data.reserve(effective_indices.size() * 3);
      for (size_t idx : effective_indices) {
        const auto& point = data[idx];
        selected_data.insert(selected_data.end(), {point[0], point[1], point[2]});
      }

      return torch::from_blob(selected_data.data(), {static_cast<int64_t>(effective_indices.size()), 3}, torch::TensorOptions().dtype(torch::kFloat32)).clone();
    }
    case DataType::SemanticState: {
      const auto& data = *static_cast<const AlignedVector<vxb::SemanticState>*>(inputData);
      auto effective_indices = getEffectiveIndices(data.size(), indices);

      std::vector<float> selected_data;
      selected_data.reserve(effective_indices.size() * kTotalNumberOfLabels);
      for (size_t idx : effective_indices) {
        const auto& state = data[idx];
        for (int i = 0; i < kTotalNumberOfLabels; ++i) {
          selected_data.push_back(state[i]);
        }
      }

      return torch::from_blob(selected_data.data(), {static_cast<int64_t>(effective_indices.size()), kTotalNumberOfLabels}, torch::TensorOptions().dtype(torch::kFloat32)).clone();
    }
    default:
      throw std::invalid_argument("Unsupported DataType");
  }
}

} // namespace kimera

