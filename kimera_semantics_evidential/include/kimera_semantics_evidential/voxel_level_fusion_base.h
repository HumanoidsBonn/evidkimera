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

#include "kimera_semantics_evidential/semantic_voxel.h"

namespace kimera {

/**
 * @brief Base class for voxel-level fusion strategies.
 */
class VoxelLevelFusionBase {
 public:
  explicit VoxelLevelFusionBase(bool replace_on_first_measurement = false)
      : replace_on_first_measurement_(replace_on_first_measurement) {}
  
  virtual ~VoxelLevelFusionBase() = default;

  /**
   * @brief Apply the fusion strategy to update the semantic voxel.
   * @param voxel The voxel to update.
   * @param measurement The semantic state to integrate.
   * @param tsdf_weight The TSDF weight for the measurement.
   * @param measurement_uncertainty The uncertainty associated with the measurement.
   */
  virtual void fuse(SemanticVoxel* voxel,
                    const voxblox::SemanticState& measurement,
                    float tsdf_weight = 1.0f,
                    float measurement_uncertainty = 1.0f) const = 0;

  /**
   * @brief Determine whether the fusion strategy requires measurement uncertainty.
   * @return True if measurement uncertainty is required, false otherwise.
   */
  virtual bool requiresMeasurementUncertainty() const { return false; }

  protected:
  /**
   * @brief Transition the voxel's update state based on its current state and the replace behavior.
   */
  void transitionUpdateState(SemanticVoxel* voxel) const {
    if (voxel->update_state == VoxelUpdateState::UnInitialized) {
      // Transition to Measured or directly to Updated based on the replace flag
      voxel->update_state = replace_on_first_measurement_
                                ? VoxelUpdateState::Measured
                                : VoxelUpdateState::Updated;
    } else {
      // For subsequent updates, transition to Updated
      voxel->update_state = VoxelUpdateState::Updated;
    }
  }

  /**
   * @brief Handles the first measurement if replace_on_first_measurement is true.
   *        Otherwise, the first measurement is treated like a normal fuse.
   */
  void handleFirstMeasurement(SemanticVoxel* voxel,
                              const voxblox::SemanticState& measurement) const {
    if (voxel->update_state == VoxelUpdateState::UnInitialized) {
      if (replace_on_first_measurement_) {
        // Replace prior with the first measurement
        voxel->dirichlet_params = measurement;
        voxel->update_state = VoxelUpdateState::Measured;
      }
    }
  }

  protected:
    bool replace_on_first_measurement_ = false;
};

}  // namespace kimera
