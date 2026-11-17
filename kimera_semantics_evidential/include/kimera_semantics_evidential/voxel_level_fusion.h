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

#include "kimera_semantics_evidential/voxel_level_fusion_base.h"

namespace kimera {

/**
 * @brief Skips fusion and leaves the voxel unchanged.
 */
class NoneFusion : public VoxelLevelFusionBase {
public:
  explicit NoneFusion(bool replace_on_first_measurement = true)
      : VoxelLevelFusionBase(replace_on_first_measurement) {}
  void fuse(SemanticVoxel* voxel,
            const voxblox::SemanticState& measurement,
            float weight = 1.0f,
            float measurement_uncertainty = 1.0f) const override {
    CHECK(voxel != nullptr) << "Voxel pointer is null.";
    
    // Do nothing
  }
};

/**
 * @brief Directly integrates the measurement without weighting.
 */
class SimpleFusion : public VoxelLevelFusionBase {
 public:
 explicit SimpleFusion(bool replace_on_first_measurement = false)
      : VoxelLevelFusionBase(replace_on_first_measurement) {}
  void fuse(SemanticVoxel* voxel,
            const voxblox::SemanticState& measurement,
            float weight = 1.0f,
            float measurement_uncertainty = 1.0f) const override {
    CHECK(voxel != nullptr) << "Voxel pointer is null.";
    CHECK(measurement.size() == voxel->dirichlet_params.size())
        << "State size mismatch between measurement and voxel.";

    handleFirstMeasurement(voxel, measurement);
    voxel->dirichlet_params += measurement;
    transitionUpdateState(voxel);
  }

};

/**
 * @brief Scales the measurement by the TSDF weight before integrating.
 */
class TsdfWeightFusion : public VoxelLevelFusionBase {
 public:
 explicit TsdfWeightFusion(bool replace_on_first_measurement = false)
      : VoxelLevelFusionBase(replace_on_first_measurement) {}
      
  void fuse(SemanticVoxel* voxel,
            const voxblox::SemanticState& measurement,
            float weight,
            float measurement_uncertainty = 1.0f) const override {
    CHECK(voxel != nullptr) << "Voxel pointer is null.";
    CHECK(measurement.size() == voxel->dirichlet_params.size())
        << "State size mismatch between measurement and voxel.";

    handleFirstMeasurement(voxel, measurement);
    voxel->dirichlet_params += measurement * weight;
    transitionUpdateState(voxel);
  }
};

/**
 * @brief Scales the measurement by its uncertainty before integrating.
 */
class MeasurementUncertaintyFusion : public VoxelLevelFusionBase {
 public:
 explicit MeasurementUncertaintyFusion(bool replace_on_first_measurement = false)
      : VoxelLevelFusionBase(replace_on_first_measurement) {}
      
  bool requiresMeasurementUncertainty() const override { return true; }

  void fuse(SemanticVoxel* voxel,
            const voxblox::SemanticState& measurement,
            float weight = 1.0f,
            float measurement_uncertainty = 1.0f) const override {
    CHECK(voxel != nullptr) << "Voxel pointer is null.";
    CHECK(measurement.size() == voxel->dirichlet_params.size())
        << "State size mismatch between measurement and voxel.";

    handleFirstMeasurement(voxel, measurement);
    voxel->dirichlet_params += measurement * (1-measurement_uncertainty);
    transitionUpdateState(voxel);
  }
};

/**
 * @brief Combines TSDF weight and measurement uncertainty for scaling.
 */
class WeightAndUncertaintyFusion : public VoxelLevelFusionBase {
 public:
 explicit WeightAndUncertaintyFusion(bool replace_on_first_measurement = false)
      : VoxelLevelFusionBase(replace_on_first_measurement) {}
      
  bool requiresMeasurementUncertainty() const override { return true; }

  void fuse(SemanticVoxel* voxel,
            const voxblox::SemanticState& measurement,
            float weight,
            float measurement_uncertainty) const override {
    CHECK(voxel != nullptr) << "Voxel pointer is null.";
    CHECK(measurement.size() == voxel->dirichlet_params.size())
        << "State size mismatch between measurement and voxel.";

    handleFirstMeasurement(voxel, measurement);
    voxel->dirichlet_params += measurement * weight * (1-measurement_uncertainty);
    transitionUpdateState(voxel);
  }
};

class VoxelMeasurementUncertaintyFusion : public VoxelLevelFusionBase {
 public:
 explicit VoxelMeasurementUncertaintyFusion(bool replace_on_first_measurement = false)
      : VoxelLevelFusionBase(replace_on_first_measurement) {}
      
  void fuse(SemanticVoxel* voxel,
            const voxblox::SemanticState& measurement,
            float weight = 1.0f,
            float measurement_uncertainty = 1.0f) const override {
    CHECK(voxel != nullptr) << "Voxel pointer is null.";
    CHECK(measurement.size() == voxel->dirichlet_params.size())
        << "State size mismatch between measurement and voxel.";

    handleFirstMeasurement(voxel, measurement);

    // Compute fusion weights
    float voxel_weight = (1.0f - voxel->uncertainty);
    float measurement_weight = 1.0f - measurement_uncertainty;
    // Fuse prior and measurement
    voxel->dirichlet_params =
        voxel_weight * voxel->dirichlet_params + measurement_weight * measurement;
    // Increment voxel uncertainty weight
    

    transitionUpdateState(voxel);
  }
};

class TsdfWeightedUncertaintyFusion : public VoxelLevelFusionBase {
 public:
 explicit TsdfWeightedUncertaintyFusion(bool replace_on_first_measurement = false)
      : VoxelLevelFusionBase(replace_on_first_measurement) {}
      
  void fuse(SemanticVoxel* voxel,
            const voxblox::SemanticState& measurement,
            float tsdf_weight,
            float measurement_uncertainty = 1.0f) const override {
    CHECK(voxel != nullptr) << "Voxel pointer is null.";
    CHECK(measurement.size() == voxel->dirichlet_params.size())
        << "State size mismatch between measurement and voxel.";

    handleFirstMeasurement(voxel, measurement);

    // Compute fusion weights
    float voxel_weight =  1.0f;
    float measurement_weight = tsdf_weight * (1.0f - measurement_uncertainty);
    // Fuse prior and measurement
    voxel->dirichlet_params =
        voxel_weight * voxel->dirichlet_params + measurement_weight * measurement;
    
    transitionUpdateState(voxel);
  }
};

class MeasurementOnlyFusion : public VoxelLevelFusionBase {
 public:
 explicit MeasurementOnlyFusion(bool replace_on_first_measurement = true)
      : VoxelLevelFusionBase(replace_on_first_measurement) {}
      
  void fuse(SemanticVoxel* voxel,
            const voxblox::SemanticState& measurement,
            float tsdf_weight,
            float measurement_uncertainty = 1.0f) const override {
    CHECK(voxel != nullptr) << "Voxel pointer is null.";
    CHECK(measurement.size() == voxel->dirichlet_params.size())
        << "State size mismatch between measurement and voxel.";

    handleFirstMeasurement(voxel, measurement);

    if(voxel->update_state != VoxelUpdateState::UnInitialized){
     LOG(INFO)<<"Skipping measurement as voxel is already initialised with first measurement and the strategy is MeasurementOnlyFusion";
    }
    transitionUpdateState(voxel);
  }
};

}  // namespace kimera
