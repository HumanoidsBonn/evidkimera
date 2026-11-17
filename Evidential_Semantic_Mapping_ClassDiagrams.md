+-----------------------------------------+
|        SemanticIntegratorBase           |
+-----------------------------------------+
| - semantic_layer_: Layer<SemanticVoxel> |
| - fusion_strategy_: VoxelFusionStrategy |
| - representation_strategy_: SemanticRep |
| - uncertainty_strategy_: VoxelUncertainty|
+-----------------------------------------+
| + setFusionStrategy(...)                |
| + setRepresentationStrategy(...)        |
| + setUncertaintyStrategy(...)           |
| + updateSemanticVoxel(...)              |
+-----------------------------------------+
                        ^
                        |
  +---------------------+---------------------+
  |                                           |
+-------------------------+     +--------------------------+
| FastSemanticTsdfIntegrator |  | MergedSemanticTsdfIntegrator |
+-------------------------+     +--------------------------+
| Inherits: vxb::TsdfIntegratorBase           |
| + integrateSemanticFunction(...)           |
| + useKernelFusion(...)                     |
+-------------------------+     +--------------------------+
                        ^
                        |
           +-----------------------------+
           | KernelFusionStrategyBase    |
           +-----------------------------+
           | + applyKernel(...)          |
           | + setKernelStrategy(...)    |
           +-----------------------------+
                        ^
                        |
   +-----------------------------------------+
   |         Derived Kernel Strategies       |
   +-----------------------------------------+
   | FixedLengthBKI                          |
   | AdaptiveLengthBKI                       |
   | UncertaintyThresholdBKI                 |
   +-----------------------------------------+

+------------------------------------------+
|    VoxelLevelFusionStrategyBase          |
+------------------------------------------+
| + fuse(...)                              |
+------------------------------------------+
                        ^
                        |
+-----------------------------------------------------+
|                       Derived Classes              |
+--------------------------+--------------------------+
| SimpleFusionStrategy     | TsdfWeightedUncertaintyFusion |
| MeasurementUncertaintyFusion                          |
+--------------------------+--------------------------+

+------------------------------------------+
| SemanticRepresentationStrategyBase       |
+------------------------------------------+
| + validate(...)                          |
| + preprocess(...)                        |
| + postprocess(...)                       |
| + calculateMeasurementUncertainty(...)   |
+------------------------------------------+
                        ^
                        |
+------------------------------------------+
|      Derived Classes                     |
+------------------------------------------+
| OneHotRepresentationStrategy             |
| ProbabilityRepresentationStrategy        |
| EvidentialRepresentationStrategy         |
+------------------------------------------+

+------------------------------------------+
| VoxelUncertaintyStrategyBase             |
+------------------------------------------+
| + computeUncertainty(...)                |
+------------------------------------------+
                        ^
                        |
+------------------------------------------+
|      Derived Classes                     |
+------------------------------------------+
| TotalUncertaintyStrategy                 |
| VarianceUncertaintyStrategy              |
+------------------------------------------+