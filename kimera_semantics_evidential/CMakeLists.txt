cmake_minimum_required(VERSION 2.8.3)
project(kimera_semantics_evidential)

add_definitions(-std=c++17)

find_package(catkin_simple REQUIRED)

#set(CMAKE_PREFIX_PATH "/home/rohit/workspace/software/libtorch;/opt/ros/noetic")
#list(REMOVE_ITEM CMAKE_PREFIX_PATH "/home/rohit/workspace/ros1/kimera_ws/devel/.private/semantic_segmentation_ros")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
#message(WARNING "CMAKE_PREFIX_PATH: ${CMAKE_PREFIX_PATH}")

#find_package(Torch REQUIRED)

include_directories(
 include
 #${TORCH_INCLUDE_DIRS}
 ${catkin_INCLUDE_DIRS}
 ${PCL_INCLUDE_DIRS}
)
catkin_simple()

cs_add_library(${PROJECT_NAME}
  include/kimera_semantics_evidential/color.h
  include/kimera_semantics_evidential/common.h
  include/kimera_semantics_evidential/csv_iterator.h
  include/kimera_semantics_evidential/semantic_voxel.h
  #include/kimera_semantics_evidential/kernels/kernels.h
  #include/kimera_semantics_evidential/kernels/kernel_decorators.h
  #include/kimera_semantics_evidential/kernels/kernel_factory.h
  #include/kimera_semantics_evidential/surface_evaluator.h
  include/kimera_semantics_evidential/semantic_integrator_base.h
  include/kimera_semantics_evidential/semantic_tsdf_integrator_factory.h
  include/kimera_semantics_evidential/semantic_tsdf_integrator_fast.h
  include/kimera_semantics_evidential/semantic_tsdf_integrator_merged.h
  include/kimera_semantics_evidential/semantic_evidential_tsdf_integrator_fast.h
  src/color.cpp
  src/csv_iterator.cpp
  src/semantic_integrator_base.cpp
  src/semantic_tsdf_integrator_factory.cpp
  src/semantic_tsdf_integrator_fast.cpp
  src/semantic_tsdf_integrator_merged.cpp
  src/semantic_evidential_tsdf_integrator_fast.cpp
  src/simulation/semantic_simulation_world.cpp
)

target_link_libraries(${PROJECT_NAME})

cs_install()

cs_export()
