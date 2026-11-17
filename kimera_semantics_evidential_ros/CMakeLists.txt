cmake_minimum_required(VERSION 3.8)
project(kimera_semantics_evidential_ros)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Release)
#set(CMAKE_PREFIX_PATH "/home/rohit/workspace/software/libtorch;/opt/ros/noetic")
#list(REMOVE_ITEM CMAKE_PREFIX_PATH "/home/rohit/workspace/ros1/kimera_ws/devel/.private/semantic_segmentation_ros")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
message(WARNING "CMAKE_PREFIX_PATH: ${CMAKE_PREFIX_PATH}")

find_package(catkin_simple REQUIRED)
#find_package(Torch REQUIRED)

include_directories(
 include
 #${TORCH_INCLUDE_DIRS}
 ${catkin_INCLUDE_DIRS}
 ${PCL_INCLUDE_DIRS}
)

catkin_simple()

cs_add_library(${PROJECT_NAME}
  include/kimera_semantics_evidential_ros/depth_map_to_pointcloud.h
  include/kimera_semantics_evidential_ros/ros_params.h
  include/kimera_semantics_evidential_ros/rosbag_data_provider.h
  include/kimera_semantics_evidential_ros/semantic_simulation_server.h
  include/kimera_semantics_evidential_ros/semantic_tsdf_server.h
  src/ros_params.cpp
  src/semantic_simulation_server.cpp
  src/semantic_tsdf_server.cpp
  src/rosbag_data_provider.cpp
  src/semantic_simulation_server.cpp
  src/semantic_tsdf_server.cpp
)

cs_add_executable(kimera_semantics_node
  src/kimera_semantics_node.cpp
)
target_link_libraries(kimera_semantics_node ${PROJECT_NAME})

cs_add_executable(kimera_semantics_rosbag
  src/kimera_semantics_rosbag.cpp
)
target_link_libraries(kimera_semantics_rosbag ${PROJECT_NAME})

cs_add_executable(semantic_simulator_eval
  src/semantic_simulation_eval.cpp
)
target_link_libraries(semantic_simulator_eval ${PROJECT_NAME} ${PCL_LIBRARIES})

cs_install()

cs_export()
