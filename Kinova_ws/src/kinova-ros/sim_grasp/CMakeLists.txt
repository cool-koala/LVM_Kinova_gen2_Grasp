cmake_minimum_required(VERSION 3.0.2)
project(sim_grasp)

## Compile as C++11, supported in ROS Kinetic and newer
add_compile_options(-std=c++11)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED dynamic_reconfigure COMPONENTS
  actionlib
  actionlib_msgs
  geometry_msgs
  kinova_msgs
  roscpp
  rospy
  sensor_msgs
  std_msgs
  tf
  interactive_markers
  visualization_msgs
  message_generation
)
generate_dynamic_reconfigure_options(
  #cfg/KinovaPositionConfig.cfg
)

find_package(Eigen3 REQUIRED)


link_directories(${PROJECT_SOURCE_DIR}/lib/${CMAKE_LIBRARY_ARCHITECTURE})

add_message_files(
  FILES
  sim_graspModel.msg
)

generate_messages(
  DEPENDENCIES
  std_msgs
)


# export library to other packages
catkin_package(
  INCLUDE_DIRS include
  LIBRARIES kinova_driver
  CATKIN_DEPENDS 
  actionlib 
  actionlib_msgs 
  geometry_msgs 
  kinova_msgs 
  roscpp 
  rospy 
  sensor_msgs 
  std_msgs 
  tf 
  interactive_markers 
  visualization_msgs
  message_runtime
#  DEPENDS system_lib
)


###########
## Build ##
###########

include_directories(
  ../kinova_driver/include
  ${catkin_INCLUDE_DIRS}
)


## Declare a C++ library
add_executable(sim_arm_grasp_kinova src/sim_arm_grasp_kinova.cpp)
add_dependencies(sim_arm_grasp_kinova ${catkin_EXPORTED_TARGETS})
target_link_libraries(sim_arm_grasp_kinova ${catkin_LIBRARIES} kinova_driver)

add_executable(sim_arm_grasp_robotiq src/sim_arm_grasp_robotiq.cpp)
add_dependencies(sim_arm_grasp_robotiq ${catkin_EXPORTED_TARGETS})
target_link_libraries(sim_arm_grasp_robotiq ${catkin_LIBRARIES} kinova_driver)


#############
## Install ##
#############

## Mark executables and/or libraries for installation
install(TARGETS sim_arm_grasp_kinova sim_arm_grasp_robotiq
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

## Mark cpp header files for installation
install(DIRECTORY include/${PROJECT_NAME}/
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
  FILES_MATCHING PATTERN "*.h"
)
