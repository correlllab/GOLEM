cmake_minimum_required(VERSION 3.8)
project(golem_safety)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(unitree_hg REQUIRED)
find_package(ament_index_cpp REQUIRED)
find_package(yaml-cpp REQUIRED)

# Every source except the node's main(); the node and the tests link it.
add_library(golem_safety_core
  src/config.cpp
  src/robot_profile.cpp)
target_include_directories(golem_safety_core PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
ament_target_dependencies(golem_safety_core ament_index_cpp)
target_link_libraries(golem_safety_core yaml-cpp)

# Prints a config's derived limits and parsed values as JSON.
add_executable(golem_safety_dump_config src/dump_config_main.cpp)
target_link_libraries(golem_safety_dump_config golem_safety_core)
install(TARGETS golem_safety_dump_config DESTINATION lib/${PROJECT_NAME})

# robots/<robot>.yaml is the robot data golem_upper_ik reads too; config/<robot>/ holds
# the safety configs, which a bare config name resolves against.
install(DIRECTORY robots config DESTINATION share/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  find_package(ament_cmake_pytest REQUIRED)

  # The tests read the installed share directory, so they run after the
  # package is installed and its install space is sourced.
  ament_add_gtest(test_robot_profile test/test_robot_profile.cpp)
  target_link_libraries(test_robot_profile golem_safety_core)

  ament_add_pytest_test(test_robot_data test/test_robot_data.py TIMEOUT 60)

  ament_add_gtest(test_config test/test_config.cpp)
  target_link_libraries(test_config golem_safety_core)

  # Runs golem_safety_dump_config, so it needs the installed executable and configs.
  ament_add_pytest_test(test_limit_parity test/test_limit_parity.py TIMEOUT 120)
endif()

ament_package()
