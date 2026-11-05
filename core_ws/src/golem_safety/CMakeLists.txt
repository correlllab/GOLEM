cmake_minimum_required(VERSION 3.8)
project(golem_safety)

# C++17 for std::filesystem, std::optional and structured bindings.
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# yaml-cpp reads the robot profiles and configs; zlib gives the npz writer its crc32.
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(unitree_hg REQUIRED)
find_package(ament_index_cpp REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(ZLIB REQUIRED)
find_package(Threads REQUIRED)

# The safety layer without ROS executables: every source except the node's main(), linked by
# the node, the config dump tool and the tests.
add_library(golem_safety_core
  src/chunk_logger.cpp
  src/config.cpp
  src/json.cpp
  src/lowcmd.cpp
  src/robot_profile.cpp
  src/safety_core.cpp)
target_include_directories(golem_safety_core PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
ament_target_dependencies(golem_safety_core ament_index_cpp unitree_hg)
# zlib is only for crc32 in the npz writer; the archive is written stored.
target_link_libraries(golem_safety_core yaml-cpp ZLIB::ZLIB)

# golem_safety_dump_config: prints a config's derived limits and parsed values as JSON.
add_executable(golem_safety_dump_config src/dump_config_main.cpp)
target_link_libraries(golem_safety_dump_config golem_safety_core)
install(TARGETS golem_safety_dump_config DESTINATION lib/${PROJECT_NAME})

# safety_node, the only writer of /lowcmd; launch/safety.launch.py starts it.
add_executable(safety_node src/safety_node.cpp)
ament_target_dependencies(safety_node rclcpp std_msgs unitree_hg)
target_link_libraries(safety_node golem_safety_core Threads::Threads)
install(TARGETS safety_node DESTINATION lib/${PROJECT_NAME})

# robots/<robot>.yaml are the robot profiles, which golem_upper_ik reads too; config/<robot>/
# holds the safety configs a bare config name resolves against, and config/example.yaml
# documents their keys; launch/ starts the node.
install(DIRECTORY robots config launch DESTINATION share/${PROJECT_NAME})

# combine_chunks.py: concatenates a run's .npz chunks into one file.
install(PROGRAMS scripts/combine_chunks.py DESTINATION lib/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  find_package(ament_cmake_pytest REQUIRED)

  # The tests read the installed share directory, so they run after the
  # package is installed and its install space is sourced.

  # The profile parser, and the installed profiles through it.
  ament_add_gtest(test_robot_profile test/test_robot_profile.cpp)
  target_link_libraries(test_robot_profile golem_safety_core)

  # The installed profiles against the joint tables they were generated from.
  ament_add_pytest_test(test_robot_data test/test_robot_data.py TIMEOUT 60)

  # The config parser, the derived limits and every installed config.
  ament_add_gtest(test_config test/test_config.cpp)
  target_link_libraries(test_config golem_safety_core)

  # The CRC and the zero and estop commands, against the Unitree SDK's CRCs in
  # test/fixtures/lowcmd_crc.json (test/fixtures/README.md).
  ament_add_gtest(test_lowcmd test/test_lowcmd.cpp)
  target_link_libraries(test_lowcmd golem_safety_core)
  target_compile_definitions(test_lowcmd PRIVATE
    GOLEM_SAFETY_FIXTURE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/test/fixtures")

  # The chunk logger's files, threads and drops.
  ament_add_gtest(test_chunk_logger test/test_chunk_logger.cpp)
  target_link_libraries(test_chunk_logger golem_safety_core)

  # SafetyCore once relaying: merge, clip, latches, estop output, exit, logging, events.
  ament_add_gtest(test_safety_core_relay test/test_safety_core_relay.cpp)
  target_link_libraries(test_safety_core_relay golem_safety_core)

  # SafetyCore before and while engaging: the zero stream, engage conditions, relay_ready.
  ament_add_gtest(test_safety_core_startup test/test_safety_core_startup.cpp)
  target_link_libraries(test_safety_core_startup golem_safety_core)

  # Writes the chunks test_chunk_files.py reads back with numpy; not installed.
  add_executable(test_chunk_writer test/test_chunk_writer.cpp)
  target_link_libraries(test_chunk_writer golem_safety_core)
  # The installed combine_chunks.py is what the test runs, so it needs the install space.
  ament_add_pytest_test(test_chunk_files test/test_chunk_files.py TIMEOUT 120
    ENV GOLEM_SAFETY_TEST_CHUNK_WRITER=${CMAKE_CURRENT_BINARY_DIR}/test_chunk_writer)

  # The boot clock and its absolute sleep.
  ament_add_gtest(test_boot_clock test/test_boot_clock.cpp)
  target_include_directories(test_boot_clock PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

  # The Spinner's stop and failure paths; creates a node, on the package's test domain.
  ament_add_gtest(test_spinner test/test_spinner.cpp ENV ROS_DOMAIN_ID=91)
  target_include_directories(test_spinner PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
  ament_target_dependencies(test_spinner rclcpp)

  # No installed code imports the Unitree SDK.
  ament_add_pytest_test(test_no_sdk test/test_no_sdk.py TIMEOUT 60)

  # The matching and statistics of tools/relay_latency_probe.py, on synthetic streams; no ROS.
  ament_add_pytest_test(test_relay_latency_probe test/test_relay_latency_probe.py TIMEOUT 60)

  # Runs the installed safety_node against a fake robot over ROS, on the package's own
  # test domain.
  ament_add_pytest_test(test_node_ros test/test_node_ros.py TIMEOUT 1500
    ENV ROS_DOMAIN_ID=91)
endif()

ament_package()
