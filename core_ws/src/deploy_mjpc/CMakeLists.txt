cmake_minimum_required(VERSION 3.20)

# The MJPC static libraries in the runtime build tree are clang-13 + LTO (LLVM
# bitcode archives), which GNU ld cannot read ("file format not recognized").
# When that build tree is present, compile and link this whole package with
# clang-13 to match. This must run before project() picks a compiler.
if(NOT DEFINED CMAKE_CXX_COMPILER AND EXISTS "/home/code/mujoco_mpc/build")
  find_program(_H12_CLANGXX clang++-13)
  find_program(_H12_CLANG clang-13)
  if(_H12_CLANGXX AND _H12_CLANG)
    set(CMAKE_C_COMPILER "${_H12_CLANG}")
    set(CMAKE_CXX_COMPILER "${_H12_CLANGXX}")
  endif()
endif()

project(deploy_mjpc)

if(NOT CMAKE_CXX_STANDARD)
  # C++20, as the fork's cmake/MpcOptions.cmake sets it: the cores compile the
  # fork's deploy_common.cc against the mjpc headers.
  set(CMAKE_CXX_STANDARD 20)
endif()
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra)
endif()

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)

# The Python package and its three rclpy parameter shells, each installed as a
# node executable. Each shell maps ROS parameters to a fork worker's flags and
# runs that worker unmodified. SAFETY: the C++ controller core runs in its own
# process (controller_launcher.py): rclcpp's rmw_cyclonedds and unitree_sdk2's
# bundled CycloneDDS both export libddsc/libddscxx.so.0, and one process
# loading both corrupts the heap.
ament_python_install_package(${PROJECT_NAME})
# estimator_node: the fork's RW-EKF base estimator.
install(PROGRAMS ${PROJECT_NAME}/estimator_node.py
        DESTINATION lib/${PROJECT_NAME}
        RENAME estimator_node)
# mjpc_deploy_lowerbody_controller: starts a controller core as a child process.
install(PROGRAMS ${PROJECT_NAME}/controller_launcher.py
        DESTINATION lib/${PROJECT_NAME}
        RENAME mjpc_deploy_lowerbody_controller)
# mjpc_debug_visualizer (debug only): renders the measured robot and a ghost of
# the planner's current best trajectory (rt/mjpc/plan) to an mp4.
install(PROGRAMS ${PROJECT_NAME}/debug_visualizer_node.py
        DESTINATION lib/${PROJECT_NAME}
        RENAME mjpc_debug_visualizer)

# The C++ controller cores link the fork's static libraries and headers straight
# from the MJPC build tree: docker-compose mounts container_cache/mjpc_build at
# /home/code/mujoco_mpc/build and launch_ros.sh hydrates it from the image's
# /opt/mjpc-build-seed before colcon runs, so MJPC_BUILD holds libmjpc,
# threadpool and absl archives and the fetched mujoco and absl headers under
# _deps. Without that tree (a dev checkout) the cores are skipped and the
# Python nodes still build.
# H12_BUILD_CONTROLLER: OFF skips the cores even where the build tree exists.
option(H12_BUILD_CONTROLLER "Build the C++ MJPC controller (needs the mjpc build tree)" ON)
# MJPC_ROOT: the mujoco_mpc fork's source; MJPC_BUILD: its CMake build tree.
set(MJPC_ROOT "/home/code/mujoco_mpc" CACHE PATH "mujoco_mpc fork root (bind-mounted source)")
set(MJPC_BUILD "${MJPC_ROOT}/build" CACHE PATH "MJPC CMake build tree (hydrated at runtime)")

# unitree_sdk2 C++, the raw-DDS channels the fork's deploy core uses.
# /opt/unitree_install is the container_cache mount launch_ros.sh hydrates with
# prebuilt libraries pinned to the fork's build.
list(APPEND CMAKE_PREFIX_PATH "/opt/unitree_install" "$ENV{HOME}/unitree_install")
find_package(unitree_sdk2 QUIET)

if(H12_BUILD_CONTROLLER AND EXISTS "${MJPC_BUILD}" AND unitree_sdk2_FOUND)
  find_package(Threads REQUIRED)

  # Include the mjpc source headers and the fetched mujoco and absl headers.
  # SAFETY: the cores must compile against the mujoco headers the build tree's
  # libmujoco.so and libmjpc.a were built with, _deps/mujoco-src. The pip
  # package reports the same release (3.2.3) but its mjModel/mjData layout
  # differs (extra flex_*/eq_objtype/nC fields before the actuator section), so
  # its headers skew every direct member access, actuator gains included. The
  # pip headers are a fallback only for a build tree without _deps.
  set(MJPC_INCLUDE_DIRS "${MJPC_ROOT}")
  file(GLOB _mj_dep "${MJPC_BUILD}/_deps/mujoco-src/include")
  if(_mj_dep)
    set(MUJOCO_INCLUDE_DIR ${_mj_dep})
  else()
    find_path(MUJOCO_INCLUDE_DIR mujoco/mujoco.h
      PATHS /usr/local/lib/python3.10/dist-packages/mujoco/include NO_DEFAULT_PATH)
  endif()
  file(GLOB _absl_src "${MJPC_BUILD}/_deps/abseil-cpp-src")
  file(GLOB _absl_bld "${MJPC_BUILD}/_deps/abseil-cpp-build")
  list(APPEND MJPC_INCLUDE_DIRS ${MUJOCO_INCLUDE_DIR} ${_absl_src} ${_absl_bld})

  # libmujoco.so: the image stages it in /usr/local/lib; the build tree has it too.
  find_library(MUJOCO_LIBRARY mujoco PATHS /usr/local/lib "${MJPC_BUILD}/lib")
  if(NOT MUJOCO_LIBRARY)
    message(FATAL_ERROR "deploy_mjpc: libmujoco not found "
                        "(set -DMUJOCO_LIBRARY=/path/to/libmujoco.so)")
  endif()

  # Link libmjpc, threadpool and abseil (the upstream deploy's set), found
  # anywhere under the build tree, in one --start-group so the cross-references
  # between them resolve. Only these archives, so lld's LTO codegen does not pull
  # in the grpc/protobuf/gtest bitcode the cores never reference.
  file(GLOB_RECURSE _mjpc_core "${MJPC_BUILD}/libmjpc.a" "${MJPC_BUILD}/libthreadpool.a")
  file(GLOB_RECURSE _mjpc_absl "${MJPC_BUILD}/libabsl_*.a")
  set(MJPC_LIBRARIES ${_mjpc_core} ${_mjpc_absl})

  # mjpc_lowerbody_core: the lower-body controller (legs only), the fork's own
  # thin main and shared deploy core compiled straight from the submodule, so
  # the fork stays the one source of its tables, flags, control loop, DDS I/O
  # and safety behaviour. It is not an rclcpp target (the CycloneDDS collision
  # above); mjpc_deploy_lowerbody_controller starts it.
  add_executable(mjpc_lowerbody_core
    "${MJPC_ROOT}/mjpc/deploy/h12_lower_body_controller.cc"
    "${MJPC_ROOT}/mjpc/deploy/deploy_common.cc")
  target_include_directories(mjpc_lowerbody_core PRIVATE ${MJPC_INCLUDE_DIRS})
  target_link_libraries(mjpc_lowerbody_core
    -Wl,--start-group ${MJPC_LIBRARIES} -Wl,--end-group
    "${MUJOCO_LIBRARY}" unitree_sdk2 Threads::Threads ${CMAKE_DL_LIBS})
  # lld (LTO-aware) is required to link the clang-13 bitcode archives above.
  if(EXISTS "/usr/bin/ld.lld-13")
    target_link_options(mjpc_lowerbody_core PRIVATE -fuse-ld=/usr/bin/ld.lld-13)
  endif()
  # unitree_sdk2's bundled CycloneDDS libraries sit next to the SDK; their
  # directory goes into the rpath so libddsc/libddscxx resolve to that build, as
  # in the fork's own deploy build.
  get_target_property(_ddsc_loc ddsc IMPORTED_LOCATION)
  get_filename_component(_dds_libdir "${_ddsc_loc}" DIRECTORY)
  set_target_properties(mjpc_lowerbody_core PROPERTIES
    BUILD_RPATH "/usr/local/lib;${MJPC_BUILD}/lib;${_dds_libdir}"
    INSTALL_RPATH "/usr/local/lib;${MJPC_BUILD}/lib;${_dds_libdir}")
  install(TARGETS mjpc_lowerbody_core DESTINATION lib/${PROJECT_NAME})

  # mjpc_fullbody_core: the whole-body controller (h12_control_node.cc, all 27
  # actuators), the same deploy core and link recipe with the other thin main,
  # so it runs through the same in-container DDS and estimator path. It
  # publishes golem_safety's full input, rt/safety/lowcmd_in, so it needs a
  # full-mode safety config.
  add_executable(mjpc_fullbody_core
    "${MJPC_ROOT}/mjpc/deploy/h12_control_node.cc"
    "${MJPC_ROOT}/mjpc/deploy/deploy_common.cc")
  target_include_directories(mjpc_fullbody_core PRIVATE ${MJPC_INCLUDE_DIRS})
  target_link_libraries(mjpc_fullbody_core
    -Wl,--start-group ${MJPC_LIBRARIES} -Wl,--end-group
    "${MUJOCO_LIBRARY}" unitree_sdk2 Threads::Threads ${CMAKE_DL_LIBS})
  if(EXISTS "/usr/bin/ld.lld-13")
    target_link_options(mjpc_fullbody_core PRIVATE -fuse-ld=/usr/bin/ld.lld-13)
  endif()
  set_target_properties(mjpc_fullbody_core PROPERTIES
    BUILD_RPATH "/usr/local/lib;${MJPC_BUILD}/lib;${_dds_libdir}"
    INSTALL_RPATH "/usr/local/lib;${MJPC_BUILD}/lib;${_dds_libdir}")
  install(TARGETS mjpc_fullbody_core DESTINATION lib/${PROJECT_NAME})

  message(STATUS "deploy_mjpc: mjpc_lowerbody_core + mjpc_fullbody_core ENABLED "
                 "(MJPC build tree ${MJPC_BUILD}; unitree_sdk2 dds at ${_dds_libdir})")
else()
  message(STATUS "deploy_mjpc: mjpc_lowerbody_core SKIPPED "
                 "(needs the MJPC build tree at '${MJPC_BUILD}' AND unitree_sdk2 "
                 "[found=${unitree_sdk2_FOUND}] -- run in the ros container after "
                 "launch_ros.sh hydrates both, or pass -DH12_BUILD_CONTROLLER=OFF). "
                 "Estimator + launcher still build.")
endif()

ament_package()
