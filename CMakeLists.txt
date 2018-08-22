cmake_minimum_required(VERSION 2.8)
project(Vulcan)

# find packages

list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)
list(APPEND REQUIRED_INCLUDE_DIRS ${CMAKE_SOURCE_DIR}/include)
list(APPEND REQUIRED_INCLUDE_DIRS ${CMAKE_BINARY_DIR}/include)

find_package(CUDA REQUIRED)
list(APPEND REQUIRED_INCLUDE_DIRS ${CUDA_INCLUDE_DIRS})
list(APPEND REQUIRED_LIBRARIES ${CUDA_LIBRARIES})

find_package(OpenCV REQUIRED)
list(APPEND REQUIRED_INCLUDE_DIRS ${OpenCV_INCLUDE_DIRS})
list(APPEND REQUIRED_LIBRARIES ${OpenCV_LIBRARIES})

find_package(Eigen3 REQUIRED)
list(APPEND REQUIRED_INCLUDE_DIRS ${EIGEN3_INCLUDE_DIRS})
list(APPEND REQUIRED_LIBRARIES ${EIGEN3_LIBRARIES})

find_package(GFlags REQUIRED)
list(APPEND REQUIRED_INCLUDE_DIRS ${GFlags_INCLUDE_DIRS})
list(APPEND REQUIRED_LIBRARIES ${GFlags_LIBRARIES})

find_package(GLog REQUIRED)
list(APPEND REQUIRED_INCLUDE_DIRS ${GLog_INCLUDE_DIRS})
list(APPEND REQUIRED_LIBRARIES ${GLog_LIBRARIES})

# set compiler flags

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -arch=compute_50")

# define c++ configs

set(CPP_CONFIG_DIR include/vulcan)

set(CPP_CONFIGS
  ${CPP_CONFIG_DIR}/vulcan.h
)

# define c++ headers

set(CPP_HEADER_DIR include/vulcan)

set(CPP_HEADERS
  ${CPP_HEADER_DIR}/block.h
  ${CPP_HEADER_DIR}/buffer.h
  ${CPP_HEADER_DIR}/color_tracker.h
  ${CPP_HEADER_DIR}/depth_tracker.h
  ${CPP_HEADER_DIR}/detector.h
  ${CPP_HEADER_DIR}/device.h
  ${CPP_HEADER_DIR}/exception.h
  ${CPP_HEADER_DIR}/exporter.h
  ${CPP_HEADER_DIR}/extractor.h
  ${CPP_HEADER_DIR}/frame.h
  ${CPP_HEADER_DIR}/hash.h
  ${CPP_HEADER_DIR}/image.h
  ${CPP_HEADER_DIR}/integrator.h
  ${CPP_HEADER_DIR}/math.h
  ${CPP_HEADER_DIR}/matrix.h
  ${CPP_HEADER_DIR}/mesh.h
  ${CPP_HEADER_DIR}/projection.h
  ${CPP_HEADER_DIR}/sampler.h
  ${CPP_HEADER_DIR}/tracer.h
  ${CPP_HEADER_DIR}/tracker.h
  ${CPP_HEADER_DIR}/transform.h
  ${CPP_HEADER_DIR}/types.h
  ${CPP_HEADER_DIR}/util.h
  ${CPP_HEADER_DIR}/volume.h
  ${CPP_HEADER_DIR}/voxel.h
)

# define c++ sources

set(CPP_SOURCE_DIR src)

set(CPP_SOURCES
  ${CPP_SOURCE_DIR}/color_tracker.cpp
  ${CPP_SOURCE_DIR}/depth_tracker.cpp
  ${CPP_SOURCE_DIR}/exporter.cpp
  ${CPP_SOURCE_DIR}/frame.cpp
  ${CPP_SOURCE_DIR}/mesh.cpp
  ${CPP_SOURCE_DIR}/tracer.cpp
  ${CPP_SOURCE_DIR}/tracker.cpp
)

# define cuda headers

set(CUDA_HEADER_DIR include/vulcan)

set(CUDA_HEADERS
  ${CUDA_HEADER_DIR}/frame.cuh
  ${CUDA_HEADER_DIR}/tracer.cuh
  ${CUDA_HEADER_DIR}/util.cuh
)

# define cuda sources

set(CUDA_SOURCE_DIR src)

set(CUDA_SOURCES
  ${CUDA_SOURCE_DIR}/color_tracker.cu
  ${CUDA_SOURCE_DIR}/depth_tracker.cu
  ${CUDA_SOURCE_DIR}/detector.cu
  ${CUDA_SOURCE_DIR}/extractor.cu
  ${CUDA_SOURCE_DIR}/frame.cu
  ${CUDA_SOURCE_DIR}/image.cu
  ${CUDA_SOURCE_DIR}/integrator.cu
  ${CUDA_SOURCE_DIR}/sampler.cu
  ${CUDA_SOURCE_DIR}/tracer.cu
  ${CUDA_SOURCE_DIR}/tracker.cu
  ${CUDA_SOURCE_DIR}/volume.cu
)

# build config files

foreach(CPP_HEADER ${CPP_HEADERS})
  string(REPLACE "include/" "" CPP_HEADER ${CPP_HEADER})
  set(VULCAN_INCLUDES "${VULCAN_INCLUDES}\n#include <${CPP_HEADER}>")
endforeach()

foreach(CPP_CONFIG ${CPP_CONFIGS})
  configure_file(${CPP_CONFIG}.in ${CPP_CONFIG} @ONLY)
endforeach()

# build library

include_directories(${REQUIRED_INCLUDE_DIRS})
cuda_add_library(vulcan SHARED ${CPP_SOURCES} ${CUDA_SOURCES})
target_link_libraries(vulcan ${REQUIRED_LIBRARIES})
cuda_add_cublas_to_target(vulcan)

# build applications

option(VULCAN_BUILD_APPS "Build applications" ON)

if(VULCAN_BUILD_APPS)
  add_subdirectory(apps)
endif()

# build unit tests

option(VULCAN_BUILD_TESTS "Build unit tests" OFF)

if(VULCAN_BUILD_TESTS)
  add_subdirectory(tests)
endif()

# export library

include(install_package)

install_package(
  PKG_NAME VULCAN
  LIB_NAME vulcan
  VERSION  1.0
  DESCRIPTION "Implementation of the SGM stereo matching algorithm"
  INSTALL_INCLUDE_DIR true
  DESTINATION ${CMAKE_INSTALL_PREFIX}
  INCLUDE_DIRS ${REQUIRED_INCLUDE_DIRS}
  LINK_LIBS ${REQUIRED_LIBRARIES}
)
