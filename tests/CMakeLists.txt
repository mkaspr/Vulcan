enable_testing()

# find packages

list(APPEND REQUIRED_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR})
list(APPEND REQUIRED_INCLUDE_DIRS ${CMAKE_CURRENT_BINARY_DIR})
list(APPEND REQUIRED_LIBRARIES vulcan)

find_package(GTest REQUIRED)
list(APPEND REQUIRED_INCLUDE_DIRS ${GTEST_INCLUDE_DIRS})
list(APPEND REQUIRED_LIBRARIES ${GTEST_LIBRARIES})
list(APPEND REQUIRED_LIBRARIES pthread)

# define c++ sources

set(CPP_SOURCES
  block_test.cpp
  buffer_test.cpp
  device_test.cpp
  exception_test.cpp
  exporter_test.cpp
  extractor_test.cpp
  frame_test.cpp
  hash_test.cpp
  image_test.cpp
  matrix_test.cpp
  mesh_test.cpp
  projection_test.cpp
  tracer_test.cpp
  tracker_test.cpp
  transform_test.cpp
  types_test.cpp
  volume_test.cpp
)

# define cuda sources

set(CUDA_SOURCES
  util_test.cu
  integrator_test.cu
)

# build executable

include_directories(${REQUIRED_INCLUDE_DIRS})
cuda_add_executable(tests tests.cpp ${CPP_SOURCES} ${CUDA_SOURCES})
target_link_libraries(tests ${REQUIRED_LIBRARIES})