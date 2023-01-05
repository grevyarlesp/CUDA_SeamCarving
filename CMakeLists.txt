cmake_minimum_required (VERSION 3.8)
project(SeamCarving LANGUAGES CXX CUDA)

message("Computing architecture ${CC}")

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Debug")
endif()


set(CMAKE_EXPORT_COMPILE_COMMANDS YES)
set(CMAKE_INCLUDE_HEADERS_IN_COMPILE_COMMANDS YES)

set(EXECUTABLE_OUTPUT_PATH  ${PROJECT_SOURCE_DIR}/build/)

set(CMAKE_CUDA_FLAGS "-Wall -Wextra ")

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${EXTRAFLAGS} ")

set(CMAKE_CUDA_ARCHITECTURES ${CC})

message("Compiling for architecture ${CMAKE_CUDA_ARCHITECTURES}")

include_directories(
        ${PROJECT_SOURCE_DIR}/include
)

link_directories(
      ${PROJECT_SOURCE_DIR}/src
  )

file(GLOB all_SRCS
        "${PROJECT_SOURCE_DIR}/src/*.cu"
        )

list(REMOVE_ITEM all_SRCS ${PROJECT_SOURCE_DIR}/src/main.cu)
 
file(GLOB test_SRCS
        "${PROJECT_SOURCE_DIR}/include/*.h"
        ${all_SRCS}
        )


add_executable(SeamCarving ${all_SRCS} ${PROJECT_SOURCE_DIR}/src/main.cu)
set_target_properties(SeamCarving PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_executable(conv_test ${all_SRCS} ${PROJECT_SOURCE_DIR}/test/conv_tests.cu)
set_target_properties(conv_test PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_executable(seam_test ${all_SRCS} ${PROJECT_SOURCE_DIR}/test/seam_test.cu)
set_target_properties(seam_test PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_executable(host_answer_gen ${all_SRCS} ${PROJECT_SOURCE_DIR}/test/host_answer_gen.cu)
set_target_properties(host_answer_gen PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_executable(gpu_v1_gen ${all_SRCS} ${PROJECT_SOURCE_DIR}/test/gpu_v1_gen.cu)
set_target_properties(gpu_v1_gen PROPERTIES CUDA_SEPARABLE_COMPILATION ON)


