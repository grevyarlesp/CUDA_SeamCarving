cmake_minimum_required (VERSION 3.8)
project(SeamCarving LANGUAGES CXX CUDA)


if (NOT CUARCH)
  message("Warning: Computing architecture not set. Defaulting to 75")
  set(CUARCH 75)
endif()

message("Computing architecture ${CUARCH}")

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Debug")
endif()



set(CMAKE_EXPORT_COMPILE_COMMANDS YES)
set(CMAKE_INCLUDE_HEADERS_IN_COMPILE_COMMANDS YES)

set(EXECUTABLE_OUTPUT_PATH  ${PROJECT_SOURCE_DIR}/build/)
set(LIBRARY_OUTPUT_PATH  ${PROJECT_SOURCE_DIR}/build/)


# set(CMAKE_CUDA_ARCHITECTURES ${CUARCH})

# message("Compiling for architecture ${CMAKE_CUDA_ARCHITECTURES}")

include_directories(
        ${PROJECT_SOURCE_DIR}/include
)

# link_directories(
#       ${PROJECT_SOURCE_DIR}/src
#   )
#
file(GLOB all_SRCS
        "${PROJECT_SOURCE_DIR}/src/*.cu"
        )

list(REMOVE_ITEM all_SRCS ${PROJECT_SOURCE_DIR}/src/main.cu)

file(GLOB all_HDRS
        "${PROJECT_SOURCE_DIR}/include/*.h"
        "${PROJECT_SOURCE_DIR}/include/*.cuh"
        )

list(REMOVE_ITEM all_HDRS ${PROJECT_SOURCE_DIR}/src/stb_image.h)
list(REMOVE_ITEM all_HDRS ${PROJECT_SOURCE_DIR}/src/stb_image_write.h)

file(GLOB stbimageHDRS
        "${PROJECT_SOURCE_DIR}/include/stb_image*"
        )


set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_${CUARCH}")

add_library(stbimage STATIC ${stbimageHDRS})
set_target_properties(stbimage PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(stbimage PROPERTIES LINKER_LANGUAGE CUDA) 

 
file(GLOB test_SRCS
        "${PROJECT_SOURCE_DIR}/include/*.h"
        ${all_SRCS}
        )



set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${EXTRAFLAGS} ")


# Compiling libraries
add_library(libs STATIC ${all_SRCS} ${all_HDRS})
set_target_properties(libs PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_executable(SeamCarving ${PROJECT_SOURCE_DIR}/src/main.cu)
set_target_properties(SeamCarving PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(SeamCarving PRIVATE libs)

add_executable(conv_test ${PROJECT_SOURCE_DIR}/test/conv_tests.cu)
set_target_properties(conv_test PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(conv_test PRIVATE libs)

add_executable(seam_test ${PROJECT_SOURCE_DIR}/test/seam_test.cu)
set_target_properties(seam_test PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(seam_test PRIVATE libs)

add_executable(host_answer_gen ${PROJECT_SOURCE_DIR}/test/host_answer_gen.cu)
set_target_properties(host_answer_gen PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(host_answer_gen PRIVATE libs stbimage)

add_executable(gpu_v1_gen ${PROJECT_SOURCE_DIR}/test/gpu_v1_gen.cu)
set_target_properties(gpu_v1_gen PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(gpu_v1_gen PRIVATE libs stbimage)

add_executable(gpu_v2_gen ${PROJECT_SOURCE_DIR}/test/gpu_v2_gen.cu)
set_target_properties(gpu_v2_gen PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(gpu_v2_gen PRIVATE libs stbimage)
