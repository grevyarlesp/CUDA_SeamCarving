cmake_minimum_required (VERSION 3.8)
project(SeamCarving LANGUAGES CXX CUDA)

message("Computing architecture ${CC}")

set(CMAKE_EXPORT_COMPILE_COMMANDS YES)
set(CMAKE_INCLUDE_HEADERS_IN_COMPILE_COMMANDS YES)

set(EXECUTABLE_OUTPUT_PATH  ${PROJECT_SOURCE_DIR}/build/)

set(
    CUDA_NVCC_FLAGS
    ${CUDA_NVCC_FLAGS};
   arch=sm_72
    )


include_directories(
        ${PROJECT_SOURCE_DIR}/include
)

file(GLOB HEADERS
			"${PROJECT_SOURCE_DIR}/include"
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
set_property(TARGET SeamCarving PROPERTY CUDA_ARCHITECTURES ${CC})

add_executable(conv_test ${all_SRCS} ${PROJECT_SOURCE_DIR}/test/conv_tests.cu)
set_property(TARGET conv_test PROPERTY CUDA_ARCHITECTURES ${CC})
