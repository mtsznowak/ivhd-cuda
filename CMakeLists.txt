cmake_minimum_required(VERSION 2.8.4)

# Set up project
project(IVHD-CUDA)
set(EXECUTABLE_NAME "ivhd")

find_package(CUDA 8.0 REQUIRED)
find_package(Boost REQUIRED)

INCLUDE_DIRECTORIES( ${Boost_INCLUDE_DIR} )

include_directories(${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/third_party)

# Compilation flags
set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS};-std=c++11)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-arch sm_21;-std=c++11;-Wno-deprecated-gpu-targets;-maxrregcount=40)

# Recurse into source directory
add_subdirectory(src)
