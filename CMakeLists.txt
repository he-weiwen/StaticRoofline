cmake_minimum_required(VERSION 3.20)
project(NVPTXArithIntensity CXX)
set(CMAKE_CXX_STANDARD 17)

find_package(LLVM REQUIRED CONFIG)
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
include(AddLLVM)

include_directories(${LLVM_INCLUDE_DIRS})
add_definitions(${LLVM_DEFINITIONS})

add_subdirectory(lib)
add_subdirectory(test)
