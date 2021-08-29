cmake_minimum_required(VERSION 2.8.12)
project(pymnn)
option(BUILD_ON_PI OFF)

set (CMAKE_CXX_STANDARD 14)

set(LIB_DIR third_party/MNN/lib_x86)
IF(BUILD_ON_PI)
    message("build for pi")
    set(LIB_DIR third_party/MNN/lib_aarch64)
else()

ENDIF()
include_directories(
    third_party/MNN/includes
)

LINK_DIRECTORIES(
    ${LIB_DIR}
)
add_subdirectory(pybind11)
pybind11_add_module(pymnn src/pymnn.cpp)
target_link_libraries(pymnn PRIVATE MNN)