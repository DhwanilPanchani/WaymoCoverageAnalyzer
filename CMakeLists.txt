# WaymoCoverageAnalyzer — CMake build configuration
# Purpose: Build the C++ kinematics shared library (.so) + Google Test suite
# Author: <your-name>

cmake_minimum_required(VERSION 3.20)
project(WaymoCoverageAnalyzer
    VERSION 0.1.0
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ---------------------------------------------------------------------------
# Compiler flags
# ---------------------------------------------------------------------------
add_compile_options(-O3 -Wall -Wextra)

# ---------------------------------------------------------------------------
# Python — use FindPython3 (CMake 3.12+) before pybind11 to avoid the
# deprecated FindPythonInterp/FindPythonLibs fallback that picks the wrong
# interpreter when multiple Python versions are installed.
# ---------------------------------------------------------------------------
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
message(STATUS "Python3 executable : ${Python3_EXECUTABLE}")
message(STATUS "Python3 version    : ${Python3_VERSION}")

# ---------------------------------------------------------------------------
# pybind11 — locate via `python -m pybind11 --cmakedir`
# ---------------------------------------------------------------------------
execute_process(
    COMMAND "${Python3_EXECUTABLE}" -m pybind11 --cmakedir
    OUTPUT_VARIABLE PYBIND11_CMAKE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
message(STATUS "pybind11 CMake dir: ${PYBIND11_CMAKE_DIR}")
set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 REQUIRED HINTS "${PYBIND11_CMAKE_DIR}")

# ---------------------------------------------------------------------------
# Eigen3 — header-only linear algebra
# ---------------------------------------------------------------------------
find_package(Eigen3 REQUIRED NO_MODULE)

# ---------------------------------------------------------------------------
# Google Test — fetched at configure time
# ---------------------------------------------------------------------------
include(FetchContent)
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG        v1.14.0
)
# Prevent overriding parent project's compiler/linker settings on Windows
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# ---------------------------------------------------------------------------
# Core kinematics static library (shared between .so and gtest)
# ---------------------------------------------------------------------------
add_library(kinematics_core STATIC
    cpp/src/kinematics.cpp
)
target_include_directories(kinematics_core PUBLIC cpp/include)
target_link_libraries(kinematics_core PUBLIC Eigen3::Eigen)
set_target_properties(kinematics_core PROPERTIES
    CXX_STANDARD 20
    POSITION_INDEPENDENT_CODE ON
)

# ---------------------------------------------------------------------------
# pybind11 shared library — outputs directly into waymo_coverage/
# ---------------------------------------------------------------------------
pybind11_add_module(waymo_kinematics
    cpp/src/bindings.cpp
)
target_include_directories(waymo_kinematics PRIVATE cpp/include)
target_link_libraries(waymo_kinematics PRIVATE kinematics_core)

# Place the .so into the Python package directory for seamless import
set_target_properties(waymo_kinematics PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/waymo_coverage"
)

# ---------------------------------------------------------------------------
# Google Test executable
# ---------------------------------------------------------------------------
add_executable(kinematics_tests
    cpp/tests/test_kinematics.cpp
)
target_include_directories(kinematics_tests PRIVATE cpp/include)
target_link_libraries(kinematics_tests
    PRIVATE kinematics_core GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(kinematics_tests)
