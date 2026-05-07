#!/usr/bin/env bash
# WaymoCoverageAnalyzer — one-command build script
# Purpose: Configure CMake, compile C++ shared library, install Python package in editable mode
# Author: <your-name>

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${PROJECT_ROOT}/build"

echo "=== WaymoCoverageAnalyzer build ==="
echo "Project root : ${PROJECT_ROOT}"
echo "Build dir    : ${BUILD_DIR}"

# Use the Python interpreter from the active environment.
PYTHON="${PYTHON:-$(which python3)}"
echo "Python       : ${PYTHON} ($(${PYTHON} --version))"

# Resolve pybind11 CMake directory from the active Python environment.
PYBIND11_CMAKE_DIR="$("${PYTHON}" -m pybind11 --cmakedir)"
echo "pybind11 dir : ${PYBIND11_CMAKE_DIR}"

# Locate Eigen3 (prefer Homebrew path on macOS).
EIGEN3_DIR=""
if [[ "$(uname)" == "Darwin" ]]; then
    EIGEN3_CANDIDATE="$(brew --prefix eigen 2>/dev/null)/share/eigen3/cmake"
    if [[ -d "${EIGEN3_CANDIDATE}" ]]; then
        EIGEN3_DIR="${EIGEN3_CANDIDATE}"
    fi
fi

# Build CMake arguments.
CMAKE_ARGS=(
    -S "${PROJECT_ROOT}"
    -B "${BUILD_DIR}"
    -DCMAKE_BUILD_TYPE=Release
    "-DPython3_EXECUTABLE=${PYTHON}"
)
if [[ -n "${EIGEN3_DIR}" ]]; then
    CMAKE_ARGS+=("-DEigen3_DIR=${EIGEN3_DIR}")
fi

# Configure CMake.
cmake "${CMAKE_ARGS[@]}"

# Compile with all available cores.
cmake --build "${BUILD_DIR}" --config Release -- -j"$(nproc 2>/dev/null || sysctl -n hw.logicalcpu)"

echo ""
echo "=== Running C++ unit tests ==="
"${BUILD_DIR}/kinematics_tests"

echo ""
echo "Build complete."
echo "Run tests   : ${PYTHON} -m pytest tests/ -v"
echo "Run pipeline: ${PYTHON} -m waymo_coverage.cli --help"
