# WaymoCoverageAnalyzer — setuptools build script with CMake integration
# Purpose: Invoke CMake to compile the pybind11 extension during `pip install -e .`
# Author: <your-name>

import os
import subprocess
import sys
from pathlib import Path

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


class CMakeBuildExt(build_ext):
    """Custom build_ext that delegates to CMake instead of the normal C compiler."""

    def build_extension(self, ext: Extension) -> None:
        project_root = Path(__file__).parent.resolve()
        build_dir = project_root / "build"
        build_dir.mkdir(parents=True, exist_ok=True)

        pybind11_cmake_dir = subprocess.check_output(
            [sys.executable, "-m", "pybind11", "--cmakedir"],
            text=True,
        ).strip()

        cmake_args = [
            f"-DCMAKE_BUILD_TYPE=Release",
            f"-Dpybind11_DIR={pybind11_cmake_dir}",
        ]

        build_args = [
            "--config", "Release",
            "--",
            f"-j{os.cpu_count() or 1}",
        ]

        subprocess.check_call(
            ["cmake", str(project_root), *cmake_args],
            cwd=str(build_dir),
        )
        subprocess.check_call(
            ["cmake", "--build", str(build_dir), *build_args],
            cwd=str(build_dir),
        )


setup(
    ext_modules=[Extension("waymo_kinematics", sources=[])],
    cmdclass={"build_ext": CMakeBuildExt},
)
