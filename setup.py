import glob
import os
import shutil
import subprocess
import sys
from pathlib import Path

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


class CMakeBuildExt(build_ext):
    def build_extension(self, ext: Extension) -> None:
        project_root = Path(__file__).parent.resolve()
        build_dir = project_root / "build"
        build_dir.mkdir(parents=True, exist_ok=True)

        pybind11_cmake_dir = subprocess.check_output(
            [sys.executable, "-m", "pybind11", "--cmakedir"],
            text=True,
        ).strip()

        subprocess.check_call(
            [
                "cmake", str(project_root),
                f"-DCMAKE_BUILD_TYPE=Release",
                f"-Dpybind11_DIR={pybind11_cmake_dir}",
            ],
            cwd=str(build_dir),
        )
        subprocess.check_call(
            ["cmake", "--build", str(build_dir),
             "--config", "Release",
             "--", f"-j{os.cpu_count() or 1}"],
            cwd=str(build_dir),
        )

        # CMake places the .so directly in waymo_coverage/; copy it to wherever
        # setuptools expects it (handles both regular and editable installs).
        built = glob.glob(
            str(project_root / "waymo_coverage" / "waymo_kinematics*.so")
        )
        if not built:
            raise RuntimeError("CMake did not produce waymo_kinematics*.so")

        dest = Path(self.get_ext_fullpath(ext.name))
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(built[0], dest)


setup(
    ext_modules=[Extension("waymo_coverage.waymo_kinematics", sources=[])],
    cmdclass={"build_ext": CMakeBuildExt},
)
