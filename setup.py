# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from distutils.errors import DistutilsModuleError
from warnings import warn

from setuptools import find_packages, setup
from setuptools.command.build_py import build_py
from setuptools.dist import Distribution


# Tested with wheel v0.29.0
class BinaryDistribution(Distribution):
    def has_ext_modules(self):
        return True


class ExtBuildPy(build_py):
    def run(self):
        # build_cmake needs to be called prior to build_py, as the latter
        # collects the files output into the package directory.
        try:
            self.run_command("build_cmake")
        except DistutilsModuleError:
            warn(
                "scikit-build-core not installed, CMake will not be invoked automatically. "
                "Please install scikit-build-core or run CMake manually to build extensions."
            )
        super().run()


setup(
    version="0.48.1.dev0",
    packages=find_packages(),
    distclass=BinaryDistribution,
    cmake_source_dir=".",
    cmdclass={
        "build_py": ExtBuildPy,
    },
)
