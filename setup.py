from setuptools import find_packages, setup
from setuptools.dist import Distribution

VERSION = "1.0.0"


# Tested with wheel v0.45.1
class BinaryDistribution(Distribution):
    def has_ext_modules(self):
        return True


def write_version_file(version, filepath="bitsandbytes/_version.py"):
    with open(filepath, "w") as f:
        f.write(f'__version__ = "{version}"\n')
    return version


setup(packages=find_packages(), distclass=BinaryDistribution, version=write_version_file(VERSION))
