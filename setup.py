from setuptools import find_packages, setup
from setuptools.dist import Distribution


# Tested with wheel v0.45.1
class BinaryDistribution(Distribution):
    def has_ext_modules(self):
        return True


setup(packages=find_packages(), distclass=BinaryDistribution)
