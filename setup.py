import os
from setuptools import setup



def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name = f"bitsandbytes-cuda{os.environ['CUDA_VERSION']}",
    version = "0.0.2",
    author = "Tim Dettmers",
    author_email = "tim.dettmers@gmail.com",
    description = ("Numpy-like library for GPUs."),
    license = "MIT",
    keywords = "gpu",
    url = "http://packages.python.org/bitsandbytes",
    packages=['bitsandbytes'],
    package_data={'': ['libClusterNet.so']},
    long_description=read('README.md'),
    classifiers=[
        "Development Status :: 1 - Planning",
        'Topic :: Scientific/Engineering :: Artificial Intelligence'
    ],
)

