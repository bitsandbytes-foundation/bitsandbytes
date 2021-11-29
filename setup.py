# Copyright (c) Facebook, Inc. and its affiliates. 
#   
# This source code is licensed under the MIT license found in the 
# LICENSE file in the root directory of this source tree.
import os
from setuptools import setup, find_packages



def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name = f"bitsandbytes-cuda{os.environ['CUDA_VERSION']}",
    version = "0.26.0",
    author = "Tim Dettmers",
    author_email = "dettmers@cs.washington.edu",
    description = ("8-bit optimizers and quantization routines."),
    license = "MIT",
    keywords = "gpu optimizers optimization 8-bit quantization compression",
    url = "http://packages.python.org/bitsandbytes",
    packages=find_packages(),
    package_data={'': ['libbitsandbytes.so']},
    long_description=read('README.md'),
    long_description_content_type = 'text/markdown',
    classifiers=[
        "Development Status :: 4 - Beta",
        'Topic :: Scientific/Engineering :: Artificial Intelligence'
    ],
)

