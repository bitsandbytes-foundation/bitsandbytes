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
    version = "0.0.23",
    author = "Tim Dettmers",
    author_email = "tim.dettmers@gmail.com",
    description = ("Numpy-like library for GPUs."),
    license = "MIT",
    keywords = "gpu",
    url = "http://packages.python.org/bitsandbytes",
    packages=find_packages(),
    package_data={'': ['libbitsandbytes.so']},
    long_description=read('README.md'),
    long_description_content_type = 'text/markdown',
    classifiers=[
        "Development Status :: 1 - Planning",
        'Topic :: Scientific/Engineering :: Artificial Intelligence'
    ],
)

