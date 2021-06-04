import os
from setuptools import setup

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name = "bitsandbytes",
    version = "0.0.1",
    author = "Tim Dettmers",
    author_email = "tim.dettmers@gmail.com",
    description = ("Numpy-like library for GPUs."),
    license = "MIT",
    keywords = "gpu",
    url = "http://packages.python.org/bitsandbytes",
    packages=['bitsandbytes'],
    long_description=read('README.md'),
    classifiers=[
        "Development Status :: 1 - Alpha",
        "Topic :: Bash",
    ],
)

