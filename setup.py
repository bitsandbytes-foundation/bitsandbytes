# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import glob
import os
import subprocess

from setuptools import find_packages, setup
from setuptools.dist import Distribution

libs = list(glob.glob("./bitsandbytes/libbitsandbytes*.*"))
libs = [os.path.basename(p) for p in libs]
print("libs:", libs)


def get_git_commit_hash():
    return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode("utf-8").strip()


def is_git_tagged_commit():
    tags = subprocess.check_output(["git", "tag", "--points-at", "HEAD"]).decode("utf-8").strip()
    return bool(tags)


def get_latest_semver_tag():
    tags = subprocess.check_output(["git", "tag"], text=True).splitlines()
    semver_tags = [tag for tag in tags if tag.count(".") == 2 and all(part.isdigit() for part in tag.split("."))]
    if not semver_tags:
        print("No valid semantic version tags found, use 1.0.0 defaultly")
        semver_tags = ["1.0.0"]
    return sorted(semver_tags, key=lambda s: list(map(int, s.split("."))))[-1]


def write_version_file(version, filepath="bitsandbytes/_version.py"):
    with open(filepath, "w") as f:
        f.write(f'__version__ = "{version}"\n')


def get_version_and_write_to_file():
    latest_semver_tag = get_latest_semver_tag()
    version = latest_semver_tag if is_git_tagged_commit() else f"{latest_semver_tag}.dev+{get_git_commit_hash()}"
    write_version_file(version)
    return version


# Tested with wheel v0.29.0
class BinaryDistribution(Distribution):
    def has_ext_modules(self):
        return True


setup(version=get_version_and_write_to_file(), packages=find_packages(), distclass=BinaryDistribution)
