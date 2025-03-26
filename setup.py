# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import os
import sys
import subprocess
from setuptools import setup, find_packages


def command_exists(cmd):
    if sys.platform == "win32":
        result = subprocess.Popen(f'{cmd}', stdout=subprocess.PIPE, shell=True)
        return result.wait() == 1
    else:
        result = subprocess.Popen(f'type {cmd}', stdout=subprocess.PIPE, shell=True)
        return result.wait() == 0


# Write out version/git info
git_hash_cmd = "git rev-parse --short HEAD"
git_branch_cmd = "git rev-parse --abbrev-ref HEAD"
if command_exists('git') and 'DS_BUILD_STRING' not in os.environ:
    try:
        result = subprocess.check_output(git_hash_cmd, shell=True)
        git_hash = result.decode('utf-8').strip()
        result = subprocess.check_output(git_branch_cmd, shell=True)
        git_branch = result.decode('utf-8').strip()
    except subprocess.CalledProcessError:
        git_hash = "unknown"
        git_branch = "unknown"
else:
    git_hash = "unknown"
    git_branch = "unknown"

# Parse the MII version string from version.txt
version_str = open('version.txt', 'r').read().strip()

# Build specifiers like .devX can be added at install time. Otherwise, add the git hash.
# example: MII_BUILD_STR=".dev20201022" python -m build --sdist --wheel

MII_BUILD_STRING = 'MII_BUILD_STRING'
BUILD_FILE = 'build.txt'
mii_build_string = os.environ.get(MII_BUILD_STRING)

# Building wheel for distribution, update version file
if mii_build_string:
    # Build string env specified, probably building for distribution
    with open(BUILD_FILE, 'w') as fd:
        fd.write(mii_build_string)
    version_str += mii_build_string
elif os.path.isfile(BUILD_FILE):
    # build.txt exists, probably installing from distribution
    with open(BUILD_FILE, 'r') as fd:
        version_str += fd.read().strip()
else:
    # None of the above, probably installing from source
    version_str += f'+{git_hash}'

# write out installed version
with open("mii/version.py", 'w') as fd:
    fd.write(f"__version__ = '{version_str}'\n")

print("PACKAGES", find_packages())
setup(name="deepspeed-mii",
      version=version_str,
      packages=find_packages(exclude=("tests",
                                      )),
      classifiers=[
          'Programming Language :: Python :: 3.8',
          'Programming Language :: Python :: 3.9',
          'Programming Language :: Python :: 3.10',
          'Programming Language :: Python :: 3.11',
          'Programming Language :: Python :: 3.12'
      ])
