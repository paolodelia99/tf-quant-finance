# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Setup for pip package."""

import datetime
import os
import subprocess
import sys
from os import path
from pathlib import Path

from setuptools import find_packages
from setuptools import setup
from setuptools.dist import Distribution
from setuptools.command.build_py import build_py as _build_py

if sys.version_info[0] < 3:
  # Need to load open from io to support encoding arg when using Python 2.
  from io import open  # pylint: disable=redefined-builtin, g-importing-member, g-import-not-at-top

try:
    from contextlib import chdir  # Python 3.11+
except ImportError:
    # Fallback for Python <3.11
    from contextlib import contextmanager

    @contextmanager
    def chdir(path):
        prev_cwd = os.getcwd()
        os.chdir(path)
        try:
            yield
        finally:
            os.chdir(prev_cwd)

class CompileProtos(_build_py):
    """Custom build command to compile protocol buffers."""
    
    def run(self):
        # Compile proto files before building
        self.compile_protos()
        super().run()
    
    def compile_protos(self):
        """Compile all .proto files to Python."""
        proto_dir = Path("tf_quant_finance/experimental/pricing_platform/instrument_protos")
        if not os.path.exists(proto_dir):
            return
        
        with chdir(proto_dir):
            project_root = Path(".")
            proto_files = list(project_root.glob("*.proto"))
            
            if not proto_files:
                return
            
            print("\033[1m" + "* Compiling protocol buffers..." + "\033[0m")

            all_proto_files = [str(f) for f in proto_files]
        
            cmd = [
                "protoc",
                f"--python_out={project_root}",
                f"--proto_path={project_root}",
            ] + all_proto_files

            try:
                subprocess.run(cmd, check=True)
                print(f"✓ Compiled all protocol buffer files successfully")
            except subprocess.CalledProcessError as e:
                print(f"✗ Failed to compile protocol buffer files")
                print(f"  Error: {e.stderr}")

# Read the contents of the README file and set that as the long package
# description.
cwd = path.abspath(path.dirname(__file__))
with open(path.join(cwd, 'README.md'), encoding='utf-8') as f:
  long_description = f.read()

description = 'High-performance TensorFlow library for quantitative finance.'

major_version = '0'
minor_version = '0'
patch_version = '1'

if '--nightly' in sys.argv:
  # Run `python3 setup.py --nightly ...` to create a nightly build.
  sys.argv.remove('--nightly')
  project_name = 'tff_nightly'
  release_suffix = datetime.datetime.utcnow().strftime('.dev%Y%m%d')
else:
  project_name = 'tf-q-finance'
  # The suffix should be replaced with 'aN', 'bN', or 'rcN' (note: no dots) for
  # respective alpha releases, beta releases, and release candidates. And it
  # should be cleared, i.e. set to '', for stable releases (c.f. PEP 440).
  release_suffix = '.dev34'

__version__ = '.'.join([major_version, minor_version, patch_version])
if release_suffix:
  __version__ += release_suffix


def find_packages_excluding_tests():
    """Find all packages excluding test files."""
    packages = find_packages(exclude=["*_test.py"])
    return [pkg for pkg in packages if not pkg.endswith('_test.py')]


class BinaryDistribution(Distribution):
  """This class is needed in order to create OS specific wheels."""

  def has_ext_modules(self):
    return False


setup(
    name=project_name,
    version=__version__,
    description=description,
    author='Google Inc.',
    author_email='paolo.delia99@gmail.com',
    url='https://github.com/paolodelia99/tf-quant-finance',
    # Contained modules and scripts.
    packages=find_packages_excluding_tests(),
    # Add in any packaged data.
    include_package_data=True,
    zip_safe=False,
    distclass=BinaryDistribution,
    cmdclass={'build_py': CompileProtos},
    # PyPI package information.
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Financial and Insurance Industry',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Natural Language :: English',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Topic :: Software Development :: Libraries',
        'Operating System :: OS Independent',
    ],
    keywords='tensorflow quantitative finance hpc gpu option pricing',
    package_data={
        'tf_quant_finance': [
            'third_party/sobol_data/new-joe-kuo-6.21201',
            'third_party/sobol_data/LICENSE',
            'experimental/pricing_platform/instrument_protos/*.proto',
        ]
    },
    long_description=long_description,
    long_description_content_type='text/markdown'
)
