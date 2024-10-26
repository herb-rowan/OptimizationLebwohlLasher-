from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np
import sys
import os

# For macOS, we need to specify the OpenMP include and library paths
if sys.platform == 'darwin':
    os.environ["CC"] = "/usr/bin/clang"
    os.environ["CXX"] = "/usr/bin/clang++"
    
    # Assuming homebrew installation path, adjust if different
    BREW_PREFIX = os.popen("brew --prefix").read().strip()
    
    extra_compile_args = [
        '-Xpreprocessor', 
        '-fopenmp', 
        f'-I{BREW_PREFIX}/opt/libomp/include'
    ]
    extra_link_args = [
        f'-L{BREW_PREFIX}/opt/libomp/lib',
        '-lomp'
    ]
else:  # Linux and Windows
    extra_compile_args = ['-fopenmp']
    extra_link_args = ['-fopenmp']

ext_modules = [
    Extension(
        "LebwohlLasher_openmp",
        ["LebwohlLasher_openmp.pyx"],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        include_dirs=[np.get_include()]
    )
]

setup(
    ext_modules=cythonize(ext_modules),
)