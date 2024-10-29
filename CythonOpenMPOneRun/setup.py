from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np
import os
import sys

# Python paths
PYTHON_BASE = "/Library/Frameworks/Python.framework/Versions/3.10"
PYTHON_INCLUDE = f"{PYTHON_BASE}/include/python3.10"
PYTHON_LIB = f"{PYTHON_BASE}/lib"

# Set compiler
os.environ["CC"] = "/opt/homebrew/bin/gcc-14"
os.environ["CXX"] = "/opt/homebrew/bin/g++-14"
os.environ["LDSHARED"] = "/opt/homebrew/bin/gcc-14 -shared"  # Specify shared library build

# Compiler flags
extra_compile_args = [
    '-fopenmp',
    '-arch', 'arm64',
    '-I/opt/homebrew/opt/libomp/include',
    '-Wno-unused-result',
    '-dynamic',
    '-fPIC'  # Position Independent Code
]

# Linker flags
extra_link_args = [
    '-fopenmp',
    '-arch', 'arm64',
    '-shared',  # Specify shared library
    '-undefined', 'dynamic_lookup',  # Allow undefined symbols
    '-L/opt/homebrew/opt/libomp/lib',
    f'-L{PYTHON_LIB}',
    '-lpython3.10',
    f'-Wl,-rpath,{PYTHON_LIB}',
    '-Wl,-rpath,/opt/homebrew/opt/libomp/lib'
]

extensions = [
    Extension(
        "ll_parallel",
        ["ll_parallel.pyx"],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        include_dirs=[
            np.get_include(),
            PYTHON_INCLUDE
        ],
        library_dirs=[
            PYTHON_LIB,
            '/opt/homebrew/opt/libomp/lib'
        ]
    )
]

setup(
    ext_modules=cythonize(extensions, 
                         compiler_directives={
                             'language_level': "3",
                             'boundscheck': False,
                             'wraparound': False,
                             'initializedcheck': False,
                             'nonecheck': False,
                         },
                         annotate=True)
)