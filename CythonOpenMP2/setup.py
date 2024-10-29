from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np
import platform
import os

# Determine the paths based on Homebrew installation
if platform.system() == "Darwin":
    # Adjust paths for Apple Silicon or Intel Macs
    if platform.machine() == "x86_64":
        llvm_path = "/usr/local/opt/llvm"
    else:
        llvm_path = "/opt/homebrew/opt/llvm"

    extra_compile_args = ['-fopenmp']
    extra_link_args = ['-fopenmp']
    os.environ["CC"] = os.path.join(llvm_path, "bin/clang")
    os.environ["CXX"] = os.path.join(llvm_path, "bin/clang++")
    include_dirs = [np.get_include(), os.path.join(llvm_path, "include")]
    library_dirs = [os.path.join(llvm_path, "lib")]
else:
    extra_compile_args = ['-fopenmp']
    extra_link_args = ['-fopenmp']
    include_dirs = [np.get_include()]
    library_dirs = []

extensions = [
    Extension(
        "monte_carlo",
        ["monte_carlo.pyx"],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        include_dirs=include_dirs,
        library_dirs=library_dirs,
        language='c++'  # Use 'c++' if needed
    )
]

setup(
    ext_modules=cythonize(extensions, compiler_directives={'language_level': "3"})
)
