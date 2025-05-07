# compile_cython_functions.py
from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

# Ensure you have NumPy version that supports API version 2.0 for compilation
# For example, in your compilation environment: pip install "numpy>=2.0.0"

extensions = [
    Extension(
        "tribal_masking_computational_core",
        ["tribal_masking_computational_core.pyx"],
        include_dirs=[np.get_include()], # Provides NumPy C header files
        # For NumPy 2.x compatibility, Cython >= 3.0.0 is recommended.
        # Cython usually handles NPY_NO_DEPRECATED_API for you with recent versions.
        # If you encounter deprecation warnings during compilation, you might add:
        # define_macros=[('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')]
        # However, with NumPy 2.0, it's more about compiling with NumPy 2.0 headers.
    )
]

setup(
    name='tribal_masking_computational_core', # It's good practice to name your package
    ext_modules=cythonize(
        extensions,
        annotate=True, # Generates an HTML file to see Cython C interactions
        compiler_directives={'language_level': "3"} # Or "2" if you're on Python 2
    ),
    # If this were a distributable package, you'd also include:
    # setup_requires=['numpy>=2.0.0', 'cython>=3.0.0'],
    # install_requires=['numpy>=2.0.0']
)