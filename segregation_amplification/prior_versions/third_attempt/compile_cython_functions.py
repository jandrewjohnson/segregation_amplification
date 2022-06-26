from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy

ext_modules = [Extension('tribal_masking_computational_core',
                         ['tribal_masking_computational_core.pyx'],
                         )]
returned = setup(
    name='tribal_masking_computational_core',
    include_dirs=[numpy.get_include()],
    cmdclass={'build_ext': build_ext},
    ext_modules=ext_modules
)
