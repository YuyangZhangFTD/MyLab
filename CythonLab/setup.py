from distutils.core import setup, Extension
from Cython.Build import cythonize


# the samplest way to build
# setup(ext_modules=cythonize('fib.pyx'))

# another way
# First creat an Extension object with the appropriate name ande sources.
ext = Extension(name="wrap_fib", sources=["cfib.c","wrap_fib.pyx"])

# Use cythonize on the extension objection
setup(ext_modules=cythonize(ext))
