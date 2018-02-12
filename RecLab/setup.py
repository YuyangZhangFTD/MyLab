
from distutils.core import setup

from distutils.extension import Extension

from Cython.Build import cythonize

 

setup(

  name = 'cd_fast',

  ext_modules=cythonize([

    Extension("cd_fast", ["cd_fast.pyx"]),

    ]), requires=['matplotlib']

)
