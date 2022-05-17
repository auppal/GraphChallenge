from distutils.core import setup, Extension
import numpy as np

# To build run: python setup.py build
#

module1 = Extension('entropy_module',
                    include_dirs=[np.get_include()],
                    sources=['entropy.c'],
                    extra_compile_args=['-O3', '-march=native'])

setup(name = 'PackageName',
      version = '1.0',
      description = 'This is the entropy functions helper package',
      ext_modules = [module1])
 
