from distutils.core import setup, Extension
import numpy as np

# To build run: python setup.py build
#

module1 = Extension('entropy_module',
                    include_dirs=[np.get_include()],
                    sources=['entropy.c'],
                    extra_compile_args=['-O3', '-march=native'])

# To debug: undef_macros = [ "NDEBUG" ]
module2 = Extension('compressed_array',
                    include_dirs=[np.get_include()],
                    sources=['compressed_array.c'],
                    extra_compile_args=['-O3', '-march=native'],
                    libraries = ['rt'])

setup(name = 'PackageName',
      version = '1.0',
      description = 'These are the helper modules.',
      ext_modules = [module1, module2])
 
