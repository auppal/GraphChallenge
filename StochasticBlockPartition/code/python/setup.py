from distutils.core import setup, Extension
import numpy as np
import os

# To build run: python setup.py build
# To build with clang: CC="clang" python setup.py build
#

module1 = Extension('entropy_module',
                    include_dirs=[np.get_include()],
                    sources=['entropy.c'],
                    extra_compile_args=['-O3', '-march=native', '-ffast-math'])

# To debug: undef_macros = [ "NDEBUG" ]

compressed_array_libs = ['rt']
if os.getenv("CC") != "clang":
    compressed_array_libs += ['atomic']

module2 = Extension('compressed_array',
                    include_dirs=[np.get_include()],
                    sources=['compressed_array.c', 'shared_mem.c'],
                    extra_compile_args=['-g', '-Og', '-march=native', '-ffast-math'],
                    libraries = compressed_array_libs,
                    library_dirs=['.'],
                    undef_macros = [ "NDEBUG" ]
)

module3 = Extension('shared_mem',
                    sources=['shared_mem.c'],
                    extra_compile_args=['-O3', '-march=native'])

setup(name = 'PackageName',
      version = '1.0',
      description = 'These are the helper modules.',
      ext_modules = [module1, module2])
 
