from distutils.core import Extension, setup
from Cython.Build import cythonize

# setup(ext_modules=cythonize('cy_geo.pyx')) # works on linux, but not on windows

ext = Extension(name="cython_geometry", sources=["cython_geometry.pyx"])
setup(ext_modules=cythonize(ext, language_level="3", annotate=True))