from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext
import numpy as np

ext_modules = [
    Pybind11Extension(
        "physics.gwphys",
        ["gwphys.cpp"],
        include_dirs=[np.get_include()],
        libraries=["fftw3"],                 # precisa do libfftw3-dev
        extra_compile_args=["-O3", "-march=native"],
    ),
]

setup(
    name="physics",
    version="0.1.0",
    packages=["physics"],
    package_dir={"physics": "."},
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
)