"""Build configuration for C extension module."""

import sys

import numpy as np
from setuptools import Extension, setup

if sys.platform == "win32":
    extra_compile_args = ["/O2"]
else:
    extra_compile_args = ["-O2", "-std=c99"]

ext_modules = [
    Extension(
        name="berryeval.metrics._native",
        sources=[
            "native/src/metrics.c",
            "native/bindings/module.c",
        ],
        include_dirs=[
            "native/src",
            np.get_include(),
        ],
        extra_compile_args=extra_compile_args,
    ),
]

setup(ext_modules=ext_modules)
