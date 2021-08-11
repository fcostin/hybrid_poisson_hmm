from setuptools import find_packages, setup, Extension
from Cython.Build import cythonize

setup(
    packages = find_packages("lib"),
    package_dir = {"": "lib"},
    package_data = {},
    ext_modules = cythonize(
        [
            Extension(
                "gamma_approx._gammaapprox",
                ["lib/gamma_approx/_gammaapprox.pyx"],
                define_macros=[],
            ),
        ],
        compiler_directives={
                    'language_level' : "3", # Py3.
        },
        # Generate a an annotated report of the cython code highlighting
        # points where interactions with the python interpreter occur.
        annotate=True,
    ),

)