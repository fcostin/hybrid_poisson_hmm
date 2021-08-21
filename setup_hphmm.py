from setuptools import find_packages, setup, Extension
from Cython.Build import cythonize

setup(
    name = "hphmm",
    packages = find_packages("lib/hphmm"),
    package_dir = {"": "lib"},
    package_data = {},
    python_requires=">=3.5",
    ext_modules = cythonize(
        [
            Extension(
                "hphmm.libhphmm",
                ["lib/hphmm/libhphmm.pyx"],
                define_macros=[],
            ),
        ],
        compiler_directives={
                    'language_level' : "3", # Py3.
        },
        # Generate an annotated report of the cython code highlighting
        # points where interactions with the python interpreter occur.
        annotate=True,
    ),

)
