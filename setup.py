from setuptools import setup, Extension
import pybind11

ext_modules = [
    Extension(
        "foodie_bagger",
        ["foodie_bagger.cpp"],
        include_dirs=[
            pybind11.get_include(),
        ],
        language="c++",
        extra_compile_args=["-std=c++11"],
        extra_link_args=[
            "-static-libgcc",
            "-static-libstdc++"
        ]
    ),
]

setup(
    name="foodie_bagger",
    ext_modules=ext_modules,
)