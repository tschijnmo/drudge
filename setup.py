"""Setup script for drudge."""

import os.path

from setuptools import setup, find_packages, Extension

with open('README.rst', 'r') as readme:
    DESCRIPTION = readme.read()

CLASSIFIERS = [
    'Development Status :: 1 - Planning',
    'Intended Audience :: Developers',
    'Intended Audience :: Science/Research',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3 :: Only',
    'Programming Language :: Python :: Implementation :: CPython',
    'Topic :: Scientific/Engineering :: Mathematics'
]

PROJ_ROOT = os.path.dirname(os.path.abspath(__file__))
INCLUDE_DIRS = [
    '/'.join([PROJ_ROOT, i])
    for i in ['deps/libcanon/include', 'drudge']
    ]
COMPILE_FLAGS = ['-std=c++14']

canonpy = Extension(
    'drudge.canonpy',
    ['drudge/canonpy.cpp'],
    include_dirs=INCLUDE_DIRS,
    extra_compile_args=COMPILE_FLAGS
)

wickcore = Extension(
    'drudge.wickcore',
    ['drudge/wickcore.cpp'],
    include_dirs=INCLUDE_DIRS,
    extra_compile_args=COMPILE_FLAGS
)

setup(
    name='drudge',
    version='0.4.0dev',
    description=DESCRIPTION.splitlines()[0],
    long_description=DESCRIPTION,
    url='https://github.com/tschijnmo/drudge',
    author='Jinmo Zhao and Gustavo E Scuseria',
    author_email='tschijnmotschau@gmail.com',
    license='MIT',
    classifiers=CLASSIFIERS,
    packages=find_packages(),
    ext_modules=[canonpy, wickcore],
    package_data={'drudge': ['templates/*']},
    install_requires=['sympy', 'ipython', 'Jinja2']
)
