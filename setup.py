"""Setup script for drudge."""

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

canonpy = Extension(
    'drudge.canonpy',
    ['drudge/canonpy.cpp'],
    include_dirs=['deps/libcanon/include', 'drudge/'],
    extra_compile_args=['-std=c++14']
)

setup(
    name='drudge',
    version='0.1.0dev',
    description=DESCRIPTION.splitlines()[0],
    long_description=DESCRIPTION,
    url='https://github.com/tschijnmo/drudge',
    author='Jinmo Zhao and Gustavo E Scuseria',
    author_email='tschijnmotschau@gmail.com',
    license='MIT',
    classifiers=CLASSIFIERS,
    packages=find_packages(),
    ext_modules=[canonpy]
)
