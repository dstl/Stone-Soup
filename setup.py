#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

from stonesoup import __version__ as version

setup(name='stonesoup',
      version=version,
      description='A target tracking development/testing framework',
      url='https://github.com/dstl/Stone-Soup',
      packages=find_packages(exclude=('docs', '*.tests')),
      install_requires=['ruamel.yaml>=0.15.45', 'scipy', 'matplotlib',
                        'lxml', 'simplekml>=1.3.1'],
      extras_require={
          'dev': [
              'pytest-flake8', 'pytest-cov', 'Sphinx', 'sphinx_rtd_theme',
              'setuptools>=30', 'pymap3d'],
      },
      )
